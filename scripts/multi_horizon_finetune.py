#!/usr/bin/env python3
"""Finetune models for multiple action horizons.

This script finetunes the teleoperation model for 3 different action horizons
and saves each model to a separate directory.
"""

from absl import app, flags, logging
import jax
import optax
import tensorflow as tf
import tqdm
import os
from pathlib import Path

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.model.components.action_heads import TeleoperationActionHead
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)
from clu import metric_writers

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string(
    "dataset_name",
    "aloha_sim_cube_scripted_dataset",
    "Name of the dataset to use (must match your dataset registration).",
)
flags.DEFINE_string(
    "image_obs_key",
    "top",
    "Key name in the raw dataset for image observations (e.g., 'top', 'wrist').",
)
flags.DEFINE_string(
    "proprio_obs_key",
    "state",
    "Key for proprioceptive observations in the dataset.",
)
flags.DEFINE_string(
    "language_key",
    "language_instruction",
    "Key for language instructions in the dataset.",
)
flags.DEFINE_string("base_save_dir", None, "Base directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 32, "Batch size for finetuning.")
flags.DEFINE_list(
    "action_horizons", 
    ["1", "4", "8"], 
    "List of action horizons to finetune models for."
)
flags.DEFINE_integer("num_steps", 1000, "Number of training steps.")
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)


def finetune_for_horizon(
    pretrained_model,
    data_dir,
    dataset_name,
    image_obs_key,
    proprio_obs_key,
    language_key,
    save_dir,
    action_horizon,
    batch_size,
    num_steps,
    freeze_transformer,
):
    """Finetune a model for a specific action horizon."""
    
    logging.info(f"Starting finetuning for action_horizon={action_horizon}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    tb_writer = metric_writers.create_default_writer(
        logdir=os.path.join(os.path.abspath(save_dir), "tb")
    )
    
    # Make finetuning dataset with specified action horizon
    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name=dataset_name,
            data_dir=data_dir,
            image_obs_keys={"primary": image_obs_key},  # Maps new name "primary" to dataset key
            proprio_obs_key=proprio_obs_key,
            language_key=language_key,
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=action_horizon,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)
        .batch(batch_size)
        .iterator()
    )

    # Process text
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # Get horizon and action dimensions
    first_obs = next(iter(example_batch["observation"].values()))
    horizon = int(first_obs.shape[1])
    
    action_shape = example_batch["action"].shape
    actual_action_horizon = int(action_shape[2])
    actual_action_dim = int(action_shape[3])
    logging.info(
        f"Detected action shape: {action_shape}, "
        f"action_horizon={actual_action_horizon}, action_dim={actual_action_dim}"
    )
    
    assert actual_action_horizon == action_horizon, (
        f"Dataset action_horizon ({actual_action_horizon}) doesn't match "
        f"requested action_horizon ({action_horizon})"
    )

    # Replace action head and update max_horizon
    config = pretrained_model.config.copy()
    config["model"]["heads"]["action"] = ModuleSpec.create(
        TeleoperationActionHead,
        action_horizon=actual_action_horizon,
        action_dim=actual_action_dim,
        readout_key="readout_action",
        use_map=True,
        min_cov_diag=1e-3,
    )
    config["model"]["max_horizon"] = max(config["model"]["max_horizon"], horizon)

    # Initialize model
    logging.info("Updating model for new action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)

    # Create optimizer & train_state
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config.get("optimizer", {}).get("frozen_keys", [])
    if freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # Define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # Run finetuning loop
    logging.info(f"Starting finetuning for {num_steps} steps...")
    for i in tqdm.tqdm(range(num_steps), total=num_steps, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            logging.info(f"Step {i+1}: {update_info}")
        if (i + 1) % 1000 == 0 or (i + 1) == num_steps:
            # Save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=save_dir)

        if (i + 1) % 10 == 0:
            info_host = jax.device_get(update_info)
            scalars = {}
            for k, v in info_host.items():
                if hasattr(v, "shape") and v.shape != ():
                    continue
                try:
                    scalars[k] = float(v)
                except Exception:
                    pass
            tb_writer.write_scalars(i + 1, scalars)
    
    # Final save
    train_state.model.save_pretrained(step=num_steps, checkpoint_path=save_dir)
    logging.info(f"Finetuning complete for action_horizon={action_horizon}. Model saved to {save_dir}")


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    # Load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # Convert horizons to integers
    action_horizons = [int(h) for h in FLAGS.action_horizons]
    logging.info(f"Will finetune models for action horizons: {action_horizons}")

    # Finetune for each horizon
    for action_horizon in action_horizons:
        save_dir = os.path.join(FLAGS.base_save_dir, f"horizon_{action_horizon}")
        finetune_for_horizon(
            pretrained_model=pretrained_model,
            data_dir=FLAGS.data_dir,
            dataset_name=FLAGS.dataset_name,
            image_obs_key=FLAGS.image_obs_key,
            proprio_obs_key=FLAGS.proprio_obs_key,
            language_key=FLAGS.language_key,
            save_dir=save_dir,
            action_horizon=action_horizon,
            batch_size=FLAGS.batch_size,
            num_steps=FLAGS.num_steps,
            freeze_transformer=FLAGS.freeze_transformer,
        )
        logging.info(f"Completed finetuning for horizon {action_horizon}")

    logging.info("All finetuning complete!")


if __name__ == "__main__":
    app.run(main)

