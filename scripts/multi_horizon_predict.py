#!/usr/bin/env python3
"""Run predictions for multiple models over multiple horizons.

This script loads models finetuned for different action horizons and runs
predictions for each model over each of the prediction horizons.
Total runs: num_training_horizons * num_prediction_horizons
"""

from absl import app, flags, logging
import jax
import numpy as np
import os
import json
from pathlib import Path
import tensorflow as tf

from octo.data.dataset import make_single_dataset
from octo.model.octo_model import OctoModel
from octo.model.components.action_heads import TeleoperationActionHead
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "base_model_dir",
    None,
    "Base directory containing finetuned models (should have subdirs horizon_1, horizon_4, horizon_8, etc.)",
)
flags.DEFINE_string("data_dir", None, "Path to dataset for running predictions.")
flags.DEFINE_list(
    "training_horizons",
    ["1", "4", "8"],
    "List of action horizons the models were trained on.",
)
flags.DEFINE_list(
    "prediction_horizons",
    ["1", "4", "8"],
    "List of action horizons to run predictions for.",
)
flags.DEFINE_integer("num_trajectories", 10, "Number of trajectories to predict on.")
flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to save prediction results.",
)
flags.DEFINE_integer("batch_size", 1, "Batch size for predictions.")


def run_predictions_for_model(
    model_path,
    training_horizon,
    prediction_horizon,
    data_dir,
    num_trajectories,
    batch_size,
    output_dir,
):
    """Run predictions for a specific model and prediction horizon."""
    
    logging.info(
        f"Running predictions: model (horizon={training_horizon}) -> "
        f"predict (horizon={prediction_horizon})"
    )
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    model = OctoModel.load_pretrained(model_path)
    
    # Create dataset with prediction horizon
    logging.info("Loading dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=prediction_horizon,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=False,  # Use validation/test split
    )
    
    # Get dataset iterator
    data_iter = (
        dataset.take(num_trajectories)
        .batch(batch_size)
        .iterator()
    )
    
    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    all_observations = []
    
    rng = jax.random.PRNGKey(42)
    
    logging.info(f"Running predictions on {num_trajectories} trajectories...")
    for batch_idx, batch in enumerate(data_iter):
        # Process observations
        observations = batch["observation"]
        
        # Create tasks (using language instructions from batch if available)
        if "task" in batch and "language" in batch["task"]:
            tasks = model.create_tasks(texts=batch["task"]["language"].tolist())
        else:
            # Default task if no language instruction
            tasks = model.create_tasks(texts=["perform task"])
        
        # Sample actions
        rng, sample_rng = jax.random.split(rng)
        predicted_actions = model.sample_actions(
            observations,
            tasks,
            rng=sample_rng,
            train=False,
        )
        
        # Get ground truth actions
        ground_truth_actions = batch["action"]
        
        # Store results
        all_predictions.append(np.array(predicted_actions))
        all_ground_truth.append(np.array(ground_truth_actions))
        all_observations.append({
            k: np.array(v) for k, v in observations.items()
        })
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"Processed {batch_idx + 1} batches")
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    # Compute metrics
    mse = np.mean((predictions - ground_truth) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth))
    
    # Save results
    results = {
        "training_horizon": training_horizon,
        "prediction_horizon": prediction_horizon,
        "num_trajectories": num_trajectories,
        "mse": float(mse),
        "mae": float(mae),
        "predictions_shape": list(predictions.shape),
        "ground_truth_shape": list(ground_truth.shape),
    }
    
    # Create output directory
    output_subdir = os.path.join(
        output_dir,
        f"train_h{training_horizon}_pred_h{prediction_horizon}"
    )
    os.makedirs(output_subdir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_subdir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions and ground truth
    np.savez(
        os.path.join(output_subdir, "predictions.npz"),
        predictions=predictions,
        ground_truth=ground_truth,
    )
    
    logging.info(
        f"Results saved to {output_subdir}\n"
        f"  MSE: {mse:.6f}\n"
        f"  MAE: {mae:.6f}"
    )
    
    return results


def main(_):
    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")
    
    # Convert horizons to integers
    training_horizons = [int(h) for h in FLAGS.training_horizons]
    prediction_horizons = [int(h) for h in FLAGS.prediction_horizons]
    
    logging.info(f"Training horizons: {training_horizons}")
    logging.info(f"Prediction horizons: {prediction_horizons}")
    logging.info(f"Total runs: {len(training_horizons) * len(prediction_horizons)}")
    
    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # Run predictions for each combination
    all_results = []
    run_count = 0
    
    for train_horizon in training_horizons:
        model_path = os.path.join(FLAGS.base_model_dir, f"horizon_{train_horizon}")
        
        if not os.path.exists(model_path):
            logging.warning(f"Model path {model_path} does not exist. Skipping...")
            continue
        
        for pred_horizon in prediction_horizons:
            run_count += 1
            logging.info(f"\n{'='*60}")
            logging.info(f"Run {run_count}/{len(training_horizons) * len(prediction_horizons)}")
            logging.info(f"{'='*60}")
            
            results = run_predictions_for_model(
                model_path=model_path,
                training_horizon=train_horizon,
                prediction_horizon=pred_horizon,
                data_dir=FLAGS.data_dir,
                num_trajectories=FLAGS.num_trajectories,
                batch_size=FLAGS.batch_size,
                output_dir=FLAGS.output_dir,
            )
            all_results.append(results)
    
    # Save summary
    summary_path = os.path.join(FLAGS.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    logging.info("\n" + "="*60)
    logging.info("SUMMARY")
    logging.info("="*60)
    logging.info(f"{'Train H':<10} {'Pred H':<10} {'MSE':<15} {'MAE':<15}")
    logging.info("-" * 60)
    for r in all_results:
        logging.info(
            f"{r['training_horizon']:<10} "
            f"{r['prediction_horizon']:<10} "
            f"{r['mse']:<15.6f} "
            f"{r['mae']:<15.6f}"
        )
    logging.info("="*60)
    logging.info(f"All results saved to {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)

