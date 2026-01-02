#!/usr/bin/env python3
"""Run predictions for multiple models over multiple horizons.

This script loads models finetuned for different action horizons and runs
predictions for each model over each of the prediction horizons.
Total runs: num_training_horizons * num_prediction_horizons
"""

from absl import app, flags, logging
import jax
import jax.numpy as jnp
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
flags.DEFINE_integer(
    "n_cov_samples",
    128,
    "Number of action samples to estimate the covariance (per timestep).",
)
flags.DEFINE_integer(
    "control_dof_k",
    2,
    "Rank k of the basis used in the linear-system reconstruction test.",
)
flags.DEFINE_float(
    "cov_eps",
    1e-6,
    "Jitter for numerical stability when estimating covariance and eigendecomposing.",
)


def estimate_cov_from_samples(samples: np.ndarray, eps: float) -> np.ndarray:
    """
    samples: (n_samples, action_dim)
    returns: (action_dim, action_dim) covariance
    """
    X = samples - samples.mean(axis=0, keepdims=True)
    # Unbiased covariance (n-1)
    cov = (X.T @ X) / max(1, (X.shape[0] - 1))
    cov = cov + eps * np.eye(cov.shape[0])
    return cov


def lowrank_basis_from_cov(cov: np.ndarray, k: int, eps: float) -> np.ndarray:
    """
    cov: (d, d) SPD
    returns: B = U_k * sqrt(Lambda_k) of shape (d, k)
    """
    # eigh returns ascending eigenvalues
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, a_min=0.0, a_max=None)

    k = min(k, cov.shape[0])
    top_idx = np.argsort(evals)[-k:]  # indices of top-k eigenvalues
    U = evecs[:, top_idx]             # (d, k)
    lam = evals[top_idx]              # (k,)
    B = U * np.sqrt(lam + eps)[None, :]  # (d, k)
    return B


def reconstruct_actions_linear_system(B: np.ndarray, a_true: np.ndarray) -> np.ndarray:
    """
    B: (d, k)
    a_true: (d,)
    returns: a_hat = B z*, where z* solves least squares
    """
    z, _, _, _ = np.linalg.lstsq(B, a_true, rcond=None)
    a_hat = B @ z
    return a_hat


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
    
    # Inspect model's expected observation format
    example_obs = model.example_batch["observation"]
    logging.info(f"Model expects observations with keys: {list(example_obs.keys())}")
    for key, value in example_obs.items():
        if hasattr(value, 'shape'):
            logging.info(f"  {key}: shape {value.shape}")
        else:
            logging.info(f"  {key}: {type(value)}")
    
    # Get expected window_size from example batch
    first_obs_value = next(iter(example_obs.values()))
    if hasattr(first_obs_value, 'shape') and len(first_obs_value.shape) >= 2:
        expected_window_size = first_obs_value.shape[1]
        logging.info(f"Model expects window_size={expected_window_size}")
    else:
        expected_window_size = 1
        logging.warning("Could not determine expected window_size, defaulting to 1")
    
    # Create dataset with TRAINING horizon to match model's expected observation shapes
    # The model was trained with training_horizon, so observations must match that
    logging.info("Loading dataset with training horizon to match model expectations...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=expected_window_size,  # Match model's expected window_size
            action_horizon=training_horizon,  # Use training horizon for dataset
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=False,  # Use validation/test split
    )
    
    # Also create dataset with prediction horizon for ground truth comparison
    logging.info("Loading dataset with prediction horizon for ground truth...")
    dataset_pred = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=expected_window_size,  # Match model's expected window_size
            action_horizon=prediction_horizon,  # Use prediction horizon for ground truth
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=False,
    )
    
    # Get dataset iterators - unbatch trajectories to process timestep by timestep
    # The model expects (batch, window_size, ...) but dataset returns (batch, traj_len, window_size, ...)
    data_iter = (
        dataset.take(num_trajectories)
        .unbatch()  # Unbatch trajectories: (traj_len, window_size, ...)
        .batch(batch_size)  # Batch timesteps: (batch, window_size, ...)
        .iterator()
    )
    
    data_iter_pred = (
        dataset_pred.take(num_trajectories)
        .unbatch()  # Unbatch trajectories
        .batch(batch_size)  # Batch timesteps
        .iterator()
    )
    
    # Note: We assume both datasets produce trajectories in the same order
    # since they use the same data source and configuration (except action_horizon)
    
    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    all_observations = []
    
    # Collect reconstruction metrics across all batches
    all_recon_mse = []
    all_recon_mae = []
    
    rng = jax.random.PRNGKey(42)
    
    # Get parameters for covariance estimation and reconstruction
    n_samples = FLAGS.n_cov_samples
    k = FLAGS.control_dof_k
    cov_eps = FLAGS.cov_eps
    
    logging.info(f"Running predictions on {num_trajectories} trajectories (processing timestep by timestep)...")
    logging.info(f"Using {n_samples} samples for covariance estimation, rank-{k} basis for reconstruction")
    for batch_idx, (batch, batch_pred) in enumerate(zip(data_iter, data_iter_pred)):
        # Process observations (use training horizon dataset for model input)
        # Now observations should be (batch, window_size, ...) instead of (batch, traj_len, window_size, ...)
        observations = batch["observation"]
        
        # Ensure observations match model's expected format
        # Filter to only include keys that the model expects
        expected_obs_keys = set(model.example_batch["observation"].keys())
        actual_obs_keys = set(observations.keys())
        
        # Check for missing keys
        missing_keys = expected_obs_keys - actual_obs_keys
        if missing_keys:
            logging.warning(f"Observations missing keys: {missing_keys}. This may cause errors.")
        
        # Filter observations to match expected format
        filtered_observations = {}
        for key in expected_obs_keys:
            if key in observations:
                filtered_observations[key] = observations[key]
            else:
                # Use example batch value as fallback (this might not work for all cases)
                logging.warning(f"Key {key} not in observations, using example batch value")
                example_value = model.example_batch["observation"][key]
                # Convert to numpy and broadcast to match batch size
                if hasattr(example_value, 'shape'):
                    example_np = np.array(example_value)
                    batch_size = list(observations.values())[0].shape[0] if observations else 1
                    # Broadcast to match batch size
                    filtered_observations[key] = np.broadcast_to(
                        example_np[None, ...],
                        (batch_size,) + example_np.shape
                    )
                else:
                    filtered_observations[key] = np.array(example_value) if hasattr(example_value, '__array__') else example_value
        
        observations = filtered_observations
        
        # Create tasks (using language instructions from batch if available)
        if "task" in batch and "language" in batch["task"]:
            tasks = model.create_tasks(texts=batch["task"]["language"].tolist())
        else:
            # Default task if no language instruction
            tasks = model.create_tasks(texts=["perform task"])
        
        # --- Estimate covariance from multiple action samples (per timestep) ---
        # Collect samples: (n_samples, batch, train_horizon, action_dim)
        samples_list = []
        rng_current = rng
        for _ in range(n_samples):
            rng_current, sample_rng = jax.random.split(rng_current)
            s = model.sample_actions(observations, tasks, rng=sample_rng, train=False)
            samples_list.append(np.array(s))
        rng = rng_current
        samples = np.stack(samples_list, axis=0)
        
        # We'll also keep one representative prediction (e.g., mean over samples) for your original MSE/MAE
        predicted_actions = samples.mean(axis=0)  # (batch, train_horizon, action_dim)
        
        # Get ground truth actions (from prediction horizon dataset)
        ground_truth_actions = np.array(batch_pred["action"])
        
        # Handle horizon mismatch: if training_horizon != prediction_horizon
        if training_horizon != prediction_horizon:
            if training_horizon < prediction_horizon:
                # Model predicts fewer steps - pad predictions or repeat last action
                # For now, just take first prediction_horizon steps from ground truth
                # and compare to repeated predictions
                pred_shape = predicted_actions.shape  # (*sample_shape, batch, train_horizon, action_dim)
                if len(pred_shape) == 3:  # (batch, train_horizon, action_dim)
                    batch_size, _, action_dim = pred_shape
                    # Repeat the last predicted action to match prediction horizon
                    last_pred = predicted_actions[:, -1:, :]  # (batch, 1, action_dim)
                    repeated = np.repeat(last_pred, prediction_horizon - training_horizon, axis=1)
                    predicted_actions = np.concatenate([predicted_actions, repeated], axis=1)
                else:
                    # Handle sample_shape case
                    last_pred = predicted_actions[..., -1:, :]
                    repeated = np.repeat(last_pred, prediction_horizon - training_horizon, axis=-2)
                    predicted_actions = np.concatenate([predicted_actions, repeated], axis=-2)
            else:
                # Model predicts more steps - truncate to prediction horizon
                predicted_actions = predicted_actions[..., :prediction_horizon, :]
        
        # --- Linear-system reconstruction test (low-rank basis) ---
        # For each batch item, estimate cov from samples at a single horizon step (shared cov in your head anyway),
        # then build B (d x k) and reconstruct each ground-truth action vector.
        batch_size_local = ground_truth_actions.shape[0]
        pred_h = ground_truth_actions.shape[1]
        action_dim = ground_truth_actions.shape[2]
        
        recon_mse_list = []
        recon_mae_list = []
        
        for b in range(batch_size_local):
            # Use horizon step 0 samples to estimate covariance of actions
            # samples shape: (n_samples, batch, train_horizon, action_dim)
            # If train_horizon != prediction_horizon, we still estimate covariance from available train_horizon samples.
            samples_b = samples[:, b, 0, :]  # (n_samples, action_dim)
            
            cov = estimate_cov_from_samples(samples_b, eps=cov_eps)
            B = lowrank_basis_from_cov(cov, k=k, eps=cov_eps)  # (action_dim, k)
            
            # Reconstruct each GT step in prediction horizon
            for h in range(pred_h):
                a_true = ground_truth_actions[b, h, :]  # (action_dim,)
                a_hat = reconstruct_actions_linear_system(B, a_true)
                
                recon_mse_list.append(np.mean((a_hat - a_true) ** 2))
                recon_mae_list.append(np.mean(np.abs(a_hat - a_true)))
        
        # Store reconstruction metrics for this batch
        if recon_mse_list:
            all_recon_mse.extend(recon_mse_list)
            all_recon_mae.extend(recon_mae_list)
        
        # Store results
        all_predictions.append(predicted_actions)
        all_ground_truth.append(ground_truth_actions)
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
    
    # Compute reconstruction metrics (averaged over all timesteps processed)
    recon_mse = float(np.mean(all_recon_mse)) if all_recon_mse else 0.0
    recon_mae = float(np.mean(all_recon_mae)) if all_recon_mae else 0.0
    
    # Save results
    results = {
        "training_horizon": training_horizon,
        "prediction_horizon": prediction_horizon,
        "num_trajectories": num_trajectories,
        "mse": float(mse),
        "mae": float(mae),
        "recon_mse": recon_mse,
        "recon_mae": recon_mae,
        "control_dof_k": k,
        "n_cov_samples": n_samples,
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
        f"  MAE: {mae:.6f}\n"
        f"  Recon MSE (basis-fit): {recon_mse:.6f}\n"
        f"  Recon MAE (basis-fit): {recon_mae:.6f}"
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
    logging.info("\n" + "="*80)
    logging.info("SUMMARY")
    logging.info("="*80)
    logging.info(f"{'Train H':<10} {'Pred H':<10} {'MSE':<15} {'MAE':<15} {'Recon MSE':<15} {'Recon MAE':<15}")
    logging.info("-" * 80)
    for r in all_results:
        logging.info(
            f"{r['training_horizon']:<10} "
            f"{r['prediction_horizon']:<10} "
            f"{r['mse']:<15.6f} "
            f"{r['mae']:<15.6f} "
            f"{r['recon_mse']:<15.6f} "
            f"{r['recon_mae']:<15.6f}"
        )
    logging.info("="*80)
    logging.info(f"All results saved to {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)

