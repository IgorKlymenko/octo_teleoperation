#!/usr/bin/env python3
"""Visualize predicted trajectories from multi-horizon prediction results.

This script loads saved predictions and creates visualizations comparing
predicted actions to ground truth actions.
"""

# CRITICAL: Set matplotlib backend BEFORE any other imports
# This must be the first thing after the docstring to prevent matplotlib
# from reading an invalid MPLBACKEND environment variable
import os
# Delete invalid backend setting if it exists (Jupyter notebook backends)
if 'MPLBACKEND' in os.environ:
    backend_value = os.environ.get('MPLBACKEND', '')
    # If it's the problematic Jupyter backend, remove it
    if 'matplotlib_inline' in backend_value or 'inline' in backend_value:
        del os.environ['MPLBACKEND']

from absl import app, flags, logging
import matplotlib
# Explicitly set backend after import but before pyplot
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "results_dir",
    None,
    "Directory containing prediction results (should have subdirs like train_h1_pred_h5/).",
)
flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to save visualization plots.",
)
flags.DEFINE_list(
    "action_dim_labels",
    ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    "Labels for action dimensions.",
)
flags.DEFINE_integer(
    "max_trajectories",
    5,
    "Maximum number of trajectories to visualize per model/horizon combination.",
)
flags.DEFINE_bool(
    "save_plots",
    True,
    "Whether to save plots to files.",
)
flags.DEFINE_bool(
    "show_plots",
    False,
    "Whether to display plots interactively.",
)


def visualize_trajectory(
    predictions,
    ground_truth,
    action_dim_labels,
    title="Trajectory Comparison",
    save_path=None,
    show=False,
):
    """Create a visualization comparing predicted and ground truth actions.
    
    Args:
        predictions: Array of shape (traj_len, action_horizon, action_dim) or (traj_len, action_dim)
        ground_truth: Array of shape (traj_len, window_size, action_horizon, action_dim) or (traj_len, action_horizon, action_dim) or (traj_len, action_dim)
        action_dim_labels: List of labels for each action dimension
        title: Title for the plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Handle different shapes
    # Ground truth might have window_size dimension: (traj_len, window_size, action_horizon, action_dim)
    if ground_truth.ndim == 4:
        ground_truth = ground_truth[:, 0, :, :]  # Remove window_size dimension: (traj_len, action_horizon, action_dim)
    
    # Handle action_horizon dimension - take first action from horizon
    if predictions.ndim == 3:
        predictions = predictions[:, 0, :]  # Take first action from horizon: (traj_len, action_dim)
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[:, 0, :]  # Take first action from horizon: (traj_len, action_dim)
    
    traj_len, action_dim = predictions.shape
    n_dims = min(action_dim, len(action_dim_labels))
    
    # Create subplots
    fig, axes = plt.subplots(
        n_dims, 1, figsize=(12, 3 * n_dims), sharex=True
    )
    if n_dims == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time_steps = np.arange(traj_len)
    
    for dim in range(n_dims):
        ax = axes[dim]
        ax.plot(
            time_steps,
            ground_truth[:, dim],
            label="Ground Truth",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            time_steps,
            predictions[:, dim],
            label="Predicted",
            linewidth=2,
            alpha=0.7,
            linestyle="--",
        )
        
        # Add trendline for predicted actions (moving average)
        if traj_len > 5:
            window_size = max(5, traj_len // 20)  # Adaptive window size
            if window_size % 2 == 0:
                window_size += 1  # Make odd for symmetric smoothing
            # Use convolution for moving average
            kernel = np.ones(window_size) / window_size
            trendline = np.convolve(predictions[:, dim], kernel, mode='same')
            ax.plot(
                time_steps,
                trendline,
                label="Predicted Trend",
                linewidth=2.5,
                alpha=0.9,
                linestyle="-",
                color='red',
            )
        
        ax.set_ylabel(action_dim_labels[dim], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        
        if dim == n_dims - 1:
            ax.set_xlabel("Timestep", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_trajectory_summary(
    predictions,
    ground_truth,
    action_dim_labels,
    title="Trajectory Summary",
    save_path=None,
    show=False,
):
    """Create a summary visualization with all action dimensions in a grid.
    
    Args:
        predictions: Array of shape (traj_len, action_horizon, action_dim) or (traj_len, action_dim)
        ground_truth: Array of shape (traj_len, window_size, action_horizon, action_dim) or (traj_len, action_horizon, action_dim) or (traj_len, action_dim)
        action_dim_labels: List of labels for each action dimension
        title: Title for the plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Handle different shapes
    # Ground truth might have window_size dimension
    if ground_truth.ndim == 4:
        ground_truth = ground_truth[:, 0, :, :]  # Remove window_size dimension
    
    # Handle action_horizon dimension
    if predictions.ndim == 3:
        predictions = predictions[:, 0, :]
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[:, 0, :]
    
    traj_len, action_dim = predictions.shape
    n_dims = min(action_dim, len(action_dim_labels))
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(n_dims + 1)))  # +1 for MSE plot
    
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size)
    )
    axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    time_steps = np.arange(traj_len)
    
    # Plot MSE over time
    mse = np.mean((predictions - ground_truth) ** 2, axis=1)
    axes[0].plot(time_steps, mse, linewidth=2, color='red')
    axes[0].set_ylabel("MSE", fontsize=10)
    axes[0].set_title("Mean Squared Error", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot each action dimension
    for dim in range(n_dims):
        ax = axes[dim + 1]
        ax.plot(
            time_steps,
            ground_truth[:, dim],
            label="Ground Truth",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            time_steps,
            predictions[:, dim],
            label="Predicted",
            linewidth=2,
            alpha=0.7,
            linestyle="--",
        )
        
        # Add trendline for predicted actions (moving average)
        if traj_len > 5:
            window_size = max(5, traj_len // 20)  # Adaptive window size
            if window_size % 2 == 0:
                window_size += 1  # Make odd for symmetric smoothing
            # Use convolution for moving average
            kernel = np.ones(window_size) / window_size
            trendline = np.convolve(predictions[:, dim], kernel, mode='same')
            ax.plot(
                time_steps,
                trendline,
                label="Predicted Trend",
                linewidth=2.5,
                alpha=0.9,
                linestyle="-",
                color='red',
            )
        
        ax.set_ylabel(action_dim_labels[dim], fontsize=10)
        ax.set_title(f"Action Dim {dim}: {action_dim_labels[dim]}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    
    # Hide unused subplots
    for i in range(n_dims + 1, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved summary plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main(_):
    results_dir = FLAGS.results_dir
    output_dir = FLAGS.output_dir or results_dir
    
    if not os.path.exists(results_dir):
        logging.error(f"Results directory {results_dir} does not exist!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all result subdirectories
    result_dirs = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("train_h")
    ]
    
    if not result_dirs:
        logging.error(f"No result directories found in {results_dir}")
        return
    
    logging.info(f"Found {len(result_dirs)} result directories")
    
    # Process each result directory
    for result_dir_name in sorted(result_dirs):
        result_path = os.path.join(results_dir, result_dir_name)
        logging.info(f"\nProcessing {result_dir_name}...")
        
        # Load predictions
        predictions_file = os.path.join(result_path, "predictions.npz")
        if not os.path.exists(predictions_file):
            logging.warning(f"Predictions file not found: {predictions_file}")
            continue
        
        data = np.load(predictions_file)
        predictions = data["predictions"]
        ground_truth = data["ground_truth"]
        
        logging.info(f"  Predictions shape: {predictions.shape}")
        logging.info(f"  Ground truth shape: {ground_truth.shape}")
        
        # Load metrics
        metrics_file = os.path.join(result_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            logging.info(f"  MSE: {metrics.get('mse', 'N/A'):.6f}")
            logging.info(f"  MAE: {metrics.get('mae', 'N/A'):.6f}")
        
        # Create visualization directory
        vis_dir = os.path.join(output_dir, result_dir_name, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Determine number of trajectories to visualize
        # Predictions shape: (timesteps, action_horizon, action_dim) or (timesteps, action_dim)
        # Ground truth shape: (timesteps, window_size, action_horizon, action_dim) or similar
        
        # Check if this is a single long trajectory or multiple trajectories
        # If first dimension is large (>100) and second is small (<20), it's likely a single trajectory
        # Otherwise, it might be multiple trajectories
        
        pred_shape = predictions.shape
        gt_shape = ground_truth.shape
        
        # Handle ground truth window_size dimension
        if len(gt_shape) == 4:
            # (timesteps, window_size, action_horizon, action_dim)
            gt_timesteps = gt_shape[0]
        elif len(gt_shape) == 3:
            # (timesteps, action_horizon, action_dim)
            gt_timesteps = gt_shape[0]
        else:
            gt_timesteps = gt_shape[0]
        
        # For predictions: (timesteps, action_horizon, action_dim) or (timesteps, action_dim)
        pred_timesteps = pred_shape[0]
        
        # If we have many timesteps, treat as a single long trajectory
        # Otherwise, might be multiple trajectories
        if pred_timesteps > 100:
            # Single long trajectory - visualize segments of it
            num_trajs = FLAGS.max_trajectories
            traj_len = pred_timesteps
            segment_size = traj_len // num_trajs if num_trajs > 1 else traj_len
        else:
            # Might be multiple trajectories
            num_trajs = min(pred_timesteps, FLAGS.max_trajectories)
            traj_len = 1  # Each is a single timestep
            segment_size = 1
        
        logging.info(f"  Visualizing {num_trajs} trajectory segment(s) from {pred_timesteps} total timesteps")
        
        # Visualize trajectory segments
        for traj_idx in range(num_trajs):
            if num_trajs > 1 and pred_timesteps > 100:
                # Extract a segment from the long trajectory
                start_idx = traj_idx * segment_size
                end_idx = min(start_idx + segment_size, pred_timesteps)
                pred_traj = predictions[start_idx:end_idx]
                
                # Handle ground truth shape
                if len(gt_shape) == 4:
                    gt_traj = ground_truth[start_idx:end_idx, 0, :, :]  # Remove window_size
                elif len(gt_shape) == 3:
                    gt_traj = ground_truth[start_idx:end_idx]
                else:
                    gt_traj = ground_truth[start_idx:end_idx]
                
                traj_suffix = f"_segment{traj_idx}"
            else:
                # Use full trajectory
                pred_traj = predictions
                
                # Handle ground truth shape
                if len(gt_shape) == 4:
                    gt_traj = ground_truth[:, 0, :, :]  # Remove window_size
                else:
                    gt_traj = ground_truth
                
                traj_suffix = ""
            
            # Individual trajectory plot
            save_path = os.path.join(vis_dir, f"trajectory{traj_suffix}.png")
            visualize_trajectory(
                pred_traj,
                gt_traj,
                FLAGS.action_dim_labels,
                title=f"{result_dir_name}{traj_suffix}",
                save_path=save_path if FLAGS.save_plots else None,
                show=FLAGS.show_plots,
            )
            
            # Summary plot
            save_path_summary = os.path.join(vis_dir, f"summary{traj_suffix}.png")
            visualize_trajectory_summary(
                pred_traj,
                gt_traj,
                FLAGS.action_dim_labels,
                title=f"{result_dir_name}{traj_suffix} - Summary",
                save_path=save_path_summary if FLAGS.save_plots else None,
                show=FLAGS.show_plots,
            )
    
    logging.info(f"\nVisualization complete! Plots saved to {output_dir}")


if __name__ == "__main__":
    app.run(main)

