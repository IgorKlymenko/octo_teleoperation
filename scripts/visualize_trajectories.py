#!/usr/bin/env python3
"""Visualize predicted trajectories from multi-horizon prediction results.

This script loads saved predictions and creates visualizations comparing
predicted actions to ground truth actions.
"""

from absl import app, flags, logging
import numpy as np
import matplotlib.pyplot as plt
import json
import os
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
        ground_truth: Array of shape (traj_len, action_horizon, action_dim) or (traj_len, action_dim)
        action_dim_labels: List of labels for each action dimension
        title: Title for the plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Handle different shapes - if action_horizon dimension exists, take first action
    if predictions.ndim == 3:
        predictions = predictions[:, 0, :]  # Take first action from horizon
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[:, 0, :]  # Take first action from horizon
    
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
        ground_truth: Array of shape (traj_len, action_horizon, action_dim) or (traj_len, action_dim)
        action_dim_labels: List of labels for each action dimension
        title: Title for the plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Handle different shapes
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
        if predictions.ndim >= 3:
            # Shape might be (num_trajs, traj_len, ...) or (traj_len, ...)
            if predictions.shape[0] > 1 and predictions.shape[1] < 1000:
                # Likely (num_trajs, traj_len, ...)
                num_trajs = min(predictions.shape[0], FLAGS.max_trajectories)
                traj_len = predictions.shape[1]
            else:
                # Likely (traj_len, ...) - single trajectory
                num_trajs = 1
                traj_len = predictions.shape[0]
        else:
            num_trajs = 1
            traj_len = predictions.shape[0]
        
        logging.info(f"  Visualizing {num_trajs} trajectory(ies)")
        
        # Visualize each trajectory
        for traj_idx in range(num_trajs):
            if num_trajs > 1:
                pred_traj = predictions[traj_idx]
                gt_traj = ground_truth[traj_idx]
                traj_suffix = f"_traj{traj_idx}"
            else:
                pred_traj = predictions
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

