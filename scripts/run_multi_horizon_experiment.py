#!/usr/bin/env python3
"""Run complete multi-horizon experiment: finetune then predict.

This script runs the complete experiment pipeline:
1. Finetune models for 3 different action horizons
2. Run predictions for each model over each horizon (9 runs total)
"""

from absl import app, flags, logging
import subprocess
import sys
import os

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
flags.DEFINE_string("base_save_dir", None, "Base directory for saving finetuning checkpoints.")
flags.DEFINE_string("output_dir", None, "Directory to save prediction results.")
flags.DEFINE_integer("batch_size", 32, "Batch size for finetuning.")
flags.DEFINE_list(
    "action_horizons", 
    ["1", "4", "8"], 
    "List of action horizons to finetune models for."
)
flags.DEFINE_integer("num_steps", 1000, "Number of training steps.")
flags.DEFINE_integer("num_trajectories", 10, "Number of trajectories for predictions.")
flags.DEFINE_bool(
    "skip_finetuning",
    False,
    "Skip finetuning step if models already exist.",
)
flags.DEFINE_bool(
    "skip_predictions",
    False,
    "Skip prediction step.",
)


def main(_):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Finetune models
    if not FLAGS.skip_finetuning:
        logging.info("="*60)
        logging.info("STEP 1: Finetuning models for multiple horizons")
        logging.info("="*60)
        
        finetune_cmd = [
            sys.executable,
            os.path.join(script_dir, "multi_horizon_finetune.py"),
            f"--pretrained_path={FLAGS.pretrained_path}",
            f"--data_dir={FLAGS.data_dir}",
            f"--dataset_name={FLAGS.dataset_name}",
            f"--base_save_dir={FLAGS.base_save_dir}",
            f"--batch_size={FLAGS.batch_size}",
            f"--action_horizons={','.join(FLAGS.action_horizons)}",
            f"--num_steps={FLAGS.num_steps}",
        ]
        
        logging.info(f"Running: {' '.join(finetune_cmd)}")
        result = subprocess.run(finetune_cmd, check=True)
        if result.returncode != 0:
            logging.error("Finetuning failed!")
            return
        logging.info("Finetuning complete!\n")
    else:
        logging.info("Skipping finetuning step.")
    
    # Step 2: Run predictions
    if not FLAGS.skip_predictions:
        logging.info("="*60)
        logging.info("STEP 2: Running predictions for all model/horizon combinations")
        logging.info("="*60)
        
        predict_cmd = [
            sys.executable,
            os.path.join(script_dir, "multi_horizon_predict.py"),
            f"--base_model_dir={FLAGS.base_save_dir}",
            f"--data_dir={FLAGS.data_dir}",
            f"--dataset_name={FLAGS.dataset_name}",
            f"--training_horizons={','.join(FLAGS.action_horizons)}",
            f"--prediction_horizons={','.join(FLAGS.action_horizons)}",
            f"--num_trajectories={FLAGS.num_trajectories}",
            f"--output_dir={FLAGS.output_dir}",
        ]
        
        logging.info(f"Running: {' '.join(predict_cmd)}")
        result = subprocess.run(predict_cmd, check=True)
        if result.returncode != 0:
            logging.error("Prediction failed!")
            return
        logging.info("Predictions complete!\n")
    else:
        logging.info("Skipping prediction step.")
    
    logging.info("="*60)
    logging.info("EXPERIMENT COMPLETE!")
    logging.info("="*60)
    logging.info(f"Models saved to: {FLAGS.base_save_dir}")
    logging.info(f"Predictions saved to: {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)

