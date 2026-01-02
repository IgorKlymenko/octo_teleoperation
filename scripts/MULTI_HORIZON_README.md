# Multi-Horizon Finetuning and Prediction Scripts

This directory contains scripts for finetuning teleoperation models for multiple action horizons and running predictions across different horizon combinations.

## Overview

The experiment consists of:
1. **Finetuning**: Train models for 3 different action horizons (e.g., 1, 4, 8)
2. **Prediction**: Run predictions for each model over each horizon (3 models × 3 horizons = 9 runs total)

## Scripts

### 1. `multi_horizon_finetune.py`
Finetunes models for multiple action horizons. Each horizon gets its own model saved in a separate directory.

**Usage:**
```bash
python scripts/multi_horizon_finetune.py \
    --pretrained_path=/path/to/pretrained/octo \
    --data_dir=/path/to/dataset \
    --base_save_dir=/path/to/save/models \
    --batch_size=32 \
    --action_horizons=1,4,8 \
    --num_steps=1000 \
    --freeze_transformer=false
```

**Output:**
- `base_save_dir/horizon_1/` - Model finetuned for horizon 1
- `base_save_dir/horizon_4/` - Model finetuned for horizon 4
- `base_save_dir/horizon_8/` - Model finetuned for horizon 8

### 2. `multi_horizon_predict.py`
Runs predictions for each model over each prediction horizon. Now includes **reconstruction metrics** that test whether the learned covariance geometry contains ground-truth actions.

**Usage:**
```bash
python scripts/multi_horizon_predict.py \
    --base_model_dir=/path/to/save/models \
    --data_dir=/path/to/dataset \
    --training_horizons=1,4,8 \
    --prediction_horizons=1,4,8 \
    --num_trajectories=10 \
    --batch_size=1 \
    --output_dir=/path/to/save/predictions \
    --n_cov_samples=128 \
    --control_dof_k=2 \
    --cov_eps=1e-6
```

**New Flags:**
- `--n_cov_samples`: Number of action samples to estimate covariance (default: 128)
- `--control_dof_k`: Rank k of the low-rank basis for reconstruction test (default: 2)
- `--cov_eps`: Numerical stability jitter (default: 1e-6)

**Output:**
- `output_dir/train_h1_pred_h1/` - Model trained on h1, predicting h1
- `output_dir/train_h1_pred_h4/` - Model trained on h1, predicting h4
- `output_dir/train_h1_pred_h8/` - Model trained on h1, predicting h8
- `output_dir/train_h4_pred_h1/` - Model trained on h4, predicting h1
- ... (9 total combinations)
- `output_dir/summary.json` - Summary of all results

Each prediction directory contains:
- `metrics.json` - MSE, MAE, reconstruction MSE/MAE, and other metrics
- `predictions.npz` - NumPy array with predictions and ground truth

**Metrics Explained:**
- **MSE/MAE**: Standard prediction error metrics
- **recon_mse/recon_mae**: Reconstruction error using low-rank basis from learned covariance
  - Low recon error → learned covariance geometry captures ground truth well
  - High recon error → learned basis misses critical components or is too low-rank

### 3. `run_multi_horizon_experiment.py`
Convenience script that runs both finetuning and prediction in sequence.

**Usage:**
```bash
python scripts/run_multi_horizon_experiment.py \
    --pretrained_path=/path/to/pretrained/octo \
    --data_dir=/path/to/dataset \
    --base_save_dir=/path/to/save/models \
    --output_dir=/path/to/save/predictions \
    --batch_size=32 \
    --action_horizons=1,4,8 \
    --num_steps=1000 \
    --num_trajectories=10
```

**Flags:**
- `--skip_finetuning`: Skip finetuning if models already exist
- `--skip_predictions`: Skip prediction step

## Example Workflow

```bash
# Step 1: Finetune models for 3 horizons
python scripts/multi_horizon_finetune.py \
    --pretrained_path=./pretrained_models/octo-small \
    --data_dir=./data/my_dataset \
    --base_save_dir=./checkpoints/multi_horizon \
    --action_horizons=1,4,8 \
    --num_steps=1000

# Step 2: Run predictions (9 runs total)
python scripts/multi_horizon_predict.py \
    --base_model_dir=./checkpoints/multi_horizon \
    --data_dir=./data/my_dataset \
    --training_horizons=1,4,8 \
    --prediction_horizons=1,4,8 \
    --num_trajectories=10 \
    --output_dir=./results/multi_horizon

# Or run both in one command:
python scripts/run_multi_horizon_experiment.py \
    --pretrained_path=./pretrained_models/octo-small \
    --data_dir=./data/my_dataset \
    --base_save_dir=./checkpoints/multi_horizon \
    --output_dir=./results/multi_horizon \
    --action_horizons=1,4,8 \
    --num_steps=1000 \
    --num_trajectories=10
```

## Results Structure

After running predictions, you'll have:

```
output_dir/
├── summary.json                    # Summary of all 9 runs
├── train_h1_pred_h1/
│   ├── metrics.json               # MSE, MAE for this combination
│   └── predictions.npz           # Predictions and ground truth arrays
├── train_h1_pred_h4/
│   ├── metrics.json
│   └── predictions.npz
├── train_h1_pred_h8/
│   ├── metrics.json
│   └── predictions.npz
├── train_h4_pred_h1/
│   ├── metrics.json
│   └── predictions.npz
... (9 total combinations)
```

## Visualization

### Plotting Training Metrics

**Option 1: TensorBoard** (for `multi_horizon_finetune.py`)
```bash
tensorboard --logdir=./checkpoints/multi_horizon/horizon_1/tb
```

**Option 2: Plot from TensorBoard logs** (creates static plots)
```bash
python scripts/plot_training_metrics.py \
    --logdir=./checkpoints/multi_horizon/horizon_1/tb \
    --output_dir=./checkpoints/multi_horizon/horizon_1/training_plots \
    --metrics=loss,mse,mae
```

### Plotting Prediction Results

After running predictions, you can visualize the trajectories:

```bash
python scripts/visualize_trajectories.py \
    --results_dir=./results/multi_horizon \
    --output_dir=./results/multi_horizon/plots \
    --action_dim_labels=x,y,z,roll,pitch,yaw,gripper \
    --max_trajectories=5 \
    --save_plots=true \
    --show_plots=false
```

This will create:
- Individual trajectory plots showing predicted vs ground truth for each action dimension
- Summary plots with all dimensions in a grid layout plus MSE over time
- All plots saved in `results_dir/train_hX_pred_hY/visualizations/`

## Notes

- The teleoperation action head uses a **single covariance matrix** shared across all actions in the horizon (as modified in the codebase)
- Each model is finetuned independently for its specific horizon
- Predictions test how well models trained on one horizon generalize to other horizons
- The `summary.json` file contains a table of all results for easy comparison

