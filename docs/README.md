# 3D Medical Image Classification

This repository contains a modular pipeline for 3D medical image classification.
It supports preprocessing, model training, evaluation, and hyperparameter optimization.

## Scope

- 3D data loading from `.npy` files (from zip or extracted folder)
- Config-driven preprocessing and data splitting
- Optional class balancing and augmentation
- CNN and GNN model options
- Training, evaluation, and model comparison workflows
- Metric reports, plots, and PDF evaluation summaries

## Key Features

- Config-based workflow through `configs/config.yaml`
- Multiple split strategies: existing, simple, stratified, patient-level
- Optional balancing: oversample, undersample, SMOTE, ADASYN, combined methods
- Threshold-aware evaluation (`fixed`, `f_beta`, `recall_at_precision`)
- Outputs in JSON, TXT, PNG, and PDF formats

## Project Structure

```text
configs/      # Runtime configuration
src/          # Core modules (models, training, preprocessing, utils)
cli/          # Command-line entry points
tools/        # Analysis and utility scripts
scripts/      # Setup and helper scripts
tests/        # Test suite
outputs/      # Generated artifacts
```

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- Some packages are optional and used only in specific workflows (for example `torch-geometric`, `streamlit`, `optuna`).
- For GPU usage, install a compatible PyTorch build for your CUDA/driver setup.

## Data Expectations

Set dataset paths in `configs/config.yaml`.

Expected metadata CSV fields:

- `ROI_id`
- `ROI_anomaly`
- `subset`

Expected subset labels:

- `train`, `dev`/`val`, `test`
- or normalized aliases like `ZS-train`, `ZS-dev`, `ZS-test`

Image source can be either:

- zip archive containing `.npy` files
- extracted directory containing `.npy` files

## Supported Models

Configured via `model.model_type` in `configs/config.yaml`.

- `cnn3d_simple`
- `resnet3d`
- `densenet3d`
- `gcn`
- `gat`
- `graphsage`

## Quick Start

1. Configure `configs/config.yaml`.
2. Run one of the entry points below.

### Main interactive menu

```bash
python main.py
```

### Train

```bash
python cli/run_training.py --config configs/config.yaml
```

### Evaluate a checkpoint

```bash
python cli/run_evaluation.py evaluate \
  --checkpoint <path_to_checkpoint> \
  --config configs/config.yaml \
  --save-dir outputs/evaluation_test \
  --test-set test
```

### Compare multiple checkpoints

```bash
python cli/run_evaluation.py compare \
  --checkpoints modelA=<path_a> modelB=<path_b> \
  --config configs/config.yaml \
  --save-dir outputs/comparison_results
```

### Hyperparameter optimization (interactive)

```bash
python cli/run_hyperparameter_optimization.py
```

## Outputs

Typical output files include:

- `evaluation_metrics.json`
- `evaluation_results_full.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `evaluation_report.pdf`

Output directories are controlled via config.

## Tests

Run tests with:

```bash
python -m pytest tests -v
```

## Privacy and Content Policy

This README intentionally avoids project-specific private details.
Use generic placeholders for dataset locations, identifiers, and external endpoints when sharing publicly.
