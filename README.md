# PSDCFN: Personalized Static–Dynamic Counterfactual Fusion Network

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://goodhotrain.github.io/FBG_prediction)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org/)

An experimental framework for personalized fasting blood glucose (FBG) regression from static clinical attributes, longitudinal nutrition records, and historical FBG measurements.

## What is implemented

- **Clinical feature tokenization**: each scalar static feature receives a learnable value embedding and participates in self-attention.
- **Masked temporal modeling**: a Transformer models variable-length dietary histories without attending to padded days.
- **Progressive cross-modal fusion**: static patient context repeatedly queries the longitudinal stream through gated residual cross-attention.
- **Adaptive history integration**: a learned gate controls how much historical FBG contributes to each prediction.
- **Counterfactual consistency**: perturbations of static attributes regularize prediction stability during training.
- **Reproducible experiments**: deterministic seeding, train-only preprocessing, gradient clipping, early stopping, resumable checkpoints, TensorBoard, and JSONL metrics.
- **Interpretability outputs**: static-feature attention, temporal attention, and the history gate are available from `ModelOutput`.

## Repository layout

```text
.
├── common/engine.py       # epochs, regression metrics, checkpoint I/O
├── common/optimizer.py    # optimizer and scheduler factories
├── data/dataset.py        # validation, preprocessing, temporal windows
├── models/model.py        # current PSDCFN implementation
├── models/                # retained research baselines and ablations
├── tests/                 # model and data-pipeline tests
├── main.py                # training entry point
├── valid.py               # evaluation and prediction export
└── opt.py                 # documented CLI configuration
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data contract

Private clinical data is not included. Three aligned CSV files are expected:

1. `labels.csv`: the first column is the numeric FBG target and the second is a split (`train` or `test`).
2. `static.csv`: `UserID`, `Day`, followed by static numeric features.
3. `nutrition.csv`: `UserID`, `Day`, followed by nutrition features; the **last feature must be historical FBG**.

Static and label files must have the same row order. Nutrition rows may contain repeated users and are sorted by day when the history window is built. Scalers are fit on the training split and reused for validation, preventing validation-statistic leakage.

## Training

```bash
python main.py \
  --label-path data/labels.csv \
  --static-path data/static.csv \
  --nutrition-path data/nutrition.csv \
  --static-dim 32 \
  --nutrition-dim 78 \
  --epochs 100 \
  --optimizer adamw \
  --scheduler cosine
```

Use `python main.py --help` for all options. Each run writes:

```text
results/<run-name>/
├── config.json
├── metrics.jsonl
├── checkpoints/best.pt
├── checkpoints/last.pt
└── tensorboard/
```

Resume a run with `--resume --checkpoint results/.../checkpoints/last.pt`.

## Evaluation

Evaluation architecture arguments must match those used for training:

```bash
python valid.py \
  --checkpoint results/<run-name>/checkpoints/best.pt \
  --label-path data/labels.csv \
  --static-path data/static.csv \
  --nutrition-path data/nutrition.csv
```

The command reports MAE, MSE, RMSE, and R², then saves `predictions.csv` alongside the run artifacts.

## Testing

Tests use synthetic data; access to the private dataset is not required.

```bash
python -m pytest -q
```

## Citation

If this repository helps your work, please cite the associated PSDCFN paper once its archival record is available.
