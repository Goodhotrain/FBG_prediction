# PSDCFN: Personalized Static-Dynamic Counterfactual Fusion Network

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://goodhotrain.github.io/FBG_prediction)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Nutrition-Aware Fasting Blood Glucose Prediction via Static-Dynamic Counterfactual Fusion**

*Haoyu Gu, Peiguang Jing, Huaiyan Jiang, Yu Liu — Tianjin University*

## Overview

PSDCFN is a deep learning framework for personalized fasting blood glucose (FBG) prediction that jointly models heterogeneous health data including physiological profiles, clinical metrics, and daily nutritional intake records. The model features a dual-stream architecture with counterfactually regularized sparse attention, inter-feature self-attention, and progressive causal fusion modulation.

![](figures/umap_ablation_2x2.png)

## Key Features

- **Dual-stream architecture** for static physiological profiles and dynamic dietary records
- **Counterfactually regularized sparse attention** for interpretable physiological representations
- **Inter-feature self-attention** to capture nutrient dependencies
- **Progressive causal fusion modulation** for personalized FBG prediction

## Project Structure

```
├── models/              # Model definitions
│   ├── model.py         # Main PSDCFN model (FBGPredictor)
│   ├── model_v.py       # Model variant
│   ├── argf.py          # Multi-modal fusion modules
│   ├── mamba/           # State-space model components
│   └── my_model/        # Custom neural network layers
├── data/
│   └── dataset.py       # FBG Dataset loader
├── common/              # Utilities (config, optimizer, plotting)
├── main.py              # Training script
├── valid.py             # Validation / testing script
├── opt.py               # Argument configuration
├── train.sh             # Launch training runs
├── figures/             # Result figures and visualizations
├── index.html           # Project showcase page
└── a.ipynb              # Data preprocessing notebook
```

## Setup

```bash
# Create environment
conda create -n fbgs python=3.10
conda activate fbgs

# Install dependencies
pip install torch pandas numpy scikit-learn matplotlib shap umap-learn tensorboard
```

## Usage

### Training

```bash
python main.py
```

Configure paths and hyperparameters in `opt.py`.

### Data

The dataset includes:
- **Static features** (32 dims): age, gender, BMI, clinical metrics, lifestyle factors
- **Dynamic features** (79 dims): daily nutritional intake records
- **Target**: Fasting blood glucose (FBG) level

> Note: Data files are not included in this repository. Please contact the authors for data access.

## Results

PSDCFN consistently outperforms baselines in FBG prediction:

- **With historical FBG**: Superior MAE and RMSE compared to existing methods
- **Without historical FBG**: Maintains robust performance using only static + nutrition data

Detailed results and visualizations are available on the [project page](https://goodhotrain.github.io/FBG_prediction).

## Citation

```bibtex
@article{gu2025psdcfn,
  title   = {PSDCFN: A Personalized Static-Dynamic Counterfactual Fusion
             Network for Nutrition-Aware Fasting Blood Glucose Prediction},
  author  = {Gu, Haoyu and Jing, Peiguang and Jiang, Huaiyan and Liu, Yu},
  journal = {Preprint},
  year    = {2025}
}
```

## License

MIT License

