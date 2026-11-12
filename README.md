# Fraud Detection System

Credit card fraud detection using the [ULB dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Goals

- Compare class imbalance strategies: Baseline, SMOTE, Class Weights, Random Undersampling.
- Evaluate with PR-AUC, recall at fixed FPR, precision and F1 (accuracy is not a primary metric).
- Optimize the decision threshold on validation data using configurable business costs (FN vs FP).
- Later: serve the model through a FastAPI service.

## Project structure

```
src/fraud_detection/   Python package
tests/                 Pytest tests
notebooks/             Exploration notebooks
data/raw/              Raw dataset (not versioned)
data/processed/        Processed data (not versioned)
models/                Model artifacts (not versioned)
reports/               Results and figures
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .[dev]
```

`requirements-freeze.txt` is a full `pip freeze` of the original environment (UTF-16 encoded), kept for reference only.

## Status

Phase 0 (project hygiene) done. Dataset is not downloaded yet.
