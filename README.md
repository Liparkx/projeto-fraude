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

### Data preparation

The raw dataset has 284,807 rows, of which 1,081 are exact duplicates (19 of them fraud). `load_data()` removes these duplicates by default, before the stratified 60/20/20 split, leaving 283,726 rows and 473 fraud cases (0.167%). Duplicates are removed first so that identical transactions cannot appear in both training and evaluation sets and inflate the metrics. Pass `drop_duplicates=False` to keep the original data.
