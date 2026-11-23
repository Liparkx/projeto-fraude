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

## How to run the experiment

Place `creditcard.csv` in `data/raw/`, then run:

```bash
python -m fraud_detection.experiment
```

The script trains the 8 combinations (logistic regression and XGBoost × baseline, SMOTE, undersampling and class weights) on the training split, evaluates them on validation, and evaluates the selected combination once on the test split. Outputs:

- `reports/tables/`: validation comparison, cost sensitivity and test results (CSV).
- `reports/figures/`: Precision-Recall curves, recall at fixed FPR with bootstrap intervals, cost versus threshold, and cost sensitivity (PNG).
- `models/best_model.joblib`: the fitted pipeline with its thresholds and metadata.

On the development machine the run took about 48 seconds. The run is deterministic for a fixed `random_state`.

## Status

- Phases 0 to 5 are done: setup, data preparation, metrics, cost-based threshold, and the experiment pipeline.
- Phase 6 (real run of the experiment) is done. Results are recorded in `PROGRESS.md`.
- Next: apply the class-weight model as the chosen combination, fix the operating threshold from business costs, and add the exploratory data analysis notebook.

### Data preparation

The raw dataset has 284,807 rows, of which 1,081 are exact duplicates (19 of them fraud). `load_data()` removes these duplicates by default, before the stratified 60/20/20 split, leaving 283,726 rows and 473 fraud cases (0.167%). Duplicates are removed first so that identical transactions cannot appear in both training and evaluation sets and inflate the metrics. Pass `drop_duplicates=False` to keep the original data.
