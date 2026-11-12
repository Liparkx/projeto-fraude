# Credit Card Fraud Detection

## Objective
Binary classification model to detect fraudulent transactions.
Dataset: Credit Card Fraud Detection (ULB, Kaggle), 284,807 transactions, ~0.17% fraud.

## Stack
- Python 3.12.10
- pandas, numpy, scikit-learn, imbalanced-learn, xgboost or lightgbm, matplotlib
- pytest for testing
- Virtual environment in `.venv/`

## Structure
- `data/raw/`: original data (do not version control)
- `data/processed/`: preprocessed data (do not version control)
- `src/fraud_detection/`: package source code
  - `data.py`: data loading and stratified splitting
  - `sampling.py`: SMOTE and undersampling
  - `models.py`: models and class weights
  - `metrics.py`: PR-AUC, recall at a fixed FPR
  - `threshold.py`: threshold selection based on cost
- `notebooks/`: for exploratory data analysis (EDA) only
- `models/`: serialized trained models
- `reports/`: figures and result tables
- `tests/`: unit tests using pytest

## Development Rules
- All production code must reside in `src/`, not in notebooks.
- Perform stratified splitting before any transformations.
- Apply SMOTE and undersampling only to the training set, using `imblearn.pipeline.Pipeline`. Never apply these before the split or during validation.
- Scale the `Time` and `Amount` features. `V1` through `V28` are already PCA components; do not rescale them.
- Use a fixed `random_state` for all stochastic operations.
- Do not print entire DataFrames; display only summaries.

## Metrics
- Primary metric: PR-AUC (`average_precision_score`). - Supplementary: ROC-AUC.
- Recall at a fixed FPR (e.g., 0.1%), with the threshold determined during validation.
- Cost-based threshold: minimize `C_FN * FN + C_FP * FP`.
  - `C_FN` (undetected fraud) and `C_FP` (blocking a legitimate transaction) are configurable parameters, not constants.
  - Test a range of values ​​and report a sensitivity analysis.

## Commands
- Installation: `pip install -e .[dev]` (adjust according to `pyproject.toml`)
- Tests: `pytest`
- Linting/formatting: `ruff`

## Language
- Code, variable names, README, and function names in English.
- Comments, high-level docstrings, and responses in Portuguese.

## Workflow
- One step at a time, with a clear scope.
- Briefly describe the plan before implementation.
- Upon completion, run the tests and report the results.