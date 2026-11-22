"""Testes do experimento ponta a ponta, com dados sintéticos pequenos."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fraud_detection.data import TARGET_COLUMN
from fraud_detection.experiment import ExperimentConfig, run_experiment


def _synthetic_df(n: int = 3_000, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    data = {"Time": rng.uniform(0, 172_000, n), "Amount": rng.exponential(80, n)}
    for i in range(1, 6):
        data[f"V{i}"] = rng.normal(size=n) + (1.2 * y if i <= 2 else 0.0)
    data[TARGET_COLUMN] = y
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _synthetic_df()


@pytest.fixture(scope="module")
def results_and_dir(df: pd.DataFrame, tmp_path_factory):
    out = tmp_path_factory.mktemp("experiment")
    config = ExperimentConfig(
        reports_dir=out / "reports",
        models_dir=out / "models",
        n_bootstrap=20,
    )
    return run_experiment(config=config, df=df, save=True), config


@pytest.fixture(scope="module")
def results(results_and_dir):
    return results_and_dir[0]


def test_validation_table_has_eight_rows_and_expected_columns(results) -> None:
    table = results.validation_comparison

    assert len(table) == 8
    assert set(table["model"]) == {"logistic", "xgboost"}
    assert set(table["strategy"]) == {"baseline", "smote", "undersampling", "class_weight"}
    for column in (
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "roc_auc",
        "recall_0.1pct",
        "threshold_0.1pct",
        "fpr_0.1pct",
        "recall_1pct",
        "precision_default",
        "recall_default",
        "f1_default",
    ):
        assert column in table.columns


def test_best_model_has_highest_validation_pr_auc(results) -> None:
    table = results.validation_comparison
    best = table.loc[
        (table["model"] == results.best_model) & (table["strategy"] == results.best_strategy)
    ]

    assert best["pr_auc"].iloc[0] == pytest.approx(table["pr_auc"].max())
    assert isinstance(results.ci_overlap, bool)


def test_test_results_has_one_row_for_the_chosen_combination(results) -> None:
    test = results.test_results

    assert len(test) == 1
    assert test["model"].iloc[0] == results.best_model
    assert test["strategy"].iloc[0] == results.best_strategy


def test_test_uses_validation_thresholds_without_reoptimizing(results) -> None:
    # O limiar aplicado ao teste deve ser exatamente o escolhido na validação.
    table = results.validation_comparison
    chosen = table.loc[
        (table["model"] == results.best_model) & (table["strategy"] == results.best_strategy)
    ]

    for label in ("0.1pct", "1pct"):
        assert results.test_results[f"threshold_{label}"].iloc[0] == pytest.approx(
            chosen[f"threshold_{label}"].iloc[0]
        )


def test_artifacts_are_written(results_and_dir) -> None:
    results, config = results_and_dir
    tables = config.reports_dir / "tables"

    for name in (
        "validation_comparison.csv",
        "validation_sensitivity.csv",
        "test_results.csv",
        "test_sensitivity.csv",
    ):
        assert (tables / name).is_file()

    figures = config.reports_dir / "figures"
    for name in (
        "pr_curves_validation.png",
        "custo_x_limiar_melhor_modelo.png",
        "sensibilidade_custo_melhor_modelo.png",
    ):
        assert (figures / name).stat().st_size > 0

    assert (config.models_dir / "best_model.joblib").is_file()


def test_saved_model_loads_and_predicts_probabilities(results_and_dir, df) -> None:
    _, config = results_and_dir
    payload = joblib.load(config.models_dir / "best_model.joblib")

    assert {"pipeline", "model_name", "strategy", "fpr_thresholds", "versions"} <= set(payload)
    proba = payload["pipeline"].predict_proba(df.drop(columns=[TARGET_COLUMN]).head(50))
    assert proba.shape == (50, 2)


def test_run_is_reproducible_with_same_config(df: pd.DataFrame, results) -> None:
    config = ExperimentConfig(n_bootstrap=20)
    again = run_experiment(config=config, df=df, save=False)

    assert_frame_equal(again.validation_comparison, results.validation_comparison)
    assert again.best_model == results.best_model
    assert again.best_strategy == results.best_strategy
