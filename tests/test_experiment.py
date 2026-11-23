"""Testes do experimento ponta a ponta, com dados sintéticos pequenos."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fraud_detection.data import TARGET_COLUMN, split_data
from fraud_detection.experiment import ExperimentConfig, run_experiment
from fraud_detection.threshold import best_threshold, cost_at_threshold, transaction_cost_arrays


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


def _row(table: pd.DataFrame, model: str, strategy: str) -> pd.Series:
    return table.loc[(table["model"] == model) & (table["strategy"] == strategy)].iloc[0]


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
        "selected",
    ):
        assert column in table.columns


def test_cost_table_has_one_row_per_combination_with_reais_columns(results) -> None:
    table = results.validation_cost_by_model

    assert len(table) == 8
    for column in (
        "threshold",
        "range_low",
        "range_high",
        "val_cost",
        "val_fn",
        "val_fp",
        "val_cost_default",
        "val_cost_no_model",
        "selected",
    ):
        assert column in table.columns
    assert table["selected"].sum() == 1


def test_selected_is_the_lowest_validation_cost(results) -> None:
    table = results.validation_cost_by_model
    chosen = _row(table, results.selected_model, results.selected_strategy)

    assert results.selection_rule == "menor_custo_validacao"
    assert chosen["val_cost"] == pytest.approx(table["val_cost"].min())


def test_leader_is_the_highest_validation_pr_auc(results) -> None:
    table = results.validation_comparison
    leader = _row(table, *results.pr_auc_leader)

    assert leader["pr_auc"] == pytest.approx(table["pr_auc"].max())
    assert leader["rank_pr_auc"] == 1
    assert isinstance(results.ci_overlap, bool)


def test_selected_flag_marks_only_the_chosen_combination(results) -> None:
    table = results.validation_comparison

    assert table["selected"].sum() == 1
    chosen = table.loc[table["selected"]].iloc[0]
    assert (chosen["model"], chosen["strategy"]) == (
        results.selected_model,
        results.selected_strategy,
    )


@pytest.mark.parametrize("label", ["0.1pct", "1pct"])
def test_recall_at_fpr_has_ordered_bootstrap_interval_in_unit_range(results, label) -> None:
    table = results.validation_comparison
    value = table[f"recall_{label}"]
    low = table[f"recall_{label}_ci_low"]
    high = table[f"recall_{label}_ci_high"]

    assert ((low >= 0) & (high <= 1)).all()
    assert (low <= high).all()
    # Os intervalos devem envolver o valor pontual da validação.
    assert ((low <= value + 1e-9) & (value - 1e-9 <= high)).all()


def test_rankings_are_consistent_with_results(results) -> None:
    table = results.validation_comparison

    assert isinstance(results.rankings_disagree, bool)
    assert len(results.best_by_recall) == 2
    assert _row(table, *results.pr_auc_leader)["rank_pr_auc"] == 1
    # Discordância = o maior recall na FPR principal não é a líder por PR-AUC.
    assert results.rankings_disagree == (tuple(results.best_by_recall) != results.pr_auc_leader)


def test_primary_fpr_must_be_one_of_the_targets(df: pd.DataFrame) -> None:
    config = ExperimentConfig(fpr_targets=(0.01,), primary_fpr=0.001, n_bootstrap=5)
    with pytest.raises(ValueError, match="primary_fpr"):
        run_experiment(config=config, df=df, save=False)


def test_invalid_selection_criterion_is_rejected(df: pd.DataFrame) -> None:
    config = ExperimentConfig(selection_criterion="sorte", n_bootstrap=5)
    with pytest.raises(ValueError, match="selection_criterion"):
        run_experiment(config=config, df=df, save=False)


def test_invalid_business_costs_are_rejected(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="fp_cost"):
        run_experiment(config=ExperimentConfig(fp_cost=0.0, n_bootstrap=5), df=df, save=False)
    with pytest.raises(ValueError, match="fraud_fixed_cost"):
        run_experiment(
            config=ExperimentConfig(fraud_fixed_cost=-1.0, n_bootstrap=5), df=df, save=False
        )


def test_pr_auc_criterion_selects_the_pr_auc_leader(df: pd.DataFrame) -> None:
    config = ExperimentConfig(selection_criterion="pr_auc", n_bootstrap=20)
    res = run_experiment(config=config, df=df, save=False)

    assert res.selection_rule == "maior_pr_auc"
    assert (res.selected_model, res.selected_strategy) == res.pr_auc_leader


def test_test_results_has_one_row_for_the_chosen_combination(results) -> None:
    test = results.test_results

    assert len(test) == 1
    assert test["model"].iloc[0] == results.selected_model
    assert test["strategy"].iloc[0] == results.selected_strategy
    assert test["selection_rule"].iloc[0] == results.selection_rule


def test_operating_point_table_has_one_row_with_the_fixed_threshold(results) -> None:
    op = results.test_operating_point

    assert len(op) == 1
    assert op["threshold"].iloc[0] == pytest.approx(results.operating_threshold)
    assert op["model"].iloc[0] == results.selected_model
    assert op["strategy"].iloc[0] == results.selected_strategy


def test_operating_threshold_is_the_validation_cost_optimum(results) -> None:
    # O limiar fixado deve ser exatamente o ótimo de custo da validação (reproduzido à mão).
    chosen = _row(
        results.validation_cost_by_model, results.selected_model, results.selected_strategy
    )
    assert chosen["threshold"] == pytest.approx(results.operating_threshold)

    s_val = results.val_scores[f"{results.selected_model}/{results.selected_strategy}"]
    c_fn, c_fp = transaction_cost_arrays(results.amounts_val, 25.0, 5.0)
    opt = best_threshold(results.y_val, s_val, c_fn, c_fp)
    assert opt.threshold == pytest.approx(results.operating_threshold)
    assert opt.cost == pytest.approx(chosen["val_cost"])


def test_test_cost_matches_independent_calculation(results, df) -> None:
    # Recalcula o custo do teste a partir do pipeline e do split, sem reutilizar o experimento.
    split = split_data(df, val_size=0.2, test_size=0.2, random_state=42)
    s_test = results.pipeline.predict_proba(split.X_test)[:, 1]
    c_fn, c_fp = transaction_cost_arrays(split.X_test["Amount"].to_numpy(), 25.0, 5.0)
    expected = cost_at_threshold(
        split.y_test.to_numpy(), s_test, results.operating_threshold, c_fn, c_fp
    )

    op = results.test_operating_point.iloc[0]
    assert op["test_cost"] == pytest.approx(expected)
    assert op["test_savings_vs_no_model"] == pytest.approx(
        op["test_cost_no_model"] - op["test_cost"]
    )


def test_test_uses_validation_thresholds_without_reoptimizing(results) -> None:
    # O limiar aplicado ao teste deve ser exatamente o escolhido na validação.
    chosen = _row(results.validation_comparison, results.selected_model, results.selected_strategy)

    for label in ("0.1pct", "1pct"):
        assert results.test_results[f"threshold_{label}"].iloc[0] == pytest.approx(
            chosen[f"threshold_{label}"]
        )


def test_cost_scenarios_have_one_row_per_scenario(results) -> None:
    config = ExperimentConfig()
    expected = len(config.scenario_fraud_fixed_costs) * len(config.scenario_fp_costs)

    assert len(results.cost_scenarios) == expected
    assert {"val_cost", "eval_cost", "eval_cost_no_model"} <= set(results.cost_scenarios.columns)


def test_artifacts_are_written(results_and_dir) -> None:
    _, config = results_and_dir
    tables = config.reports_dir / "tables"

    for name in (
        "validation_comparison.csv",
        "validation_cost_by_model.csv",
        "validation_sensitivity.csv",
        "test_results.csv",
        "test_operating_point.csv",
        "test_sensitivity.csv",
        "cost_scenarios.csv",
    ):
        assert (tables / name).is_file()

    figures = config.reports_dir / "figures"
    for name in (
        "pr_curves_validation.png",
        "recall_em_fpr_fixa_validacao.png",
        "custo_x_limiar_melhor_modelo.png",
        "sensibilidade_custo_melhor_modelo.png",
        "custo_reais_limiar_selecionado.png",
        "custo_otimo_por_combinacao.png",
        "limiar_por_cenario_de_custo.png",
    ):
        assert (figures / name).stat().st_size > 0

    assert (config.models_dir / "best_model.joblib").is_file()


def test_saved_model_loads_and_carries_the_operating_threshold(results_and_dir, df) -> None:
    results, config = results_and_dir
    payload = joblib.load(config.models_dir / "best_model.joblib")

    assert {
        "pipeline",
        "model_name",
        "strategy",
        "selection_rule",
        "operating_threshold",
        "operating_range",
        "cost_scenario",
        "fpr_thresholds",
        "test_operating_point",
        "versions",
    } <= set(payload)
    assert payload["operating_threshold"] == pytest.approx(results.operating_threshold)
    assert payload["cost_scenario"]["fraud_fixed_cost"] == 25.0
    proba = payload["pipeline"].predict_proba(df.drop(columns=[TARGET_COLUMN]).head(50))
    assert proba.shape == (50, 2)


def test_run_is_reproducible_with_same_config(df: pd.DataFrame, results) -> None:
    config = ExperimentConfig(n_bootstrap=20)
    again = run_experiment(config=config, df=df, save=False)

    assert_frame_equal(again.validation_comparison, results.validation_comparison)
    assert_frame_equal(again.validation_cost_by_model, results.validation_cost_by_model)
    assert again.selected_model == results.selected_model
    assert again.selected_strategy == results.selected_strategy
    assert again.operating_threshold == pytest.approx(results.operating_threshold)
