"""Testes da seleção de limiar pelo custo de negócio."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.metrics import metrics_at_threshold
from fraud_detection.threshold import (
    DEFAULT_COST_RATIOS,
    bayes_threshold,
    best_threshold,
    cost_at_threshold,
    cost_curve,
    sensitivity_analysis,
)

# 10 negativos (0,1 a 1,0) e 4 positivos (1,05; 0,55; 0,25; 0,01).
# Valores calculados à mão nos testes abaixo.
Y_EXAMPLE = np.array([0] * 10 + [1] * 4)
SCORES_EXAMPLE = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [1.05, 0.55, 0.25, 0.01]
)


def test_cost_curve_has_expected_columns_and_order() -> None:
    curve = cost_curve(Y_EXAMPLE, SCORES_EXAMPLE, c_fn=1, c_fp=1)

    assert list(curve.columns) == ["threshold", "tp", "fp", "tn", "fn", "cost"]
    # 14 scores distintos + o candidato inf (nenhum bloqueio).
    assert len(curve) == 15
    assert curve["threshold"].is_monotonic_decreasing
    assert curve.iloc[0]["threshold"] == np.inf


def test_cost_curve_with_no_blocking_counts_all_frauds_as_fn() -> None:
    curve = cost_curve(Y_EXAMPLE, SCORES_EXAMPLE, c_fn=1, c_fp=1)
    no_block = curve.iloc[0]

    assert (no_block["tp"], no_block["fp"], no_block["fn"]) == (0, 0, 4)
    assert no_block["cost"] == 4


def test_cost_curve_matches_metrics_at_threshold_for_every_candidate() -> None:
    curve = cost_curve(Y_EXAMPLE, SCORES_EXAMPLE, c_fn=2, c_fp=1)

    for _, row in curve.iloc[1:].iterrows():
        m = metrics_at_threshold(Y_EXAMPLE, SCORES_EXAMPLE, row["threshold"])
        assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (m.tp, m.fp, m.fn, m.tn)


def test_best_threshold_with_equal_costs() -> None:
    # Com custos iguais: limiar 1,05 deixa 1 TP, 3 FN, 0 FP, custo 3 (mínimo).
    opt = best_threshold(Y_EXAMPLE, SCORES_EXAMPLE, c_fn=1, c_fp=1)

    assert opt.threshold == pytest.approx(1.05)
    assert opt.cost == pytest.approx(3)
    assert (opt.metrics.fn, opt.metrics.fp) == (3, 0)


def test_best_threshold_blocks_everything_when_fn_is_expensive() -> None:
    # Com C_FN = 10 e C_FP = 1, bloquear tudo (limiar 0,01) custa 10 e é o mínimo.
    opt = best_threshold(Y_EXAMPLE, SCORES_EXAMPLE, c_fn=10, c_fp=1)

    assert opt.threshold == pytest.approx(0.01)
    assert opt.cost == pytest.approx(10)
    assert (opt.metrics.fn, opt.metrics.fp) == (0, 10)


def test_best_threshold_moderate_ratio() -> None:
    # Com C_FN = 3 e C_FP = 1: limiar 1,05 custa 9; bloquear tudo custa 10.
    opt = best_threshold(Y_EXAMPLE, SCORES_EXAMPLE, c_fn=3, c_fp=1)

    assert opt.threshold == pytest.approx(1.05)
    assert opt.cost == pytest.approx(9)


def test_tie_prefers_highest_threshold() -> None:
    # Bloquear o único score (custo 1, FP) ou não bloquear nada (custo 1, FN) empatam.
    # O empate deve resolver para o maior limiar, inf (nenhum bloqueio).
    opt = best_threshold(np.array([1, 0]), np.array([0.5, 0.5]), c_fn=1, c_fp=1)

    assert opt.threshold == np.inf
    assert opt.cost == pytest.approx(1)


@pytest.mark.parametrize("ratio", [1, 10, 100])
def test_best_threshold_matches_brute_force(ratio: float) -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    s = rng.random(300)

    candidates = [*np.unique(s), np.inf]
    costs = [cost_at_threshold(y, s, t, c_fn=ratio, c_fp=1) for t in candidates]

    opt = best_threshold(y, s, c_fn=ratio, c_fp=1)
    assert opt.cost == pytest.approx(min(costs))
    assert cost_at_threshold(y, s, opt.threshold, c_fn=ratio, c_fp=1) == pytest.approx(opt.cost)


def test_optimal_threshold_is_non_increasing_with_cost_ratio() -> None:
    # Quanto mais caro for deixar fraude passar, mais baixo deve ser o limiar.
    rng = np.random.default_rng(1)
    y = (rng.random(2_000) < 0.05).astype(int)
    s = np.clip(0.3 * y + rng.random(2_000) * 0.7, 0, 1)

    thresholds = [
        best_threshold(y, s, c_fn=ratio, c_fp=1).threshold for ratio in DEFAULT_COST_RATIOS
    ]
    assert all(a >= b for a, b in zip(thresholds, thresholds[1:], strict=False))


def test_bayes_threshold_values() -> None:
    assert bayes_threshold(c_fn=1, c_fp=1) == pytest.approx(0.5)
    assert bayes_threshold(c_fn=100, c_fp=1) == pytest.approx(1 / 101)


def test_bayes_threshold_is_near_empirical_optimum_for_calibrated_scores() -> None:
    # Probabilidades calibradas por construção: y ~ Bernoulli(p).
    # No limiar de Bayes o custo deve ficar próximo do ótimo empírico.
    rng = np.random.default_rng(2)
    p = rng.beta(0.5, 150, size=200_000)
    y = (rng.random(200_000) < p).astype(int)

    c_fn, c_fp = 100.0, 1.0
    best = best_threshold(y, p, c_fn=c_fn, c_fp=c_fp)
    at_bayes = cost_at_threshold(y, p, bayes_threshold(c_fn, c_fp), c_fn, c_fp)

    assert at_bayes <= best.cost * 1.05


@pytest.mark.parametrize(("c_fn", "c_fp"), [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)])
def test_invalid_costs_raise(c_fn: float, c_fp: float) -> None:
    with pytest.raises(ValueError):
        best_threshold(Y_EXAMPLE, SCORES_EXAMPLE, c_fn=c_fn, c_fp=c_fp)
    with pytest.raises(ValueError):
        bayes_threshold(c_fn, c_fp)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="tamanhos"):
        cost_curve(Y_EXAMPLE[:-1], SCORES_EXAMPLE, c_fn=1, c_fp=1)


def test_non_binary_target_raises() -> None:
    with pytest.raises(ValueError, match="0 e 1"):
        cost_curve(np.array([0, 2]), np.array([0.1, 0.9]), c_fn=1, c_fp=1)


def test_sensitivity_has_one_row_per_ratio_and_expected_columns() -> None:
    df = sensitivity_analysis(Y_EXAMPLE, SCORES_EXAMPLE, cost_ratios=(1, 10))

    assert len(df) == 2
    expected = {
        "ratio",
        "c_fn",
        "c_fp",
        "threshold",
        "bayes_threshold",
        "val_cost",
        "val_fn",
        "val_fp",
        "val_recall",
        "val_precision",
        "val_cost_default",
        "val_cost_no_model",
    }
    assert expected.issubset(df.columns)


def test_sensitivity_default_and_no_model_costs_by_hand() -> None:
    # Razão 10 (C_FN = 10, C_FP = 1), limiar padrão 0,5.
    # Positivos >= 0,5: 1,05 e 0,55 -> FN = 2 (0,25 e 0,01 ficam abaixo).
    # Negativos >= 0,5: 0,5, 0,6, 0,7, 0,8, 0,9, 1,0 -> FP = 6.
    # Custo = 10·2 + 1·6 = 26.
    # Sem modelo (deixar tudo passar): custo = C_FN · P = 10 · 4 = 40.
    df = sensitivity_analysis(Y_EXAMPLE, SCORES_EXAMPLE, cost_ratios=(10,))
    row = df.iloc[0]

    assert row["val_cost_default"] == pytest.approx(26)
    assert row["val_cost_no_model"] == pytest.approx(40)


def test_sensitivity_eval_applies_validation_threshold_without_reoptimizing() -> None:
    rng = np.random.default_rng(3)
    y_val = (rng.random(500) < 0.1).astype(int)
    s_val = rng.random(500)
    y_eval = (rng.random(500) < 0.1).astype(int)
    s_eval = rng.random(500)

    df = sensitivity_analysis(y_val, s_val, cost_ratios=(1, 50), y_eval=y_eval, s_eval=s_eval)

    for _, row in df.iterrows():
        expected = cost_at_threshold(y_eval, s_eval, row["threshold"], row["c_fn"], row["c_fp"])
        assert row["eval_cost"] == pytest.approx(expected)


def test_sensitivity_requires_eval_inputs_together() -> None:
    with pytest.raises(ValueError, match="informados juntos"):
        sensitivity_analysis(Y_EXAMPLE, SCORES_EXAMPLE, y_eval=Y_EXAMPLE)


def test_sensitivity_thresholds_are_non_increasing_with_ratio() -> None:
    rng = np.random.default_rng(4)
    y = (rng.random(2_000) < 0.05).astype(int)
    s = np.clip(0.3 * y + rng.random(2_000) * 0.7, 0, 1)

    df = sensitivity_analysis(y, s)
    assert np.all(np.diff(df["threshold"].to_numpy()) <= 0)


def test_sensitivity_rejects_non_positive_c_fp() -> None:
    with pytest.raises(ValueError, match="c_fp"):
        sensitivity_analysis(Y_EXAMPLE, SCORES_EXAMPLE, c_fp=0.0)
