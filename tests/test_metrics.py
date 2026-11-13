"""Testes das métricas usando casos sintéticos com resultado calculado à mão."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection.metrics import metrics_at_threshold, pr_auc, recall_at_fpr, roc_auc

# 10 negativos e 4 positivos. Os positivos e negativos se separam bem,
# exceto por um negativo (1.0) acima de dois positivos (0.55 e 0.25).
Y_EXAMPLE = np.array([0] * 10 + [1] * 4)
SCORES_EXAMPLE = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # negativos
    + [1.05, 0.55, 0.25, 0.01]  # positivos
)


def test_pr_auc_is_one_for_perfect_ranking() -> None:
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert pr_auc(y, s) == pytest.approx(1.0)


def test_roc_auc_is_one_for_perfect_ranking_and_half_for_constant_scores() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)


def test_pr_auc_is_lower_for_inverted_ranking() -> None:
    y = np.array([0, 0, 0, 1, 1])
    good = pr_auc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    bad = pr_auc(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1]))
    assert bad < good


def test_recall_at_fpr_picks_best_recall_within_limit() -> None:
    # FPR alvo de 10% permite 1 falso positivo (de 10 negativos).
    # Com limiar em (0.9, 1.0], o único FP é o negativo 1.0 e o único TP é o positivo 1.05.
    # O sklearn devolve o próprio score do ponto da curva como limiar: 1.0.
    op = recall_at_fpr(Y_EXAMPLE, SCORES_EXAMPLE, target_fpr=0.10)

    assert op.fpr == pytest.approx(0.10)
    assert op.recall == pytest.approx(0.25)
    assert op.threshold == pytest.approx(1.0)


def test_recall_at_zero_fpr_uses_threshold_above_all_negatives() -> None:
    # Com FPR zero, o limiar precisa ficar acima de todos os negativos (1.0).
    op = recall_at_fpr(Y_EXAMPLE, SCORES_EXAMPLE, target_fpr=0.0)

    assert op.fpr == 0.0
    assert op.recall == pytest.approx(0.25)


def test_recall_at_fpr_is_one_for_full_separation() -> None:
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    op = recall_at_fpr(y, s, target_fpr=0.001)

    assert op.recall == pytest.approx(1.0)
    assert op.fpr == 0.0


def test_recall_at_fpr_threshold_reproduces_reported_metrics() -> None:
    # O limiar escolhido deve reproduzir o recall e a FPR reportados ao ser reaplicado.
    op = recall_at_fpr(Y_EXAMPLE, SCORES_EXAMPLE, target_fpr=0.10)
    check = metrics_at_threshold(Y_EXAMPLE, SCORES_EXAMPLE, op.threshold)

    assert check.recall == pytest.approx(op.recall)
    assert check.fpr == pytest.approx(op.fpr)


@pytest.mark.parametrize("bad_fpr", [-0.1, 1.5])
def test_recall_at_fpr_rejects_target_outside_unit_interval(bad_fpr: float) -> None:
    with pytest.raises(ValueError, match="target_fpr"):
        recall_at_fpr(Y_EXAMPLE, SCORES_EXAMPLE, target_fpr=bad_fpr)


def test_metrics_at_threshold_confusion_counts_and_derived_values() -> None:
    # Limiar 0.5: prevê positivo para score >= 0.5.
    # Negativos: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 -> 6 FP e 4 TN.
    # Positivos: 1.05 e 0.55 -> 2 TP; 0.25 e 0.01 -> 2 FN.
    m = metrics_at_threshold(Y_EXAMPLE, SCORES_EXAMPLE, threshold=0.5)

    assert (m.tp, m.fp, m.tn, m.fn) == (2, 6, 4, 2)
    assert m.precision == pytest.approx(2 / 8)
    assert m.recall == pytest.approx(0.5)
    assert m.fpr == pytest.approx(0.6)
    assert m.f1 == pytest.approx(1 / 3)


def test_metrics_at_threshold_handles_no_positive_predictions() -> None:
    # Limiar acima de todos os scores: nenhuma predição positiva, precision deve ser 0.
    m = metrics_at_threshold(Y_EXAMPLE, SCORES_EXAMPLE, threshold=10.0)

    assert (m.tp, m.fp) == (0, 0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
