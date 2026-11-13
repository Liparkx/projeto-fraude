"""Métricas de avaliação para classes desbalanceadas.

Acurácia não é usada aqui de propósito: com ~0,17% de fraudes, um classificador
que nunca detecta fraude já teria acurácia acima de 99%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ArrayLike = npt.ArrayLike


@dataclass(frozen=True)
class OperatingPoint:
    """Ponto de operação escolhido para uma FPR alvo.

    Attributes:
        threshold: limiar aplicado como ``score >= threshold``.
        recall: TPR (recall) obtido com esse limiar.
        fpr: taxa de falsos positivos obtida com esse limiar (<= alvo).
    """

    threshold: float
    recall: float
    fpr: float


@dataclass(frozen=True)
class ThresholdMetrics:
    """Matriz de confusão e métricas derivadas para um limiar fixo."""

    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Área sob a curva Precision-Recall (average precision). Métrica principal."""
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Área sob a curva ROC. Métrica complementar."""
    return float(roc_auc_score(y_true, y_score))


def recall_at_fpr(y_true: ArrayLike, y_score: ArrayLike, target_fpr: float) -> OperatingPoint:
    """Maior recall possível mantendo a FPR menor ou igual a ``target_fpr``.

    O limiar deve ser escolhido na validação e reaplicado, sem reajuste, no teste
    (ver ``metrics_at_threshold``).

    Args:
        y_true: rótulos binários (0/1).
        y_score: scores ou probabilidades da classe positiva.
        target_fpr: FPR máxima aceita, em [0, 1]. Ex.: 0.001 para 0,1%.

    Raises:
        ValueError: se ``target_fpr`` estiver fora de [0, 1].
    """
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr deve estar em [0, 1].")

    # drop_intermediate=False mantém todos os limiares; com True, o sklearn remove
    # pontos colineares e a busca pelo ponto de FPR alvo pode perder o melhor limiar.
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)

    # Ao longo da curva, FPR e TPR são não decrescentes. O último ponto com
    # FPR <= alvo é o de maior recall dentro da restrição.
    idx = int(np.flatnonzero(fpr <= target_fpr)[-1])

    return OperatingPoint(
        threshold=float(thresholds[idx]),
        recall=float(tpr[idx]),
        fpr=float(fpr[idx]),
    )


def metrics_at_threshold(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float
) -> ThresholdMetrics:
    """Matriz de confusão e métricas para um limiar fixo, com regra ``score >= threshold``."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return ThresholdMetrics(
        tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, f1=f1, fpr=fpr
    )
