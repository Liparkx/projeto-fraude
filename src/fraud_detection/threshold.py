"""Seleção do limiar de decisão pelo custo de negócio.

Custo total = C_FN * FN + C_FP * FP, em que:
- C_FN: custo de uma fraude não detectada (falso negativo).
- C_FP: custo de bloquear uma transação legítima (falso positivo).

Os custos são parâmetros configuráveis. A regra de decisão é ``score >= limiar``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fraud_detection.metrics import ThresholdMetrics, metrics_at_threshold

# Razões C_FN / C_FP usadas na análise de sensibilidade.
DEFAULT_COST_RATIOS: tuple[float, ...] = (1, 5, 10, 50, 100, 500)

# Limiar de referência: o padrão de 0,5 de classificadores probabilísticos.
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class CostOptimum:
    """Limiar de custo mínimo e as métricas correspondentes."""

    threshold: float
    cost: float
    metrics: ThresholdMetrics


def total_cost(fn: int, fp: int, c_fn: float, c_fp: float) -> float:
    """Custo total dado o número de falsos negativos e falsos positivos."""
    return c_fn * fn + c_fp * fp


def cost_at_threshold(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float, c_fn: float, c_fp: float
) -> float:
    """Custo total obtido ao aplicar um limiar fixo."""
    _validate_costs(c_fn, c_fp)
    m = metrics_at_threshold(y_true, y_score, threshold)
    return total_cost(m.fn, m.fp, c_fn, c_fp)


def cost_curve(y_true: ArrayLike, y_score: ArrayLike, c_fn: float, c_fp: float) -> pd.DataFrame:
    """Matriz de confusão e custo para cada limiar candidato.

    Os candidatos são cada score distinto e ``inf`` (nenhuma transação bloqueada).
    As linhas estão em ordem decrescente de limiar. Os contadores são calculados
    com somas acumuladas, sem recalcular a predição para cada limiar.

    Returns:
        DataFrame com as colunas ``threshold, tp, fp, tn, fn, cost``.
    """
    _validate_costs(c_fn, c_fp)
    y, s = _validate_inputs(y_true, y_score)

    order = np.argsort(s, kind="stable")
    s_asc = s[order]
    y_asc = y[order]

    # cum_pos[i] e cum_neg[i] contam positivos e negativos entre os i menores scores.
    cum_pos = np.concatenate(([0], np.cumsum(y_asc)))
    cum_neg = np.concatenate(([0], np.cumsum(1 - y_asc)))
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    # Limiar t bloqueia os scores >= t, que começam em searchsorted(s_asc, t, 'left').
    candidates = np.concatenate(([np.inf], np.unique(s)[::-1]))
    idx = np.searchsorted(s_asc, candidates, side="left")

    tp = n_pos - cum_pos[idx]
    fp = n_neg - cum_neg[idx]
    fn = n_pos - tp
    tn = n_neg - fp

    return pd.DataFrame(
        {
            "threshold": candidates,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "cost": c_fn * fn + c_fp * fp,
        }
    )


def best_threshold(y_true: ArrayLike, y_score: ArrayLike, c_fn: float, c_fp: float) -> CostOptimum:
    """Limiar que minimiza ``C_FN * FN + C_FP * FP``.

    Em caso de empate, escolhe o maior limiar, ou seja, o que bloqueia menos
    transações. Deve ser usado na validação; o resultado é então reaplicado no teste.
    """
    curve = cost_curve(y_true, y_score, c_fn, c_fp)

    # A curva está em ordem decrescente de limiar, e idxmin devolve a primeira
    # ocorrência do mínimo, que é o maior limiar entre os empatados.
    best = curve["cost"].idxmin()
    threshold = float(curve.at[best, "threshold"])
    y, s = _validate_inputs(y_true, y_score)

    return CostOptimum(
        threshold=threshold,
        cost=float(curve.at[best, "cost"]),
        metrics=metrics_at_threshold(y, s, threshold),
    )


def bayes_threshold(c_fn: float, c_fp: float) -> float:
    """Limiar teórico ``C_FP / (C_FP + C_FN)``.

    É o ótimo para probabilidades bem calibradas. Serve como referência para o
    limiar empírico, e não substitui a otimização na validação.
    """
    _validate_costs(c_fn, c_fp)
    return c_fp / (c_fp + c_fn)


def sensitivity_analysis(
    y_val: ArrayLike,
    s_val: ArrayLike,
    cost_ratios: Sequence[float] = DEFAULT_COST_RATIOS,
    c_fp: float = 1.0,
    y_eval: ArrayLike | None = None,
    s_eval: ArrayLike | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Compara limiares ótimos para várias razões C_FN / C_FP.

    Para cada razão, otimiza o limiar na validação e registra também o custo com
    o limiar padrão (0,5) e o custo de não usar modelo (deixar passar tudo, C_FN·P).
    Se ``y_eval``/``s_eval`` forem informados, aplica o limiar da validação a eles
    sem reotimizar, como no conjunto de teste.

    Args:
        y_val, s_val: rótulos e scores da validação, usados para otimizar.
        cost_ratios: razões C_FN / C_FP a testar.
        c_fp: custo de um falso positivo (positivo). C_FN = razão · c_fp.
        y_eval, s_eval: rótulos e scores do conjunto de avaliação (opcional).
        default_threshold: limiar de referência para comparação.

    Returns:
        DataFrame com uma linha por razão.
    """
    if c_fp <= 0:
        raise ValueError("c_fp deve ser positivo.")
    if (y_eval is None) != (s_eval is None):
        raise ValueError("y_eval e s_eval devem ser informados juntos.")
    if len(cost_ratios) == 0:
        raise ValueError("cost_ratios não pode ser vazio.")

    y, s = _validate_inputs(y_val, s_val)
    n_pos = int(y.sum())
    if y_eval is not None and s_eval is not None:
        y_e, s_e = _validate_inputs(y_eval, s_eval)

    rows = []
    for ratio in cost_ratios:
        if ratio < 0:
            raise ValueError("As razões de custo devem ser não negativas.")
        c_fn = ratio * c_fp

        opt = best_threshold(y, s, c_fn, c_fp)
        default = metrics_at_threshold(y, s, default_threshold)

        row: dict[str, float] = {
            "ratio": ratio,
            "c_fn": c_fn,
            "c_fp": c_fp,
            "threshold": opt.threshold,
            "bayes_threshold": bayes_threshold(c_fn, c_fp),
            "val_cost": opt.cost,
            "val_fn": opt.metrics.fn,
            "val_fp": opt.metrics.fp,
            "val_recall": opt.metrics.recall,
            "val_precision": opt.metrics.precision,
            "val_cost_default": total_cost(default.fn, default.fp, c_fn, c_fp),
            "val_cost_no_model": c_fn * n_pos,
        }

        if y_eval is not None and s_eval is not None:
            m = metrics_at_threshold(y_e, s_e, opt.threshold)
            row.update(
                eval_cost=total_cost(m.fn, m.fp, c_fn, c_fp),
                eval_fn=m.fn,
                eval_fp=m.fp,
                eval_recall=m.recall,
                eval_precision=m.precision,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _validate_costs(c_fn: float, c_fp: float) -> None:
    if c_fn < 0 or c_fp < 0:
        raise ValueError("Os custos devem ser não negativos.")
    if c_fn == 0 and c_fp == 0:
        raise ValueError("Ao menos um dos custos deve ser positivo.")


def _validate_inputs(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true e y_score devem ser unidimensionais.")
    if len(y) != len(s):
        raise ValueError("y_true e y_score têm tamanhos diferentes.")
    if len(y) == 0:
        raise ValueError("As entradas não podem ser vazias.")
    if not set(np.unique(y).tolist()).issubset({0, 1}):
        raise ValueError("y_true deve conter apenas 0 e 1.")

    return y.astype(int), s
