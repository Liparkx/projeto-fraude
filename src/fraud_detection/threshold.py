"""Seleção do limiar de decisão pelo custo de negócio.

Custo total = custo dos falsos negativos + custo dos falsos positivos, em que:
- C_FN: custo de uma fraude não detectada (falso negativo). Pode ser um valor fixo
  ou um valor por transação, como ``Amount + taxa fixa``.
- C_FP: custo de bloquear uma transação legítima (falso positivo). Também pode ser
  fixo ou por linha.

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

# Tolerância padrão para a faixa quase-ótima: custo até 2% acima do mínimo.
DEFAULT_NEAR_OPTIMAL_TOLERANCE = 0.02


@dataclass(frozen=True)
class CostOptimum:
    """Limiar de custo mínimo e as métricas correspondentes."""

    threshold: float
    cost: float
    metrics: ThresholdMetrics


def total_cost(fn: int, fp: int, c_fn: float, c_fp: float) -> float:
    """Custo total dado o número de falsos negativos e falsos positivos (custos fixos)."""
    return c_fn * fn + c_fp * fp


def transaction_cost_arrays(
    amounts: ArrayLike, fraud_fixed_cost: float, fp_cost: float
) -> tuple[np.ndarray, np.ndarray]:
    """Custos por transação: C_FN = valor + taxa fixa de fraude; C_FP = custo fixo.

    Args:
        amounts: valor de cada transação (``Amount``), não negativo.
        fraud_fixed_cost: custo fixo adicional de uma fraude não detectada.
        fp_cost: custo fixo de bloquear uma transação legítima.

    Returns:
        (custo de FN por linha, custo de FP por linha), ambos com o tamanho de ``amounts``.
    """
    amounts_arr = np.asarray(amounts, dtype=float)
    if amounts_arr.ndim != 1:
        raise ValueError("amounts deve ser unidimensional.")
    if np.any(amounts_arr < 0):
        raise ValueError("Os valores das transações devem ser não negativos.")
    if fraud_fixed_cost < 0 or fp_cost < 0:
        raise ValueError("Os custos fixos devem ser não negativos.")
    c_fn = amounts_arr + fraud_fixed_cost
    c_fp = np.full(len(amounts_arr), float(fp_cost))
    return c_fn, c_fp


def cost_at_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float,
    c_fn: float | ArrayLike,
    c_fp: float | ArrayLike,
) -> float:
    """Custo total obtido ao aplicar um limiar fixo (custos escalares ou por linha)."""
    y, s = _validate_inputs(y_true, y_score)
    c_fn_r, c_fp_r = _per_row_costs(c_fn, c_fp, len(y))
    blocked = s >= threshold
    fn_cost = c_fn_r[(y == 1) & ~blocked].sum()
    fp_cost = c_fp_r[(y == 0) & blocked].sum()
    return float(fn_cost + fp_cost)


def cost_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    c_fn: float | ArrayLike,
    c_fp: float | ArrayLike,
) -> pd.DataFrame:
    """Matriz de confusão e custo para cada limiar candidato.

    Os candidatos são cada score distinto e ``inf`` (nenhuma transação bloqueada).
    As linhas estão em ordem decrescente de limiar. Os custos são calculados com
    somas acumuladas, sem recalcular a predição para cada limiar. Custos podem ser
    escalares ou um valor por linha.

    Returns:
        DataFrame com as colunas ``threshold, tp, fp, tn, fn, fn_cost, fp_cost, cost``.
    """
    y, s = _validate_inputs(y_true, y_score)
    c_fn_r, c_fp_r = _per_row_costs(c_fn, c_fp, len(y))

    order = np.argsort(s, kind="stable")
    s_asc = s[order]
    y_asc = y[order]
    fn_asc = c_fn_r[order]
    fp_asc = c_fp_r[order]

    # Acumulados entre os i menores scores: contagens e custos de FN e de FP.
    cum_pos = np.concatenate(([0], np.cumsum(y_asc)))
    cum_neg = np.concatenate(([0], np.cumsum(1 - y_asc)))
    cum_fn_cost = np.concatenate(([0.0], np.cumsum(fn_asc * y_asc)))
    cum_fp_cost = np.concatenate(([0.0], np.cumsum(fp_asc * (1 - y_asc))))
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    total_fp_cost = cum_fp_cost[-1]

    # Limiar t bloqueia os scores >= t, que começam em searchsorted(s_asc, t, 'left').
    # Os i menores scores ficam abaixo de t: são FN (se positivos) ou TN (se negativos).
    candidates = np.concatenate(([np.inf], np.unique(s)[::-1]))
    idx = np.searchsorted(s_asc, candidates, side="left")

    fn = cum_pos[idx]
    tn = cum_neg[idx]
    tp = n_pos - fn
    fp = n_neg - tn
    fn_cost = cum_fn_cost[idx]
    fp_cost = total_fp_cost - cum_fp_cost[idx]

    return pd.DataFrame(
        {
            "threshold": candidates,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "fn_cost": fn_cost,
            "fp_cost": fp_cost,
            "cost": fn_cost + fp_cost,
        }
    )


def best_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    c_fn: float | ArrayLike,
    c_fp: float | ArrayLike,
) -> CostOptimum:
    """Limiar que minimiza o custo total (custos escalares ou por linha).

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


def near_optimal_range(
    y_true: ArrayLike,
    y_score: ArrayLike,
    c_fn: float | ArrayLike,
    c_fp: float | ArrayLike,
    tolerance: float = DEFAULT_NEAR_OPTIMAL_TOLERANCE,
) -> tuple[float, float]:
    """Menor e maior limiar cujo custo está até ``tolerance`` acima do mínimo.

    Mostra a incerteza do limiar: a curva de custo costuma ser plana perto do ótimo,
    então há uma faixa de limiares quase tão boa quanto o mínimo exato.

    Returns:
        (limiar mínimo da faixa, limiar máximo da faixa).
    """
    if tolerance < 0:
        raise ValueError("tolerance deve ser não negativa.")
    curve = cost_curve(y_true, y_score, c_fn, c_fp)
    cost_min = float(curve["cost"].min())
    within = curve["cost"] <= cost_min * (1 + tolerance) + 1e-9
    thresholds = curve.loc[within, "threshold"]
    return float(thresholds.min()), float(thresholds.max())


def cost_scenario_sensitivity(
    y_val: ArrayLike,
    s_val: ArrayLike,
    amount_val: ArrayLike,
    fraud_fixed_costs: Sequence[float],
    fp_costs: Sequence[float],
    tolerance: float = DEFAULT_NEAR_OPTIMAL_TOLERANCE,
    default_threshold: float = DEFAULT_THRESHOLD,
    y_eval: ArrayLike | None = None,
    s_eval: ArrayLike | None = None,
    amount_eval: ArrayLike | None = None,
) -> pd.DataFrame:
    """Limiar ótimo e custos para cada cenário de (taxa fixa de fraude, custo de FP).

    Em cada cenário, C_FN = valor da transação + taxa fixa, e C_FP = custo fixo.
    O limiar é otimizado na validação. Se os dados de avaliação forem informados,
    o limiar da validação é aplicado a eles sem reotimizar.

    Returns:
        DataFrame com uma linha por cenário.
    """
    if (y_eval is None) != (s_eval is None) or (y_eval is None) != (amount_eval is None):
        raise ValueError("y_eval, s_eval e amount_eval devem ser informados juntos.")
    if len(fraud_fixed_costs) == 0 or len(fp_costs) == 0:
        raise ValueError("Os cenários não podem ser vazios.")

    y, s = _validate_inputs(y_val, s_val)
    if y_eval is not None:
        y_e, s_e = _validate_inputs(y_eval, s_eval)

    rows: list[dict[str, float]] = []
    for fraud_fixed in fraud_fixed_costs:
        for fp in fp_costs:
            c_fn, c_fp = transaction_cost_arrays(amount_val, fraud_fixed, fp)
            opt = best_threshold(y, s, c_fn, c_fp)
            low, high = near_optimal_range(y, s, c_fn, c_fp, tolerance)
            m_default = metrics_at_threshold(y, s, default_threshold)

            row: dict[str, float] = {
                "fraud_fixed_cost": float(fraud_fixed),
                "fp_cost": float(fp),
                "threshold": opt.threshold,
                "range_low": low,
                "range_high": high,
                "val_cost": opt.cost,
                "val_fn": opt.metrics.fn,
                "val_fp": opt.metrics.fp,
                "val_recall": opt.metrics.recall,
                "val_precision": opt.metrics.precision,
                "val_cost_default": cost_at_threshold(y, s, default_threshold, c_fn, c_fp),
                "val_cost_no_model": float(c_fn[y == 1].sum()),
                "val_fn_default": float(m_default.fn),
            }

            if y_eval is not None:
                c_fn_e, c_fp_e = transaction_cost_arrays(amount_eval, fraud_fixed, fp)
                m = metrics_at_threshold(y_e, s_e, opt.threshold)
                row.update(
                    eval_cost=cost_at_threshold(y_e, s_e, opt.threshold, c_fn_e, c_fp_e),
                    eval_cost_default=cost_at_threshold(
                        y_e, s_e, default_threshold, c_fn_e, c_fp_e
                    ),
                    eval_cost_no_model=float(c_fn_e[y_e == 1].sum()),
                    eval_fn=float(m.fn),
                    eval_fp=float(m.fp),
                    eval_recall=m.recall,
                    eval_precision=m.precision,
                )

            rows.append(row)

    return pd.DataFrame(rows)


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


def _per_row_costs(
    c_fn: float | ArrayLike, c_fp: float | ArrayLike, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Expande custos escalares ou por linha para arrays de tamanho ``n`` e valida."""
    c_fn_r = _expand(c_fn, n, "c_fn")
    c_fp_r = _expand(c_fp, n, "c_fp")
    if np.any(c_fn_r < 0) or np.any(c_fp_r < 0):
        raise ValueError("Os custos devem ser não negativos.")
    if not (np.any(c_fn_r > 0) or np.any(c_fp_r > 0)):
        raise ValueError("Ao menos um dos custos deve ser positivo.")
    return c_fn_r, c_fp_r


def _expand(cost: float | ArrayLike, n: int, name: str) -> np.ndarray:
    arr = np.asarray(cost, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} deve ser escalar ou ter o mesmo tamanho de y_true.")
    return arr


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
