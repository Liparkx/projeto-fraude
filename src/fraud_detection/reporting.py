"""Figuras do experimento, salvas em PNG com matplotlib.

Segue a paleta de referência do skill dataviz (modo claro, superfície #fcfcfb):
- Cada estratégia tem cor fixa, e a cor acompanha a estratégia em todos os painéis.
- Cada painel tem no máximo 4 séries, todas da ordem categórica validada.
- Linhas sólidas e finas; grade em hairline; sem eixo duplo.
- Legenda presente em todos os gráficos com 2 ou mais séries.
- As cores aqua e amarela ficam abaixo de 3:1 de contraste sobre a superfície clara,
  então cada figura tem a tabela CSV correspondente como versão tabular.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from fraud_detection.models import MODEL_NAMES, SAMPLING_STRATEGIES  # noqa: E402
from fraud_detection.threshold import best_threshold, cost_curve  # noqa: E402

# Tokens de cor da paleta de referência (modo claro).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"  # slot 1
ORANGE = "#eb6834"  # slot 2
AQUA = "#1baf7a"  # slot 3
YELLOW = "#eda100"  # slot 4

STRATEGY_COLORS = {
    "baseline": BLUE,
    "smote": ORANGE,
    "undersampling": AQUA,
    "class_weight": YELLOW,
}
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "smote": "SMOTE",
    "undersampling": "Undersampling",
    "class_weight": "Class weights",
}
MODEL_LABELS = {"logistic": "Regressão Logística", "xgboost": "XGBoost"}

FIGURE_DPI = 130


def _new_figure(nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] = (8, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _style(ax) -> None:
    """Aplica os tokens de cor: superfície, grade hairline e tinta secundária nos eixos."""
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curves(
    y_val: ArrayLike,
    val_scores: dict[str, np.ndarray],
    path: Path,
) -> Path:
    """Curvas Precisão-Recall das 4 estratégias, um painel por modelo."""
    y = np.asarray(y_val)
    prevalence = float(y.mean())
    fig, axes = _new_figure(1, 2, figsize=(12, 5))

    for ax, model_name in zip(axes[0], MODEL_NAMES, strict=True):
        for strategy in SAMPLING_STRATEGIES:
            scores = val_scores[f"{model_name}/{strategy}"]
            precision, recall, _ = precision_recall_curve(y, scores)
            ax.plot(
                recall,
                precision,
                color=STRATEGY_COLORS[strategy],
                linewidth=1.8,
                label=STRATEGY_LABELS[strategy],
            )
        # Linha de referência: classificador aleatório tem precisão igual à prevalência.
        ax.axhline(prevalence, color=INK_MUTED, linewidth=0.9)
        ax.text(
            0.99,
            prevalence,
            f" aleatório ({prevalence:.4f})",
            color=INK_SECONDARY,
            fontsize=8,
            va="bottom",
            ha="right",
            transform=ax.get_yaxis_transform(),
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisão")
        ax.set_title(f"{MODEL_LABELS[model_name]}: validação", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        _style(ax)

    fig.suptitle("Curvas Precisão-Recall por estratégia", color=INK_PRIMARY, fontsize=13)
    return _save(fig, path)


def plot_cost_curves(
    y_val: ArrayLike,
    scores: ArrayLike,
    cost_ratios: Sequence[float],
    c_fp: float,
    default_threshold: float,
    path: Path,
) -> Path:
    """Custo total em função do limiar, um painel por razão C_FN/C_FP."""
    n = len(cost_ratios)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = _new_figure(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))

    for ax, ratio in zip(axes.flat, cost_ratios, strict=False):
        c_fn = ratio * c_fp
        curve = cost_curve(y_val, scores, c_fn, c_fp)
        # Escala log no eixo x exige limiares positivos e finitos; o candidato inf fica de fora.
        finite = curve[np.isfinite(curve["threshold"]) & (curve["threshold"] > 0)]
        ax.plot(finite["threshold"], finite["cost"], color=BLUE, linewidth=1.5)

        opt = best_threshold(y_val, scores, c_fn, c_fp)
        ax.axvline(default_threshold, color=INK_MUTED, linewidth=0.9)
        if np.isfinite(opt.threshold) and opt.threshold > 0:
            ax.scatter([opt.threshold], [opt.cost], s=40, color=BLUE, zorder=3)
            ax.annotate(
                f"ótimo {opt.threshold:.4f}",
                xy=(opt.threshold, opt.cost),
                xytext=(6, 8),
                textcoords="offset points",
                color=INK_PRIMARY,
                fontsize=8,
            )

        ax.set_xscale("log")
        ax.set_title(f"C_FN / C_FP = {ratio:g}", fontsize=10)
        ax.set_xlabel("Limiar (escala log)")
        ax.set_ylabel("Custo total")
        _style(ax)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    # Evita que o rótulo do eixo x de uma linha colida com os títulos da linha de baixo.
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle(
        f"Custo × limiar na validação (linha cinza: limiar {default_threshold})",
        color=INK_PRIMARY,
        fontsize=12,
    )
    return _save(fig, path)


def plot_sensitivity_summary(sensitivity: pd.DataFrame, path: Path) -> Path:
    """Custo ótimo, custo com limiar 0,5 e custo sem modelo ao longo das razões."""
    fig, axes = _new_figure(figsize=(8, 5))
    ax = axes[0, 0]
    ratios = sensitivity["ratio"].to_numpy()

    ax.plot(
        ratios,
        sensitivity["val_cost"],
        "o-",
        color=BLUE,
        linewidth=1.8,
        markersize=6,
        label="Limiar ótimo",
    )
    ax.plot(
        ratios,
        sensitivity["val_cost_default"],
        "o-",
        color=ORANGE,
        linewidth=1.8,
        markersize=6,
        label="Limiar 0,5",
    )
    ax.plot(
        ratios,
        sensitivity["val_cost_no_model"],
        "o-",
        color=INK_MUTED,
        linewidth=1.8,
        markersize=6,
        label="Sem modelo",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Razão C_FN / C_FP (escala log)")
    ax.set_ylabel("Custo total na validação")
    ax.set_title("Sensibilidade do custo à razão de custos", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, path)


def plot_recall_at_fpr(
    validation_comparison: pd.DataFrame,
    fpr_targets: Sequence[float],
    path: Path,
) -> Path:
    """Recall a FPR fixa por combinação, com intervalo bootstrap de 95%.

    Um painel por FPR alvo. Barras agrupadas por modelo e coloridas pela estratégia,
    com barras de erro do intervalo. A FPR de 0,1% é indicativa: o limiar usa só
    algumas dezenas de falsos positivos permitidos, então o intervalo é largo.
    """
    fig, axes = _new_figure(1, len(fpr_targets), figsize=(6 * len(fpr_targets), 5))
    strategies = list(SAMPLING_STRATEGIES)
    width = 0.8 / len(strategies)

    for ax, fpr in zip(axes[0], fpr_targets, strict=True):
        label = f"{fpr * 100:g}pct"
        for j, strategy in enumerate(strategies):
            values, lows, highs, xs = [], [], [], []
            for i, model_name in enumerate(MODEL_NAMES):
                row = validation_comparison[
                    (validation_comparison["model"] == model_name)
                    & (validation_comparison["strategy"] == strategy)
                ].iloc[0]
                value = float(row[f"recall_{label}"])
                values.append(value)
                lows.append(value - float(row[f"recall_{label}_ci_low"]))
                highs.append(float(row[f"recall_{label}_ci_high"]) - value)
                xs.append(i + (j - (len(strategies) - 1) / 2) * width)
            ax.bar(
                xs,
                values,
                width=width * 0.9,
                color=STRATEGY_COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
            )
            ax.errorbar(
                xs,
                values,
                yerr=[lows, highs],
                fmt="none",
                ecolor=INK_PRIMARY,
                elinewidth=0.9,
                capsize=3,
            )

        ax.set_xticks(range(len(MODEL_NAMES)))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Recall")
        ax.set_title(f"Recall com FPR fixa de {fpr * 100:g}% (validação)", fontsize=11)
        _style(ax)

    # Legenda única abaixo dos painéis: sobre as barras ela cobria a cor do SMOTE.
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle("Recall a FPR fixa com IC 95% (bootstrap)", color=INK_PRIMARY, fontsize=13)
    return _save(fig, path)


def save_figures(
    figures_dir: Path,
    y_val: ArrayLike,
    val_scores: dict[str, np.ndarray],
    validation_sensitivity: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    fpr_targets: Sequence[float],
    best_label: str,
    best_val_scores: np.ndarray,
    cost_ratios: Sequence[float],
    c_fp: float,
    default_threshold: float,
) -> list[Path]:
    """Gera as quatro figuras do experimento e devolve os caminhos."""
    model_name, strategy = best_label.split("/")
    best_sens = validation_sensitivity[
        (validation_sensitivity["model"] == model_name)
        & (validation_sensitivity["strategy"] == strategy)
    ].sort_values("ratio")

    return [
        plot_pr_curves(y_val, val_scores, figures_dir / "pr_curves_validation.png"),
        plot_recall_at_fpr(
            validation_comparison,
            fpr_targets,
            figures_dir / "recall_em_fpr_fixa_validacao.png",
        ),
        plot_cost_curves(
            y_val,
            best_val_scores,
            cost_ratios,
            c_fp,
            default_threshold,
            figures_dir / "custo_x_limiar_melhor_modelo.png",
        ),
        plot_sensitivity_summary(best_sens, figures_dir / "sensibilidade_custo_melhor_modelo.png"),
    ]
