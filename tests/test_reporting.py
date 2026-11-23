"""Testes das figuras do experimento."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.models import MODEL_NAMES, SAMPLING_STRATEGIES
from fraud_detection.reporting import (
    plot_cost_by_model,
    plot_cost_curves,
    plot_operating_cost_curve,
    plot_pr_curves,
    plot_recall_at_fpr,
    plot_sensitivity_summary,
    plot_threshold_by_fp_cost,
    save_figures,
)


def _comparison() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1)
    for m in MODEL_NAMES:
        for s in SAMPLING_STRATEGIES:
            row = {"model": m, "strategy": s}
            for label in ("0.1pct", "1pct"):
                value = float(rng.uniform(0.3, 0.9))
                row[f"recall_{label}"] = value
                row[f"recall_{label}_ci_low"] = value - 0.1
                row[f"recall_{label}_ci_high"] = min(1.0, value + 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


def _data(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    scores = {
        f"{m}/{s}": np.clip(rng.random(n) * 0.6 + 0.3 * y, 0, 1)
        for m in MODEL_NAMES
        for s in SAMPLING_STRATEGIES
    }
    return y, scores


def _sensitivity(model: str, strategy: str) -> pd.DataFrame:
    ratios = [1.0, 10.0, 100.0]
    return pd.DataFrame(
        {
            "model": model,
            "strategy": strategy,
            "ratio": ratios,
            "val_cost": [50.0, 20.0, 9.0],
            "val_cost_default": [60.0, 30.0, 15.0],
            "val_cost_no_model": [10.0, 100.0, 1000.0],
        }
    )


def test_pr_curves_figure_is_written(tmp_path: Path) -> None:
    y, scores = _data()
    path = plot_pr_curves(y, scores, tmp_path / "pr.png")

    assert path.is_file() and path.stat().st_size > 0


def test_cost_curves_figure_is_written(tmp_path: Path) -> None:
    y, scores = _data()
    path = plot_cost_curves(
        y,
        scores["xgboost/smote"],
        (1, 10, 100),
        c_fp=1.0,
        default_threshold=0.5,
        path=tmp_path / "cost.png",
    )

    assert path.is_file() and path.stat().st_size > 0


def test_sensitivity_figure_is_written(tmp_path: Path) -> None:
    path = plot_sensitivity_summary(_sensitivity("xgboost", "smote"), tmp_path / "sens.png")

    assert path.is_file() and path.stat().st_size > 0


def test_recall_at_fpr_figure_is_written(tmp_path: Path) -> None:
    path = plot_recall_at_fpr(_comparison(), (0.001, 0.01), tmp_path / "recall.png")

    assert path.is_file() and path.stat().st_size > 0


def test_save_figures_returns_four_existing_paths(tmp_path: Path) -> None:
    y, scores = _data()
    sens = pd.concat(
        [_sensitivity(m, s) for m in MODEL_NAMES for s in SAMPLING_STRATEGIES],
        ignore_index=True,
    )
    paths = save_figures(
        figures_dir=tmp_path / "figures",
        y_val=y,
        val_scores=scores,
        validation_sensitivity=sens,
        validation_comparison=_comparison(),
        fpr_targets=(0.001, 0.01),
        best_label="xgboost/smote",
        best_val_scores=scores["xgboost/smote"],
        cost_ratios=(1, 10, 100),
        c_fp=1.0,
        default_threshold=0.5,
    )

    assert len(paths) == 4
    assert all(p.is_file() for p in paths)


def test_operating_cost_curve_figure_is_written(tmp_path: Path) -> None:
    y, scores = _data()
    rng = np.random.default_rng(3)
    amounts = rng.exponential(80, len(y))
    c_fn = amounts + 25.0
    c_fp = np.full(len(y), 5.0)
    path = plot_operating_cost_curve(
        y,
        scores["xgboost/smote"],
        c_fn,
        c_fp,
        threshold=0.4,
        range_low=0.3,
        range_high=0.5,
        default_threshold=0.5,
        path=tmp_path / "operating.png",
    )

    assert path.is_file() and path.stat().st_size > 0


def test_cost_by_model_figure_is_written(tmp_path: Path) -> None:
    rows = []
    for i, m in enumerate(MODEL_NAMES):
        for j, s in enumerate(SAMPLING_STRATEGIES):
            rows.append(
                {
                    "model": m,
                    "strategy": s,
                    "val_cost": 1000.0 + 37.0 * (i * 4 + j),
                    "threshold": 0.1,
                    "range_low": 0.05,
                    "range_high": 0.2,
                    "val_fn": 3,
                    "val_fp": 40,
                    "selected": (i == 1 and j == 0),
                }
            )
    path = plot_cost_by_model(pd.DataFrame(rows), tmp_path / "cost_by_model.png")

    assert path.is_file() and path.stat().st_size > 0


def test_threshold_by_fp_cost_figure_is_written(tmp_path: Path) -> None:
    rows = [
        {"fraud_fixed_cost": ff, "fp_cost": fp, "threshold": 0.5 / (1 + fp * ff / 100 + 1)}
        for ff in (0.0, 25.0, 100.0)
        for fp in (1.0, 5.0, 20.0, 50.0)
    ]
    path = plot_threshold_by_fp_cost(pd.DataFrame(rows), tmp_path / "threshold_fp.png")

    assert path.is_file() and path.stat().st_size > 0
