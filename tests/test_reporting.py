"""Testes das figuras do experimento."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detection.models import MODEL_NAMES, SAMPLING_STRATEGIES
from fraud_detection.reporting import (
    plot_cost_curves,
    plot_pr_curves,
    plot_sensitivity_summary,
    save_figures,
)


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


def test_save_figures_returns_three_existing_paths(tmp_path: Path) -> None:
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
        best_label="xgboost/smote",
        best_val_scores=scores["xgboost/smote"],
        cost_ratios=(1, 10, 100),
        c_fp=1.0,
        default_threshold=0.5,
    )

    assert len(paths) == 3
    assert all(p.is_file() for p in paths)
