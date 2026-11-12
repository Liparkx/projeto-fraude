"""Testes da montagem dos pipelines e das estratégias de desbalanceamento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud_detection.models import MODEL_NAMES, SAMPLING_STRATEGIES, build_pipeline


def _make_dataset(n_rows: int = 2_000, fraud_rate: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    y = np.zeros(n_rows, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)
    X = pd.DataFrame(
        {
            "Time": rng.uniform(0, 172_000, n_rows),
            "V1": rng.normal(size=n_rows) + y * 2.0,  # sinal fraco para o modelo aprender
            "V2": rng.normal(size=n_rows),
            "Amount": rng.exponential(80, n_rows),
        }
    )
    return X, pd.Series(y, name="Class")


def test_all_combinations_are_built() -> None:
    _, y = _make_dataset()
    for model_name in MODEL_NAMES:
        for strategy in SAMPLING_STRATEGIES:
            pipe = build_pipeline(model_name, strategy, y)
            assert isinstance(pipe, Pipeline)
            assert [name for name, _ in pipe.steps] == ["preprocess", "sampler", "classifier"]


@pytest.mark.parametrize(
    ("strategy", "sampler_type"),
    [
        ("baseline", type(None)),
        ("class_weight", type(None)),
        ("smote", SMOTE),
        ("undersampling", RandomUnderSampler),
    ],
)
def test_sampler_step_matches_strategy(strategy: str, sampler_type: type) -> None:
    _, y = _make_dataset()
    pipe = build_pipeline("logistic", strategy, y)  # type: ignore[arg-type]
    sampler = pipe.named_steps["sampler"]

    if sampler_type is type(None):
        assert sampler == "passthrough"
    else:
        assert isinstance(sampler, sampler_type)


def test_logistic_class_weight_only_in_class_weight_strategy() -> None:
    _, y = _make_dataset()
    for strategy in SAMPLING_STRATEGIES:
        clf = build_pipeline("logistic", strategy, y).named_steps["classifier"]
        assert isinstance(clf, LogisticRegression)
        expected = "balanced" if strategy == "class_weight" else None
        assert clf.class_weight == expected


def test_xgboost_scale_pos_weight_matches_class_ratio_only_in_class_weight() -> None:
    _, y = _make_dataset(fraud_rate=0.05)
    expected_ratio = (y == 0).sum() / (y == 1).sum()

    for strategy in SAMPLING_STRATEGIES:
        clf = build_pipeline("xgboost", strategy, y).named_steps["classifier"]
        assert isinstance(clf, XGBClassifier)
        if strategy == "class_weight":
            assert clf.get_params()["scale_pos_weight"] == pytest.approx(expected_ratio)
        else:
            assert clf.get_params()["scale_pos_weight"] == 1.0


def test_build_pipeline_rejects_unknown_model() -> None:
    _, y = _make_dataset()
    with pytest.raises(ValueError, match="desconhecido"):
        build_pipeline("rede_neural", "baseline", y)  # type: ignore[arg-type]


def test_class_weight_requires_positive_examples() -> None:
    y = pd.Series(np.zeros(100, dtype=int))
    with pytest.raises(ValueError, match="positivos"):
        build_pipeline("xgboost", "class_weight", y)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("strategy", SAMPLING_STRATEGIES)
def test_pipeline_fits_and_predicts_probabilities(model_name: str, strategy: str) -> None:
    X, y = _make_dataset()
    X_train, y_train = X.iloc[:1_500], y.iloc[:1_500]
    X_val = X.iloc[1_500:]

    pipe = build_pipeline(model_name, strategy, y_train)  # type: ignore[arg-type]
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)

    # Reamostragem não deve alterar o número de linhas na predição.
    assert proba.shape == (len(X_val), 2)
    assert np.all((proba >= 0) & (proba <= 1))
