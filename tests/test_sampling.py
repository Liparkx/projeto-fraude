"""Testes do pré-processamento e da reamostragem."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from fraud_detection.sampling import SCALED_COLUMNS, build_preprocessor, build_sampler


def _make_features(n_rows: int = 1_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"Time": rng.uniform(0, 172_000, n_rows), "V1": rng.normal(size=n_rows)}
    data["V2"] = rng.normal(loc=5.0, scale=3.0, size=n_rows)
    data["Amount"] = rng.exponential(80, n_rows)
    return pd.DataFrame(data)


def test_preprocessor_scales_only_time_and_amount() -> None:
    X = _make_features()
    out = build_preprocessor().fit_transform(X)

    # Ordem de saída: Time, Amount (escalonadas) seguidas das demais colunas.
    scaled_out = out[:, : len(SCALED_COLUMNS)]
    passthrough_out = out[:, len(SCALED_COLUMNS) :]

    np.testing.assert_allclose(passthrough_out, X[["V1", "V2"]].to_numpy())
    assert not np.allclose(scaled_out, X[list(SCALED_COLUMNS)].to_numpy())


def test_preprocessor_fit_on_train_does_not_use_validation_statistics() -> None:
    X_train = _make_features(seed=1)
    X_val = _make_features(seed=2)
    X_val["Amount"] = X_val["Amount"] * 1_000  # distribuição muito diferente

    preprocessor = build_preprocessor().fit(X_train)
    out_val = preprocessor.transform(X_val)

    # Transformação deve usar mediana/IQR do treino, não da validação.
    scaler = preprocessor.named_transformers_["scale"]
    expected_amount = (X_val["Amount"].to_numpy() - scaler.center_[1]) / scaler.scale_[1]
    np.testing.assert_allclose(out_val[:, 1], expected_amount)


def test_build_sampler_returns_expected_types() -> None:
    assert isinstance(build_sampler("smote"), SMOTE)
    assert isinstance(build_sampler("undersampling"), RandomUnderSampler)
    assert build_sampler("baseline") == "passthrough"
    assert build_sampler("class_weight") == "passthrough"


def test_build_sampler_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="desconhecida"):
        build_sampler("oversample_magico")  # type: ignore[arg-type]


def test_samplers_use_fixed_random_state() -> None:
    assert build_sampler("smote", random_state=123).random_state == 123
    assert build_sampler("undersampling", random_state=123).random_state == 123
