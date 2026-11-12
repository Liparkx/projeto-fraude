"""Testes de carregamento e divisão estratificada dos dados."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import DEFAULT_DATA_PATH, TARGET_COLUMN, load_data, split_data


def _make_synthetic_df(
    n_rows: int = 5_000, fraud_rate: float = 0.02, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    labels = np.zeros(n_rows, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)
    return pd.DataFrame(
        {
            "Time": rng.uniform(0, 172_000, n_rows),
            "V1": rng.normal(size=n_rows),
            "Amount": rng.exponential(80, n_rows),
            TARGET_COLUMN: labels,
        }
    )


def test_load_data_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "inexistente.csv")


def test_load_data_rejects_missing_target(tmp_path: Path) -> None:
    path = tmp_path / "sem_alvo.csv"
    pd.DataFrame({"Time": [0, 1], "Amount": [1.0, 2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Coluna alvo"):
        load_data(path)


def test_load_data_rejects_non_binary_target(tmp_path: Path) -> None:
    path = tmp_path / "alvo_ruim.csv"
    pd.DataFrame({"Amount": [1.0, 2.0], TARGET_COLUMN: [0, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="apenas 0 e 1"):
        load_data(path)


def test_split_sizes_match_requested_fractions() -> None:
    df = _make_synthetic_df()
    split = split_data(df, val_size=0.2, test_size=0.2, random_state=42)

    total = len(df)
    assert len(split.X_train) / total == pytest.approx(0.6, abs=0.01)
    assert len(split.X_val) / total == pytest.approx(0.2, abs=0.01)
    assert len(split.X_test) / total == pytest.approx(0.2, abs=0.01)
    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == total


def test_split_is_stratified() -> None:
    df = _make_synthetic_df(fraud_rate=0.02)
    overall_rate = df[TARGET_COLUMN].mean()
    split = split_data(df)

    for y in (split.y_train, split.y_val, split.y_test):
        assert y.mean() == pytest.approx(overall_rate, abs=0.003)


def test_split_has_no_overlapping_rows() -> None:
    df = _make_synthetic_df()
    split = split_data(df)

    train_idx = set(split.X_train.index)
    val_idx = set(split.X_val.index)
    test_idx = set(split.X_test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_split_separates_features_and_target() -> None:
    df = _make_synthetic_df()
    split = split_data(df)

    assert TARGET_COLUMN not in split.X_train.columns
    assert TARGET_COLUMN not in split.X_val.columns
    assert TARGET_COLUMN not in split.X_test.columns
    assert list(split.X_train.columns) == [c for c in df.columns if c != TARGET_COLUMN]


def test_split_is_reproducible_with_same_seed() -> None:
    df = _make_synthetic_df()
    first = split_data(df, random_state=7)
    second = split_data(df, random_state=7)

    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


@pytest.mark.parametrize(
    ("val_size", "test_size"),
    [(0.0, 0.2), (0.2, 1.0), (0.5, 0.5), (0.6, 0.5)],
)
def test_split_rejects_invalid_fractions(val_size: float, test_size: float) -> None:
    with pytest.raises(ValueError):
        split_data(_make_synthetic_df(), val_size=val_size, test_size=test_size)


@pytest.mark.skipif(not DEFAULT_DATA_PATH.is_file(), reason="creditcard.csv não disponível")
def test_real_dataset_shape_and_fraud_count() -> None:
    df = load_data()

    assert df.shape == (284_807, 31)
    assert int(df[TARGET_COLUMN].sum()) == 492
