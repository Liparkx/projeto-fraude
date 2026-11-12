"""Carregamento do dataset e divisão estratificada treino/validação/teste."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

TARGET_COLUMN = "Class"
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class DataSplit:
    """Conjuntos de treino, validação e teste, já separados em X e y."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_data(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Lê o CSV do ULB e valida a presença da coluna alvo.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se a coluna alvo não existir ou não for binária.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. Baixe creditcard.csv do Kaggle "
            "e coloque em data/raw/."
        )

    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET_COLUMN}' ausente no dataset.")
    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Coluna '{TARGET_COLUMN}' deve conter apenas 0 e 1.")

    return df


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DataSplit:
    """Divide o DataFrame em treino, validação e teste de forma estratificada.

    Deve ser chamada antes de qualquer transformação (escalonamento, resampling),
    para que nenhuma informação dos conjuntos de validação/teste vaze para o treino.

    Args:
        df: DataFrame completo, contendo a coluna alvo.
        val_size: fração do total destinada à validação.
        test_size: fração do total destinada ao teste.
        random_state: semente para reprodutibilidade.
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size e test_size devem estar entre 0 e 1.")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size deve ser menor que 1.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Converte a fração do total em fração do restante (treino + validação).
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=relative_val_size,
        stratify=y_trainval,
        random_state=random_state,
    )

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
