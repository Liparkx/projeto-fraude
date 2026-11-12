"""Modelos e montagem dos pipelines para cada estratégia de desbalanceamento."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraud_detection.data import DEFAULT_RANDOM_STATE
from fraud_detection.sampling import SamplingStrategy, build_preprocessor, build_sampler

ModelName = Literal["logistic", "xgboost"]

MODEL_NAMES: tuple[ModelName, ...] = ("logistic", "xgboost")
SAMPLING_STRATEGIES: tuple[SamplingStrategy, ...] = (
    "baseline",
    "smote",
    "undersampling",
    "class_weight",
)


def _negative_positive_ratio(y_train: pd.Series) -> float:
    """Razão entre legítimas e fraudes no treino, usada como peso da classe positiva."""
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    if n_positive == 0:
        raise ValueError("y_train não contém exemplos positivos (fraudes).")
    return n_negative / n_positive


def build_classifier(
    model_name: ModelName,
    strategy: SamplingStrategy,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Cria o classificador, aplicando class weights apenas na estratégia 'class_weight'."""
    use_class_weight = strategy == "class_weight"

    if model_name == "logistic":
        return LogisticRegression(
            class_weight="balanced" if use_class_weight else None,
            max_iter=1000,
            random_state=random_state,
        )
    if model_name == "xgboost":
        scale_pos_weight = _negative_positive_ratio(y_train) if use_class_weight else 1.0
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Modelo desconhecido: {model_name!r}")


def build_pipeline(
    model_name: ModelName,
    strategy: SamplingStrategy,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Monta o pipeline completo: pré-processamento → reamostragem → classificador.

    Ordem dos passos:
    1. Escalonamento de Time e Amount (ajustado só no treino).
    2. Reamostragem (SMOTE / undersampling), aplicada somente no fit.
    3. Classificador.

    Args:
        model_name: 'logistic' ou 'xgboost'.
        strategy: 'baseline', 'smote', 'undersampling' ou 'class_weight'.
        y_train: alvo do conjunto de treino (usado para calcular class weights).
        random_state: semente para todas as operações estocásticas.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("sampler", build_sampler(strategy, random_state=random_state)),
            ("classifier", build_classifier(model_name, strategy, y_train, random_state)),
        ]
    )
