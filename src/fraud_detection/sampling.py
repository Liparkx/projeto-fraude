"""Pré-processamento e técnicas de reamostragem para lidar com o desbalanceamento."""

from __future__ import annotations

from typing import Literal

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from fraud_detection.data import DEFAULT_RANDOM_STATE

# Apenas Time e Amount não são componentes PCA e precisam de escalonamento.
# V1 a V28 já estão em escala comparável e não devem ser alterados.
SCALED_COLUMNS: tuple[str, ...] = ("Time", "Amount")

SamplingStrategy = Literal["baseline", "smote", "undersampling", "class_weight"]


def build_preprocessor() -> ColumnTransformer:
    """Escalona Time e Amount; as demais colunas passam sem alteração.

    Deve ser usado dentro de um Pipeline, para ser ajustado apenas no treino.
    """
    return ColumnTransformer(
        transformers=[("scale", RobustScaler(), list(SCALED_COLUMNS))],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_sampler(strategy: SamplingStrategy, random_state: int = DEFAULT_RANDOM_STATE):
    """Retorna o reamostrador da estratégia, ou 'passthrough' quando não há reamostragem.

    - baseline: sem reamostragem.
    - smote: oversampling sintético da classe minoritária (fraude).
    - undersampling: reduz aleatoriamente a classe majoritária (legítima).
    - class_weight: sem reamostragem; o desbalanceamento é tratado pelo classificador.

    O reamostrador só deve ser executado dentro de imblearn.pipeline.Pipeline,
    que o aplica apenas durante o fit (treino), nunca na predição.
    """
    if strategy == "smote":
        return SMOTE(random_state=random_state)
    if strategy == "undersampling":
        return RandomUnderSampler(random_state=random_state)
    if strategy in ("baseline", "class_weight"):
        return "passthrough"
    raise ValueError(f"Estratégia desconhecida: {strategy!r}")
