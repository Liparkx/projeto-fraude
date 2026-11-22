"""Experimento ponta a ponta: compara as 8 combinações e avalia a escolhida no teste.

Fluxo:
1. Carrega os dados (sem duplicatas) e faz o split estratificado 60/20/20.
2. Treina cada combinação (modelo × estratégia) só no treino e avalia na validação.
3. Escolhe a de maior PR-AUC na validação e sinaliza se o intervalo bootstrap
   se sobrepõe ao da segunda colocada (diferença dentro do ruído).
4. Usa o conjunto de teste uma única vez, só na combinação escolhida, aplicando
   os limiares definidos na validação, sem reotimizá-los.
5. Salva tabelas, figuras e o modelo.

Executar: ``python -m fraud_detection.experiment``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.data import (
    DEFAULT_DATA_PATH,
    DEFAULT_RANDOM_STATE,
    PROJECT_ROOT,
    load_data,
    split_data,
)
from fraud_detection.metrics import (
    metrics_at_threshold,
    pr_auc,
    pr_auc_bootstrap_ci,
    recall_at_fpr,
    roc_auc,
)
from fraud_detection.models import MODEL_NAMES, SAMPLING_STRATEGIES, build_pipeline
from fraud_detection.reporting import save_figures
from fraud_detection.threshold import DEFAULT_COST_RATIOS, DEFAULT_THRESHOLD, sensitivity_analysis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    """Parâmetros do experimento. Nada é fixo no código além dos padrões."""

    data_path: Path = DEFAULT_DATA_PATH
    reports_dir: Path = PROJECT_ROOT / "reports"
    models_dir: Path = PROJECT_ROOT / "models"
    random_state: int = DEFAULT_RANDOM_STATE
    val_size: float = 0.2
    test_size: float = 0.2
    fpr_targets: tuple[float, ...] = (0.001, 0.01)
    cost_ratios: tuple[float, ...] = DEFAULT_COST_RATIOS
    c_fp: float = 1.0
    default_threshold: float = DEFAULT_THRESHOLD
    n_bootstrap: int = 200


@dataclass
class ExperimentResults:
    """Resultados do experimento, em memória e salvos em disco."""

    validation_comparison: pd.DataFrame
    validation_sensitivity: pd.DataFrame
    test_results: pd.DataFrame
    test_sensitivity: pd.DataFrame
    best_model: str
    best_strategy: str
    ci_overlap: bool
    y_val: np.ndarray
    val_scores: dict[str, np.ndarray] = field(default_factory=dict)
    pipeline: object | None = None
    output_files: list[Path] = field(default_factory=list)


def _combo_label(model_name: str, strategy: str) -> str:
    return f"{model_name}/{strategy}"


def _fpr_label(fpr: float) -> str:
    """Rótulo legível para colunas: 0.001 -> '0.1pct', 0.01 -> '1pct'."""
    return f"{fpr * 100:g}pct"


def _validation_row(
    model_name: str, strategy: str, y: np.ndarray, s: np.ndarray, config: ExperimentConfig
) -> dict[str, float | str]:
    ci_low, ci_high = pr_auc_bootstrap_ci(
        y, s, n_boot=config.n_bootstrap, random_state=config.random_state
    )
    row: dict[str, float | str] = {
        "model": model_name,
        "strategy": strategy,
        "pr_auc": pr_auc(y, s),
        "pr_auc_ci_low": ci_low,
        "pr_auc_ci_high": ci_high,
        "roc_auc": roc_auc(y, s),
    }
    for fpr in config.fpr_targets:
        op = recall_at_fpr(y, s, fpr)
        label = _fpr_label(fpr)
        row[f"recall_{label}"] = op.recall
        row[f"threshold_{label}"] = op.threshold
        row[f"fpr_{label}"] = op.fpr

    default = metrics_at_threshold(y, s, config.default_threshold)
    row["precision_default"] = default.precision
    row["recall_default"] = default.recall
    row["f1_default"] = default.f1
    return row


def _test_row(
    model_name: str,
    strategy: str,
    validation_pr_auc: float,
    y_val: np.ndarray,
    s_val: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float | str]:
    """Avalia no teste com limiares definidos na validação (sem reotimizar)."""
    row: dict[str, float | str] = {
        "model": model_name,
        "strategy": strategy,
        "validation_pr_auc": validation_pr_auc,
        "pr_auc": pr_auc(y_test, s_test),
        "roc_auc": roc_auc(y_test, s_test),
    }
    for fpr in config.fpr_targets:
        label = _fpr_label(fpr)
        threshold = recall_at_fpr(y_val, s_val, fpr).threshold
        m = metrics_at_threshold(y_test, s_test, threshold)
        row[f"threshold_{label}"] = threshold
        row[f"recall_{label}"] = m.recall
        row[f"fpr_{label}"] = m.fpr
        row[f"precision_{label}"] = m.precision
    return row


def run_experiment(
    config: ExperimentConfig | None = None,
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> ExperimentResults:
    """Executa o experimento completo.

    Args:
        config: parâmetros; padrões se omitido.
        df: dados já carregados (com a coluna ``Class``). Se omitido, usa
            ``load_data`` com o caminho de ``config.data_path``.
        save: se True, grava tabelas, figuras e o modelo em disco.
    """
    config = config or ExperimentConfig()

    if df is None:
        df = load_data(config.data_path)
    split = split_data(
        df,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    y_val = split.y_val.to_numpy()
    y_test = split.y_test.to_numpy()

    rows: list[dict[str, float | str]] = []
    sens_frames: list[pd.DataFrame] = []
    val_scores: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}

    for model_name in MODEL_NAMES:
        for strategy in SAMPLING_STRATEGIES:
            label = _combo_label(model_name, strategy)
            logger.info("Treinando %s", label)

            pipe = build_pipeline(model_name, strategy, split.y_train, config.random_state)
            pipe.fit(split.X_train, split.y_train)
            s_val = pipe.predict_proba(split.X_val)[:, 1]

            val_scores[label] = s_val
            fitted[label] = pipe
            rows.append(_validation_row(model_name, strategy, y_val, s_val, config))

            sens = sensitivity_analysis(
                y_val,
                s_val,
                cost_ratios=config.cost_ratios,
                c_fp=config.c_fp,
                default_threshold=config.default_threshold,
            )
            sens.insert(0, "strategy", strategy)
            sens.insert(0, "model", model_name)
            sens_frames.append(sens)

    comparison = pd.DataFrame(rows)
    validation_sensitivity = pd.concat(sens_frames, ignore_index=True)

    # Ranking pela PR-AUC na validação. Se o intervalo bootstrap do vencedor se
    # sobrepõe ao da segunda colocada, a diferença está dentro do ruído.
    ranked = comparison.sort_values("pr_auc", ascending=False, kind="stable").reset_index(drop=True)
    best, second = ranked.iloc[0], ranked.iloc[1]
    ci_overlap = bool(
        best["pr_auc_ci_low"] <= second["pr_auc_ci_high"]
        and second["pr_auc_ci_low"] <= best["pr_auc_ci_high"]
    )
    if ci_overlap:
        logger.warning(
            "Intervalos bootstrap de %s e %s se sobrepõem: diferença pode ser ruído.",
            _combo_label(best["model"], best["strategy"]),
            _combo_label(second["model"], second["strategy"]),
        )

    best_name, best_strategy = str(best["model"]), str(best["strategy"])
    best_label = _combo_label(best_name, best_strategy)
    best_pipe = fitted[best_label]
    s_val_best = val_scores[best_label]

    # Teste: usado uma única vez, só na combinação escolhida.
    s_test = best_pipe.predict_proba(split.X_test)[:, 1]
    test_results = pd.DataFrame(
        [
            _test_row(
                best_name,
                best_strategy,
                float(best["pr_auc"]),
                y_val,
                s_val_best,
                y_test,
                s_test,
                config,
            )
        ]
    )
    test_sensitivity = sensitivity_analysis(
        y_val,
        s_val_best,
        cost_ratios=config.cost_ratios,
        c_fp=config.c_fp,
        y_eval=y_test,
        s_eval=s_test,
        default_threshold=config.default_threshold,
    )
    test_sensitivity.insert(0, "strategy", best_strategy)
    test_sensitivity.insert(0, "model", best_name)

    results = ExperimentResults(
        validation_comparison=comparison,
        validation_sensitivity=validation_sensitivity,
        test_results=test_results,
        test_sensitivity=test_sensitivity,
        best_model=best_name,
        best_strategy=best_strategy,
        ci_overlap=ci_overlap,
        y_val=y_val,
        val_scores=val_scores,
        pipeline=best_pipe,
    )

    if save:
        _save_outputs(results, config, s_val_best)

    return results


def _save_outputs(
    results: ExperimentResults, config: ExperimentConfig, s_val_best: np.ndarray
) -> None:
    tables_dir = config.reports_dir / "tables"
    figures_dir = config.reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "validation_comparison.csv": results.validation_comparison,
        "validation_sensitivity.csv": results.validation_sensitivity,
        "test_results.csv": results.test_results,
        "test_sensitivity.csv": results.test_sensitivity,
    }
    for name, table in tables.items():
        path = tables_dir / name
        table.to_csv(path, index=False)
        results.output_files.append(path)

    figure_paths = save_figures(
        figures_dir=figures_dir,
        y_val=results.y_val,
        val_scores=results.val_scores,
        validation_sensitivity=results.validation_sensitivity,
        best_label=_combo_label(results.best_model, results.best_strategy),
        best_val_scores=s_val_best,
        cost_ratios=config.cost_ratios,
        c_fp=config.c_fp,
        default_threshold=config.default_threshold,
    )
    results.output_files.extend(figure_paths)

    payload = {
        "pipeline": results.pipeline,
        "model_name": results.best_model,
        "strategy": results.best_strategy,
        "fpr_thresholds": {
            col.removeprefix("threshold_"): float(results.test_results.iloc[0][col])
            for col in results.test_results.columns
            if col.startswith("threshold_")
        },
        "test_metrics": results.test_results.iloc[0].to_dict(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.__dict__.items()
        },
        "random_state": config.random_state,
        "versions": {
            name: version(name)
            for name in ("scikit-learn", "imbalanced-learn", "xgboost", "pandas", "numpy")
        },
    }
    model_path = config.models_dir / "best_model.joblib"
    joblib.dump(payload, model_path)
    results.output_files.append(model_path)
    logger.info("Artefatos salvos: %d arquivos.", len(results.output_files))


def main() -> None:
    """Ponto de entrada: executa o experimento e imprime um resumo curto."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = run_experiment()

    best_row = results.validation_comparison.set_index(["model", "strategy"]).loc[
        (results.best_model, results.best_strategy)
    ]
    test_row = results.test_results.iloc[0]
    print(f"Melhor combinação (validação): {results.best_model}/{results.best_strategy}")
    print(
        f"  PR-AUC validação: {best_row['pr_auc']:.4f} "
        f"[IC95%: {best_row['pr_auc_ci_low']:.4f}; {best_row['pr_auc_ci_high']:.4f}]"
    )
    print(f"  Intervalos sobrepostos com a segunda colocada: {results.ci_overlap}")
    print(f"  PR-AUC teste: {test_row['pr_auc']:.4f}")
    print(f"  Arquivos gerados: {len(results.output_files)} em reports/ e models/")


if __name__ == "__main__":
    main()
