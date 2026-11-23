"""Experimento ponta a ponta: compara as 8 combinações, fixa o limiar por custo de negócio
e avalia a combinação escolhida uma única vez no teste.

Fluxo:
1. Carrega os dados (sem duplicatas) e faz o split estratificado 60/20/20.
2. Treina cada combinação (modelo × estratégia) só no treino e avalia na validação:
   PR-AUC, recall a FPR fixa (0,1% e 1%) e custo de negócio (R$) no limiar ótimo.
3. Seleciona a combinação de menor custo esperado na validação (ou, se configurado,
   a de maior PR-AUC). A líder por PR-AUC e o ranking por recall são sempre reportados.
4. Fixa o limiar ótimo da validação e o aplica ao teste uma única vez, sem reotimizá-lo.
5. Salva tabelas, figuras e o modelo com o limiar fixado.

Os custos são premissas editáveis em ``ExperimentConfig``:
- C_FN = valor da transação + taxa fixa de fraude (``fraud_fixed_cost``);
- C_FP = custo fixo de bloquear uma transação legítima (``fp_cost``).

Executar: ``python -m fraud_detection.experiment``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

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
    recall_at_fpr_bootstrap_ci,
    roc_auc,
)
from fraud_detection.models import MODEL_NAMES, SAMPLING_STRATEGIES, build_pipeline
from fraud_detection.reporting import (
    plot_cost_by_model,
    plot_operating_cost_curve,
    plot_threshold_by_fp_cost,
    save_figures,
)
from fraud_detection.threshold import (
    DEFAULT_COST_RATIOS,
    DEFAULT_NEAR_OPTIMAL_TOLERANCE,
    DEFAULT_THRESHOLD,
    best_threshold,
    cost_at_threshold,
    cost_scenario_sensitivity,
    near_optimal_range,
    sensitivity_analysis,
    transaction_cost_arrays,
)

logger = logging.getLogger(__name__)

SELECTION_CRITERIA = ("validation_cost", "pr_auc")
COST_UNIT_NOTE = "unidade monetária do dataset ULB (moeda não informada); custos são premissas"


@dataclass(frozen=True)
class ExperimentConfig:
    """Parâmetros do experimento. Os custos de negócio são premissas editáveis."""

    data_path: Path = DEFAULT_DATA_PATH
    reports_dir: Path = PROJECT_ROOT / "reports"
    models_dir: Path = PROJECT_ROOT / "models"
    random_state: int = DEFAULT_RANDOM_STATE
    val_size: float = 0.2
    test_size: float = 0.2
    fpr_targets: tuple[float, ...] = (0.001, 0.01)
    # FPR principal: indicativa (~57 falsos positivos permitidos na validação).
    primary_fpr: float = 0.001
    # Custos de negócio, na unidade monetária do dataset (premissas).
    fraud_fixed_cost: float = 25.0
    fp_cost: float = 5.0
    # Cenários de sensibilidade do custo de negócio.
    scenario_fraud_fixed_costs: tuple[float, ...] = (0.0, 25.0, 100.0)
    scenario_fp_costs: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0)
    near_optimal_tolerance: float = DEFAULT_NEAR_OPTIMAL_TOLERANCE
    # "validation_cost": menor custo esperado na validação. "pr_auc": maior PR-AUC.
    selection_criterion: str = "validation_cost"
    # Sensibilidade relativa por razão C_FN / C_FP (escala relativa, sem R$).
    cost_ratios: tuple[float, ...] = DEFAULT_COST_RATIOS
    c_fp: float = 1.0
    default_threshold: float = DEFAULT_THRESHOLD
    n_bootstrap: int = 200


@dataclass
class ExperimentResults:
    """Resultados do experimento, em memória e salvos em disco."""

    validation_comparison: pd.DataFrame
    validation_cost_by_model: pd.DataFrame
    validation_sensitivity: pd.DataFrame
    test_results: pd.DataFrame
    test_operating_point: pd.DataFrame
    test_sensitivity: pd.DataFrame
    cost_scenarios: pd.DataFrame
    selected_model: str
    selected_strategy: str
    selection_rule: str
    pr_auc_leader: tuple[str, str]
    operating_threshold: float
    operating_range: tuple[float, float]
    ci_overlap: bool
    best_by_recall: tuple[str, str]
    rankings_disagree: bool
    y_val: np.ndarray
    amounts_val: np.ndarray
    val_scores: dict[str, np.ndarray] = field(default_factory=dict)
    pipeline: object | None = None
    output_files: list[Path] = field(default_factory=list)


def _combo_label(model_name: str, strategy: str) -> str:
    return f"{model_name}/{strategy}"


def _fpr_label(fpr: float) -> str:
    """Rótulo legível para colunas: 0.001 -> '0.1pct', 0.01 -> '1pct'."""
    return f"{fpr * 100:g}pct"


def _validate_config(config: ExperimentConfig) -> None:
    if config.primary_fpr not in config.fpr_targets:
        raise ValueError("primary_fpr deve estar em fpr_targets.")
    if config.selection_criterion not in SELECTION_CRITERIA:
        raise ValueError(f"selection_criterion deve ser um de {SELECTION_CRITERIA}.")
    if config.fraud_fixed_cost < 0 or config.fp_cost <= 0:
        raise ValueError("fraud_fixed_cost deve ser não negativo e fp_cost deve ser positivo.")
    if config.near_optimal_tolerance < 0:
        raise ValueError("near_optimal_tolerance deve ser não negativa.")
    if not config.scenario_fraud_fixed_costs or not config.scenario_fp_costs:
        raise ValueError("Os cenários de custo não podem ser vazios.")


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
        r_low, r_high = recall_at_fpr_bootstrap_ci(
            y, s, fpr, n_boot=config.n_bootstrap, random_state=config.random_state
        )
        row[f"recall_{label}"] = op.recall
        row[f"recall_{label}_ci_low"] = r_low
        row[f"recall_{label}_ci_high"] = r_high
        row[f"threshold_{label}"] = op.threshold
        row[f"fpr_{label}"] = op.fpr

    default = metrics_at_threshold(y, s, config.default_threshold)
    row["precision_default"] = default.precision
    row["recall_default"] = default.recall
    row["f1_default"] = default.f1
    return row


def _cost_row(
    model_name: str,
    strategy: str,
    y: np.ndarray,
    s: np.ndarray,
    amounts: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float | str]:
    """Limiar ótimo de custo na validação, com faixa quase-ótima e custos em R$."""
    c_fn, c_fp = transaction_cost_arrays(amounts, config.fraud_fixed_cost, config.fp_cost)
    opt = best_threshold(y, s, c_fn, c_fp)
    low, high = near_optimal_range(y, s, c_fn, c_fp, config.near_optimal_tolerance)
    no_model = float(c_fn[y == 1].sum())
    return {
        "model": model_name,
        "strategy": strategy,
        "threshold": opt.threshold,
        "range_low": low,
        "range_high": high,
        "val_cost": opt.cost,
        "val_fn": opt.metrics.fn,
        "val_fp": opt.metrics.fp,
        "val_recall": opt.metrics.recall,
        "val_precision": opt.metrics.precision,
        "val_cost_default": cost_at_threshold(y, s, config.default_threshold, c_fn, c_fp),
        "val_cost_no_model": no_model,
        "val_savings_vs_no_model": no_model - opt.cost,
    }


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
    """Métricas de ranking e de recall a FPR fixa no teste, com limiares da validação."""
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


def _operating_point_row(
    model_name: str,
    strategy: str,
    threshold: float,
    cost_row: dict[Any, Any],
    y_test: np.ndarray,
    s_test: np.ndarray,
    amounts_test: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float | str]:
    """Aplica o limiar fixado na validação ao teste (sem reotimizar) e mede o custo em R$."""
    c_fn, c_fp = transaction_cost_arrays(amounts_test, config.fraud_fixed_cost, config.fp_cost)
    m = metrics_at_threshold(y_test, s_test, threshold)
    test_cost = cost_at_threshold(y_test, s_test, threshold, c_fn, c_fp)
    no_model = float(c_fn[y_test == 1].sum())
    return {
        "model": model_name,
        "strategy": strategy,
        "threshold": threshold,
        "range_low": cost_row["range_low"],
        "range_high": cost_row["range_high"],
        "val_cost": cost_row["val_cost"],
        "test_cost": test_cost,
        "test_cost_default": cost_at_threshold(
            y_test, s_test, config.default_threshold, c_fn, c_fp
        ),
        "test_cost_no_model": no_model,
        "test_savings_vs_no_model": no_model - test_cost,
        "test_tp": m.tp,
        "test_fp": m.fp,
        "test_tn": m.tn,
        "test_fn": m.fn,
        "test_recall": m.recall,
        "test_precision": m.precision,
        "test_fpr": m.fpr,
    }


def _select(
    config: ExperimentConfig, comparison: pd.DataFrame, cost_table: pd.DataFrame
) -> tuple[tuple[str, str], str]:
    """Combinação escolhida e a regra usada."""
    if config.selection_criterion == "validation_cost":
        ordered = cost_table.merge(
            comparison[["model", "strategy", "pr_auc"]], on=["model", "strategy"]
        ).sort_values(["val_cost", "pr_auc"], ascending=[True, False], kind="stable")
        return (str(ordered.iloc[0]["model"]), str(ordered.iloc[0]["strategy"])), (
            "menor_custo_validacao"
        )
    ranked = comparison.sort_values("pr_auc", ascending=False, kind="stable")
    return (str(ranked.iloc[0]["model"]), str(ranked.iloc[0]["strategy"])), "maior_pr_auc"


def run_experiment(
    config: ExperimentConfig | None = None,
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> ExperimentResults:
    """Executa o experimento completo.

    Args:
        config: parâmetros; padrões se omitido.
        df: dados já carregados (com as colunas ``Amount`` e ``Class``). Se omitido,
            usa ``load_data`` com o caminho de ``config.data_path``.
        save: se True, grava tabelas, figuras e o modelo em disco.
    """
    config = config or ExperimentConfig()
    _validate_config(config)

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
    amounts_val = split.X_val["Amount"].to_numpy(dtype=float)
    amounts_test = split.X_test["Amount"].to_numpy(dtype=float)

    rows: list[dict[str, float | str]] = []
    cost_rows: list[dict[str, float | str]] = []
    sens_frames: list[pd.DataFrame] = []
    val_scores: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}

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
            cost_rows.append(_cost_row(model_name, strategy, y_val, s_val, amounts_val, config))

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
    cost_table = pd.DataFrame(cost_rows)
    validation_sensitivity = pd.concat(sens_frames, ignore_index=True)

    selected_key, selection_rule = _select(config, comparison, cost_table)
    ranked = comparison.sort_values("pr_auc", ascending=False, kind="stable").reset_index(drop=True)
    leader = ranked.iloc[0]
    leader_key = (str(leader["model"]), str(leader["strategy"]))
    is_leader = selected_key == leader_key
    selected_row = comparison[
        (comparison["model"] == selected_key[0]) & (comparison["strategy"] == selected_key[1])
    ].iloc[0]
    selected_rank = (
        int(
            ranked.index[
                (ranked["model"] == selected_key[0]) & (ranked["strategy"] == selected_key[1])
            ][0]
        )
        + 1
    )

    # Referência para o IC: a líder por PR-AUC, ou a segunda se a escolhida for a líder.
    reference = ranked.iloc[1] if is_leader else leader
    ci_overlap = bool(
        selected_row["pr_auc_ci_low"] <= reference["pr_auc_ci_high"]
        and reference["pr_auc_ci_low"] <= selected_row["pr_auc_ci_high"]
    )
    if ci_overlap:
        logger.warning(
            "Intervalos bootstrap de %s e %s se sobrepõem: diferença pode ser ruído.",
            _combo_label(*selected_key),
            _combo_label(reference["model"], reference["strategy"]),
        )
    if not is_leader:
        logger.info(
            "Escolhida %s (PR-AUC %.4f, posição %d) difere da líder %s (PR-AUC %.4f).",
            _combo_label(*selected_key),
            float(selected_row["pr_auc"]),
            selected_rank,
            _combo_label(*leader_key),
            float(leader["pr_auc"]),
        )

    # Ranking pelo recall na FPR principal, com desempate pela PR-AUC.
    primary_col = f"recall_{_fpr_label(config.primary_fpr)}"
    by_recall = comparison.sort_values(
        [primary_col, "pr_auc"], ascending=False, kind="stable"
    ).iloc[0]
    best_by_recall = (str(by_recall["model"]), str(by_recall["strategy"]))
    rankings_disagree = best_by_recall != leader_key
    comparison["rank_pr_auc"] = comparison["pr_auc"].rank(ascending=False, method="min").astype(int)
    comparison[f"rank_{primary_col}"] = (
        comparison[primary_col].rank(ascending=False, method="min").astype(int)
    )
    for table in (comparison, cost_table):
        table["selected"] = (table["model"] == selected_key[0]) & (
            table["strategy"] == selected_key[1]
        )
    if rankings_disagree:
        logger.warning(
            "Maior recall na FPR principal (%s) é %s/%s, diferente da líder por PR-AUC.",
            _fpr_label(config.primary_fpr),
            *best_by_recall,
        )

    selected_label = _combo_label(*selected_key)
    selected_pipe = fitted[selected_label]
    s_val_selected = val_scores[selected_label]
    selected_cost = cost_table[
        (cost_table["model"] == selected_key[0]) & (cost_table["strategy"] == selected_key[1])
    ].iloc[0]
    operating_threshold = float(selected_cost["threshold"])
    operating_range = (float(selected_cost["range_low"]), float(selected_cost["range_high"]))

    # Teste: usado uma única vez, só na combinação escolhida e só com o limiar da validação.
    s_test = selected_pipe.predict_proba(split.X_test)[:, 1]
    test_results = pd.DataFrame(
        [
            _test_row(
                selected_key[0],
                selected_key[1],
                float(selected_row["pr_auc"]),
                y_val,
                s_val_selected,
                y_test,
                s_test,
                config,
            )
        ]
    )
    test_results.insert(2, "selection_rule", selection_rule)
    test_operating_point = pd.DataFrame(
        [
            _operating_point_row(
                selected_key[0],
                selected_key[1],
                operating_threshold,
                selected_cost.to_dict(),
                y_test,
                s_test,
                amounts_test,
                config,
            )
        ]
    )
    test_sensitivity = sensitivity_analysis(
        y_val,
        s_val_selected,
        cost_ratios=config.cost_ratios,
        c_fp=config.c_fp,
        y_eval=y_test,
        s_eval=s_test,
        default_threshold=config.default_threshold,
    )
    test_sensitivity.insert(0, "strategy", selected_key[1])
    test_sensitivity.insert(0, "model", selected_key[0])

    cost_scenarios = cost_scenario_sensitivity(
        y_val,
        s_val_selected,
        amounts_val,
        fraud_fixed_costs=config.scenario_fraud_fixed_costs,
        fp_costs=config.scenario_fp_costs,
        tolerance=config.near_optimal_tolerance,
        default_threshold=config.default_threshold,
        y_eval=y_test,
        s_eval=s_test,
        amount_eval=amounts_test,
    )
    cost_scenarios.insert(0, "strategy", selected_key[1])
    cost_scenarios.insert(0, "model", selected_key[0])

    results = ExperimentResults(
        validation_comparison=comparison,
        validation_cost_by_model=cost_table,
        validation_sensitivity=validation_sensitivity,
        test_results=test_results,
        test_operating_point=test_operating_point,
        test_sensitivity=test_sensitivity,
        cost_scenarios=cost_scenarios,
        selected_model=selected_key[0],
        selected_strategy=selected_key[1],
        selection_rule=selection_rule,
        pr_auc_leader=leader_key,
        operating_threshold=operating_threshold,
        operating_range=operating_range,
        ci_overlap=ci_overlap,
        best_by_recall=best_by_recall,
        rankings_disagree=rankings_disagree,
        y_val=y_val,
        amounts_val=amounts_val,
        val_scores=val_scores,
        pipeline=selected_pipe,
    )

    if save:
        _save_outputs(results, config, s_val_selected)

    return results


def _save_outputs(
    results: ExperimentResults, config: ExperimentConfig, s_val_selected: np.ndarray
) -> None:
    tables_dir = config.reports_dir / "tables"
    figures_dir = config.reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "validation_comparison.csv": results.validation_comparison,
        "validation_cost_by_model.csv": results.validation_cost_by_model,
        "validation_sensitivity.csv": results.validation_sensitivity,
        "test_results.csv": results.test_results,
        "test_operating_point.csv": results.test_operating_point,
        "test_sensitivity.csv": results.test_sensitivity,
        "cost_scenarios.csv": results.cost_scenarios,
    }
    for name, table in tables.items():
        path = tables_dir / name
        table.to_csv(path, index=False)
        results.output_files.append(path)

    selected_label = _combo_label(results.selected_model, results.selected_strategy)
    figure_paths = save_figures(
        figures_dir=figures_dir,
        y_val=results.y_val,
        val_scores=results.val_scores,
        validation_sensitivity=results.validation_sensitivity,
        validation_comparison=results.validation_comparison,
        fpr_targets=config.fpr_targets,
        best_label=selected_label,
        best_val_scores=s_val_selected,
        cost_ratios=config.cost_ratios,
        c_fp=config.c_fp,
        default_threshold=config.default_threshold,
    )
    c_fn_val, c_fp_val = transaction_cost_arrays(
        results.amounts_val, config.fraud_fixed_cost, config.fp_cost
    )
    figure_paths.extend(
        [
            plot_operating_cost_curve(
                results.y_val,
                s_val_selected,
                c_fn_val,
                c_fp_val,
                threshold=results.operating_threshold,
                range_low=results.operating_range[0],
                range_high=results.operating_range[1],
                default_threshold=config.default_threshold,
                path=figures_dir / "custo_reais_limiar_selecionado.png",
            ),
            plot_cost_by_model(
                results.validation_cost_by_model,
                figures_dir / "custo_otimo_por_combinacao.png",
            ),
            plot_threshold_by_fp_cost(
                results.cost_scenarios,
                figures_dir / "limiar_por_cenario_de_custo.png",
            ),
        ]
    )
    results.output_files.extend(figure_paths)

    operating = results.test_operating_point.iloc[0]
    payload = {
        "pipeline": results.pipeline,
        "model_name": results.selected_model,
        "strategy": results.selected_strategy,
        "selection_rule": results.selection_rule,
        "pr_auc_leader": results.pr_auc_leader,
        "operating_threshold": results.operating_threshold,
        "operating_range": results.operating_range,
        "cost_scenario": {
            "fraud_fixed_cost": config.fraud_fixed_cost,
            "fp_cost": config.fp_cost,
            "unit_note": COST_UNIT_NOTE,
        },
        "fpr_thresholds": {
            col.removeprefix("threshold_"): float(results.test_results.iloc[0][col])
            for col in results.test_results.columns
            if col.startswith("threshold_")
        },
        "test_metrics": results.test_results.iloc[0].to_dict(),
        "test_operating_point": operating.to_dict(),
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
    """Ponto de entrada: executa o experimento e imprime um resumo curto (sem DataFrames)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = ExperimentConfig()
    results = run_experiment(config)

    cost_table = results.validation_cost_by_model
    selected = f"{results.selected_model}/{results.selected_strategy}"
    print(
        f"Custos (premissas): taxa fixa de fraude = {config.fraud_fixed_cost:g}; "
        f"custo de FP = {config.fp_cost:g} ({COST_UNIT_NOTE})."
    )
    print("Validação: limiar ótimo e custo por combinação (menor custo primeiro):")
    for _, row in cost_table.sort_values("val_cost").iterrows():
        mark = " <- escolhida" if row["selected"] else ""
        print(
            f"  {row['model']}/{row['strategy']}: limiar {row['threshold']:.4f} "
            f"[faixa {row['range_low']:.4f}; {row['range_high']:.4f}], "
            f"custo {row['val_cost']:,.2f} (FN {int(row['val_fn'])}, FP {int(row['val_fp'])})"
            f"{mark}"
        )
    print(f"Escolhida ({results.selection_rule}): {selected}")
    print(f"  Limiar fixado: {results.operating_threshold:.4f}")
    if results.pr_auc_leader != (results.selected_model, results.selected_strategy):
        lm, ls = results.pr_auc_leader
        print(f"  Líder por PR-AUC: {lm}/{ls}")
    print(f"  Intervalo sobreposto com a referência por PR-AUC: {results.ci_overlap}")
    print(
        f"  Rankings discordam (PR-AUC x recall a {_fpr_label(config.primary_fpr)}): "
        f"{results.rankings_disagree}"
    )

    op = results.test_operating_point.iloc[0]
    print("Teste (uma vez, limiar fixado na validação):")
    print(
        f"  custo {op['test_cost']:,.2f} | sem modelo {op['test_cost_no_model']:,.2f} | "
        f"limiar 0,5: {op['test_cost_default']:,.2f}"
    )
    print(
        f"  FN {int(op['test_fn'])}, FP {int(op['test_fp'])}, recall {op['test_recall']:.4f}, "
        f"precisão {op['test_precision']:.4f}"
    )
    print(f"Arquivos gerados: {len(results.output_files)} em reports/ e models/")


if __name__ == "__main__":
    main()
