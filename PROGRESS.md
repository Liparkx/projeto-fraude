# Progresso do projeto

Registro do que foi feito em cada etapa, com os conceitos e erros encontrados. Serve como diário de aprendizado.

## Estado atual

- Repositório: https://github.com/Liparkx/projeto-fraude (público)
- Branch `main`: inclui o merge do PR #1 (`f1a6fbf`), com métricas e remoção de duplicatas.
- Branch `feat/threshold-por-custo`: Passo 4 implementado; ainda não commitado.
- Testes: 76 passando; `ruff check` e `ruff format --check` limpos.
- Próxima etapa: Passo 5 (experimento com as 8 combinações).

## Passos concluídos

### Passo 0: setup
- `pyproject.toml` com dependências de runtime, extra `dev` (pytest, ruff) e configuração do ruff.
- Instalação com `pip install -e .[dev]`.
- README corrigido: não existe `requirements.txt`; o `requirements-freeze.txt` está em UTF-16 e serve só como referência.

### Passo 1: `data.py`
- `load_data()` lê o CSV de `data/raw/`, valida a coluna `Class` e remove duplicatas.
- `split_data()` faz a divisão estratificada 60/20/20 com `random_state=42`, antes de qualquer transformação.

### Passo 2: `sampling.py` e `models.py`
- `build_preprocessor()`: `RobustScaler` apenas em `Time` e `Amount`. `V1` a `V28` não são alteradas, porque já são componentes PCA.
- `build_sampler()`: SMOTE, `RandomUnderSampler`, ou `"passthrough"` para baseline e class weights.
- `build_pipeline()`: `imblearn.pipeline.Pipeline` com pré-processamento, reamostragem e classificador. A reamostragem só roda no treino.
- 8 combinações: 4 estratégias (baseline, SMOTE, undersampling, class weights) × 2 modelos (Regressão Logística e XGBoost).

### Passo 3: `metrics.py`
- `pr_auc()`: área sob a curva Precision-Recall. É a métrica principal.
- `roc_auc()`: área sob a curva ROC. É a métrica complementar.
- `recall_at_fpr()`: maior recall com FPR ≤ alvo, e o limiar correspondente. O limiar é escolhido na validação e reaplicado no teste.
- `metrics_at_threshold()`: matriz de confusão (TP, FP, TN, FN), precision, recall, F1 e FPR para um limiar fixo.

### Passo 4: `threshold.py`
- `cost_curve()`: para cada score distinto (e para `inf`, que não bloqueia nada), calcula TP, FP, TN, FN e o custo `C_FN·FN + C_FP·FP`. Usa somas acumuladas, sem recalcular a predição para cada limiar.
- `best_threshold()`: limiar de custo mínimo. Em caso de empate, escolhe o maior limiar, ou seja, o que bloqueia menos transações.
- `bayes_threshold()`: referência teórica `C_FP / (C_FP + C_FN)`.
- `sensitivity_analysis()`: uma linha por razão C_FN/C_FP (1, 5, 10, 50, 100, 500). Compara o limiar ótimo com o limiar 0,5 e com "não usar modelo" (`C_FN · P`). Se receber dados de avaliação, aplica o limiar da validação a eles sem reotimizar.
- Custos são parâmetros; nada fica fixo no código.

### Remoção de duplicatas
- O dataset tem 284.807 linhas, das quais 1.081 são duplicatas exatas (19 delas fraudes).
- Após a remoção: 283.726 linhas e 473 fraudes (0,1667%).
- A remoção acontece antes do split, para que linhas idênticas não caiam em treino e teste ao mesmo tempo.
- Taxa de fraude por conjunto: treino 0,1668%, validação 0,1657%, teste 0,1674%.

## Pull requests

### PR #1: `feat/metricas-e-remocao-de-duplicatas` (mergeado)

Commits:
1. `ecf93b3`: adiciona `metrics.py` e seus testes.
2. `9e6659f`: remove duplicatas em `data.py`, atualiza os testes e o README.

Motivo de usar PR: documentar o aprendizado e revisar o diff antes do merge.

## Conceitos aprendidos

- **Acurácia engana em dados desbalanceados.** Com 0,17% de fraude, um modelo que classifica tudo como legítimo acerta 99,83%.
- **PR-AUC** é a soma em degraus de (Rₙ − Rₙ₋₁)·Pₙ. O baseline de um modelo aleatório é a prevalência (≈ 0,0017), e não 0,5.
- **ROC-AUC** equivale a P(score de fraude > score de legítimo). É insensível à prevalência, por isso é só complementar.
- **Recall em FPR fixa:** o maior recall com FPR ≤ α é o último ponto da curva ROC com FPR ≤ α. O limiar vem da validação.
- **`roc_curve(drop_intermediate=False)`:** com `True`, o sklearn remove pontos colineares, e o melhor limiar pode se perder.
- **Duplicatas antes do split:** duplicatas espalhadas entre treino e teste inflam as métricas.
- **Limiar de Bayes:** para probabilidades calibradas, o custo mínimo está no limiar `C_FP / (C_FP + C_FN)`. Com C_FN = 100·C_FP, o limiar cai para ~0,0099, bem abaixo de 0,5. Por isso o 0,5 mede calibração, e não custo.
- **Empate no limiar:** com custos iguais, dois limiares podem dar o mesmo custo. Escolher o maior evita bloqueios desnecessários.

## Erros e correções

- Nos testes de `metrics.py`, errei contas feitas à mão. O limiar de `recall_at_fpr` é o próprio score do ponto da curva (1.0, e não 0.95). No limiar 0.5, eram 6 falsos positivos, e não 5. O código estava certo; os testes não.
- A validação rejeitava `target_fpr=0`, que é um caso válido (limiar acima de todos os legítimos). Corrigido para o intervalo [0, 1].
- Nos commits, há erros de digitação nas mensagens: "Adicina", "duplicatras", "load-data" e "threshold" no corpo. Ainda não foram corrigidos.

## Próximos passos

- **Passo 5:** experimento completo com as 8 combinações, relatórios em `reports/` e modelo salvo em `models/`.
- **Passo 6:** notebook de EDA e resultados no README.
