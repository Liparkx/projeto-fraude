# Progresso do projeto

Registro do que foi feito em cada etapa, com os conceitos e erros encontrados. Serve como diário de aprendizado.

## Estado atual

- Repositório: https://github.com/Liparkx/projeto-fraude (público)
- Branch `main`: inclui os Passos 0 a 6, com os merges dos PRs #1 a #4 (`eed3324`).
- Branch `feat/threshold-custo-de-negocio`: B2 (custos de negócio e threshold fixado). Alterações ainda não commitadas.
- Testes: 132 passando; `ruff check` e `ruff format --check` limpos.
- Execução real do experimento: **feita** (49 s, reprodutível em duas execuções); resultados nas seções Passo 6 e B2.
- Próxima etapa: EDA (B3), em branch própria.

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

### Passo 5: experimento ponta a ponta (código e testes)
- `metrics.py`: `pr_auc_bootstrap_ci()` estima um intervalo de confiança da PR-AUC por bootstrap. Serve para mostrar o ruído de uma validação com poucas fraudes (~94).
- `experiment.py`: treina as 8 combinações só no treino e avalia cada uma na validação. Escolhe a de maior PR-AUC na validação e sinaliza se o intervalo dela se sobrepõe ao da segunda colocada. Usa o conjunto de teste **uma única vez**, só na combinação escolhida, com os limiares definidos na validação. Gera tabelas CSV, figuras e `models/best_model.joblib`. Executa com `python -m fraud_detection.experiment`.
- `reporting.py`: três figuras em PNG: curvas Precisão-Recall (um painel por modelo), custo × limiar (um painel por razão de custo) e sensibilidade do custo.
- Decisões: cada painel tem no máximo 4 séries, com cor fixa por estratégia, para ficar dentro da paleta de referência do skill `dataviz`.
- Testes: 17 novos (6 do bootstrap, 11 do experimento e das figuras), com dados sintéticos. Um deles confere que o teste reaproveita os limiares da validação, sem reotimizá-los.
- Status: a execução no dataset real foi feita no Passo 6 (ver abaixo).

Conceitos novos:
- **Intervalo bootstrap:** reamostrar as linhas com reposição e recalcular a métrica muitas vezes. A faixa de 95% mostra quanto o resultado oscila com os dados disponíveis.
- **Uso único do teste:** o conjunto de teste só é lido depois que a combinação e os limiares estão fixados pela validação. Consultá-lo antes disso contamina a escolha.

Ressalvas:
- O validador de paleta `validate_palette.js` não rodou porque o Node.js não está instalado. Confiei na validação documentada na paleta de referência.
- As cores aqua e amarela ficam abaixo de 3:1 de contraste sobre o fundo claro. Por isso cada figura tem a tabela CSV correspondente.
- A comparação entre combinações usa só a validação, com cerca de 94 fraudes. Diferenças pequenas podem ser ruído.

### Passo 6: execução real do experimento
- **Código adicionado:** `recall_at_fpr_bootstrap_ci()` dá o IC 95% do recall a FPR fixa, recalculando o limiar em cada reamostra. O experimento agora mostra o IC do recall a 0,1% e a 1%, além de dois rankings: por PR-AUC e pelo recall a 0,1%, com sinal quando discordam. Foi adicionada a figura `recall_em_fpr_fixa_validacao.png`.
- **Execução:** 48 segundos. Rodada duas vezes, com as 4 tabelas idênticas. Gerou 4 CSVs, 4 PNGs e `models/best_model.joblib`.
- **Validação (8 combinações):**

| Combinação | PR-AUC [IC 95%] | Recall a 0,1% [IC] | Recall a 1% [IC] |
|---|---|---|---|
| xgboost/baseline | **0,882** [0,820; 0,928] | 0,872 [0,802; 0,921] | 0,904 [0,848; 0,953] |
| xgboost/class_weight | 0,880 [0,819; 0,927] | 0,872 [0,804; 0,925] | 0,904 [0,848; 0,953] |
| xgboost/smote | 0,879 [0,817; 0,925] | 0,872 [0,798; 0,925] | **0,915** [0,865; 0,959] |
| logistic/baseline | 0,803 [0,733; 0,864] | **0,894** [0,823; 0,941] | 0,894 [0,833; 0,942] |
| xgboost/undersampling | 0,552 [0,449; 0,657] | 0,596 [0,466; 0,719] | 0,883 [0,817; 0,936] |
| logistic/undersampling | 0,490 [0,380; 0,596] | 0,521 [0,433; 0,667] | 0,798 [0,720; 0,867] |

- **Leitura:** o undersampling fica claramente atrás. As três combinações XGBoost com baseline, class weights e SMOTE estão dentro do IC uma da outra, então o empate é real. Pelo recall a 0,1%, a líder é `logistic/baseline`, que discorda da líder por PR-AUC.
- **Teste (só `xgboost/baseline`, com os limiares da validação):** PR-AUC 0,827 (queda de 0,882 da validação, perto do limite inferior do IC); ROC-AUC 0,970; recall 0,842 a 0,1% com FPR realizada de 0,078%; recall 0,874 a 1% com FPR realizada de 1,059%.
- **F1 a 0,5 engana:** as estratégias com reamostragem têm F1 entre 0,05 e 0,10 com o limiar 0,5. Elas deslocam as probabilidades para cima, o que confirma que 0,5 não serve para comparar estratégias.
- **Calibração:** o limiar ótimo empírico fica bem abaixo do limiar de Bayes teórico. Isso sugere que as probabilidades do XGBoost não estão bem calibradas.

Decisão do usuário (revogada depois): tratar `xgboost/class_weight` como vencedor pela estabilidade. O usuário decidiu não declarar vencedor no projeto; ele indicará o melhor modelo no README ao final.

Conceitos novos:
- **Estabilidade da FPR fixa:** o limiar depende dos k falsos positivos permitidos (k = FPR × legítimos). Com 0,1% são ~57 falsos positivos (ruído de ~13% no limiar); com 1% são ~567 (ruído de ~4%).
- **IC do recall:** o recall tem erro padrão de 4 a 5 pontos com ~94 fraudes, independentemente da FPR. O IC mostra essa incerteza.

Ressalvas:
- O validador de paleta não rodou (Node.js ausente).
- A comparação usa só a validação, com cerca de 94 fraudes. Diferenças pequenas são ruído.
- As figuras tiveram três problemas visuais corrigidos: a legenda do recall cobria a cor do SMOTE; a legenda das curvas PR estava sobre as curvas; e os rótulos do gráfico de custo colidiam.

### B2: custos de negócio e threshold fixado
- **Premissas de custo** (editáveis em `ExperimentConfig`; a moeda do dataset ULB não é informada, então a unidade é a "unidade monetária do dataset", provavelmente euro):
  - C_FN = valor da transação + taxa fixa de fraude (25).
  - C_FP = custo fixo de bloquear um legítimo (5).
- **Código:** `threshold.py` passa a aceitar custos por linha (`transaction_cost_arrays`); `cost_curve` soma os custos com acumulados ponderados e devolve `fn_cost` e `fp_cost`; novas `near_optimal_range` (faixa com custo até 2% acima do mínimo) e `cost_scenario_sensitivity` (taxa fixa × C_FP). `experiment.py` calcula o custo em R$ das 8 combinações na validação, seleciona a de menor custo esperado, fixa o limiar ótimo e o aplica uma única vez ao teste.
- **Validação, custo ótimo por combinação (R$):**

| Combinação | Limiar ótimo | Faixa (≤ 2%) | Custo na validação | FN | FP |
|---|---|---|---|---|---|
| xgboost/baseline | 0,0032 | [0,0031; 0,0032] | **2.010** | 9 | 83 |
| logistic/baseline | 0,0264 | [0,0202; 0,0294] | 2.177 | 10 | 49 |
| logistic/class_weight | 0,9996 | [0,9982; 0,9996] | 2.241 | 13 | 26 |
| logistic/smote | 0,9994 | [0,9985; 0,9994] | 2.251 | 13 | 28 |
| xgboost/class_weight | 0,1612 | [0,0603; 0,7520] | 2.292 | 12 | 14 |
| xgboost/smote | 0,6637 | [0,5427; 0,6637] | 2.313 | 13 | 13 |
| xgboost/undersampling | 0,9941 | [0,9937; 0,9942] | 3.817 | 16 | 273 |
| logistic/undersampling | 0,9962 | [0,9952; 0,9962] | 4.507 | 24 | 134 |

- **Limiar fixado: 0,0032** (xgboost/baseline, menor custo esperado na validação; faixa [0,0031; 0,0032]).
- **Teste, uma única vez, com o limiar 0,0032:**
  - Custo: R$ 4.071,76. Com limiar 0,5: R$ 5.524,73. Sem modelo: R$ 17.141,31. Economia de R$ 13.069,55 (76% do custo sem modelo).
  - FN 14, FP 80, TP 81; recall 0,853, precisão 0,503, FPR 0,141%.
- **Sensibilidade aos custos** (limiar ótimo na validação, por taxa fixa de fraude e C_FP):
  - Taxa 0 ou 25, C_FP 1 ou 5: limiar 0,0032.
  - Taxa 25, C_FP 20: limiar 0,0413. Taxa 0, C_FP 20 ou 50: limiar 0,2979.
  - Taxa 100, C_FP 20: limiar 0,0413. Faixa e limiar mudam bastante com os custos, então o limiar depende das premissas.
- **Conceitos:**
  - Com custo por valor, o limiar depende da distribuição dos valores das fraudes. Fraudes pequenas pesam pouco, e o modelo tolera deixá-las passar (mediana de Amount nas fraudes: 9,82).
  - A curva de custo é plana acima do mínimo. Por isso a faixa quase-ótima é a medida honesta da incerteza.
  - O limiar 0,0032 bloqueia cerca de 0,14% dos legítimos. Muito abaixo de 0,5, como esperado para probabilidades não calibradas.
- **Ressalvas:**
  - Os custos (25 e 5) são premissas, não dados. O threshold depende deles, por isso a sensibilidade por cenário.
  - **O custo no teste (R$ 4.072) é o dobro do custo na validação (R$ 2.010).** Os FN sobem de 9 para 14, e o custo é dominado por poucas fraudes de valor alto. Com ~94 fraudes na validação, a estimativa de custo tem variância alta.
  - A faixa quase-ótima é estreita para xgboost/baseline (0,0031 a 0,0032). Isso não significa precisão; significa que a curva sobe rápido de um lado do mínimo.
  - As 8 combinações têm calibrações diferentes, então os limiares não são comparáveis entre si. Só os custos são.

### Status dos entregáveis
1. **Tratamento do desbalanceamento:** feito (4 estratégias, 8 combinações, dataset real).
2. **Análise exploratória:** pendente (B3; `notebooks/` só tem `.gitkeep`).
3. **PR-AUC e recall a FPR fixa:** feito (8 combinações, IC bootstrap, teste usado uma vez).
4. **Threshold por custo:** feito, com custos premissas (B2). O limiar fixado na validação é 0,0032 e foi avaliado no teste uma única vez.

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

### PR #2: `feat/threshold-por-custo` (mergeado)

Commits:
1. `b67e850`: adiciona `threshold.py` e `test_threshold.py`.
2. `8d0427c`: atualiza o `PROGRESS.md`. O título do PR e a mensagem trazem o erro de digitação "threshould".

Merge: `f8c9d53`.

Branches locais já mergeadas podem ser apagadas: `feat/metricas-e-remocao-de-duplicatas` e `feat/threshold-por-custo`.

### PR #3: `feat/experimento-completo` (mergeado)

Commits:
1. `5acd636`: adiciona o intervalo bootstrap para a PR-AUC.
2. `191fbce`: adiciona o experimento ponta a ponta e as figuras.
3. `16f4c13`: atualiza o `PROGRESS.md` com o Passo 5.

Merge: `33883df`.

### PR #4: `feat/execucao-do-experimento` (mergeado)

Commits:
1. `fc9fa83`: adiciona o intervalo bootstrap para o recall a FPR fixa.
2. `695ecf5`: adiciona rankings, recall a FPR fixa e nova figura ao experimento.
3. `e755265`: documenta a execução real do Passo 6.

Merge: `eed3324`.

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
- Nos commits, há erros de digitação nas mensagens: "Adicina", "duplicatras", "load-data" e "threshould" no título do PR #2 e no commit `8d0427c`. Ainda não foram corrigidos; corrigir exige reescrever o histórico.

## Próximos passos

- **B2 — Threshold por custo:** com custos de negócio definidos, fixar o threshold do modelo final na validação (mínimo exato, com faixa quase-ótima reportada), avaliar no teste só com esse limiar e salvar o threshold e o cenário de custos no modelo.
- **B3 — EDA:** notebook em `notebooks/` (distribuições de `Amount` e `Time`, fraude × legítimo, nulos, correlações), sem lógica de produção.
