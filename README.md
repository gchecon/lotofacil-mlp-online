# Lotofácil — MLP Online Prequencial

Experimento exploratório: rede neural (MLP) treinada de forma **incremental, sorteio a sorteio**, sobre o histórico de sorteios da Lotofácil (15 números de 1–25), com entrada enriquecida por features heurísticas clássicas.

> **Premissa:** se os sorteios forem i.i.d., não existe dependência temporal a ser aprendida. O projeto serve para testar essa hipótese e demonstrar a diferença entre padrão real e padrão percebido.

---

## Instalação

Requer Python 3.12+ e [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

---

## Dados

Coloque o histórico de sorteios em `data/`. São aceitos:

- **XLSX/XLS** no formato oficial da CEF (com colunas `Bola1`–`Bola15`)
- **CSV** simples sem cabeçalho (15 colunas de inteiros por linha)

Os sorteios devem estar em **ordem cronológica**. O arquivo de dados é ignorado pelo git.

---

## Uso

### Treinamento

```bash
uv run python -m src.train_online --data data/Lotofácil.xlsx --output outputs/
```

Opções disponíveis:

| Flag | Padrão | Descrição |
|---|---|---|
| `--data` | — | Caminho para o arquivo de sorteios (obrigatório) |
| `--output` | `outputs/` | Diretório de saída |
| `--lr` | `1e-3` | Learning rate do Adam |
| `--seed` | `42` | Seed para reprodutibilidade |
| `--device` | `auto` | `auto`, `cpu` ou `cuda` |
| `--log-every` | `500` | Intervalo de log no terminal |

Artefatos gerados em `outputs/`:

```
outputs/
├── historico_acertos.csv       # acertos e loss por concurso
├── sugestao_proximo_sorteio.json
└── modelo_final.pt
```

### Comparação com baseline

```bash
uv run python -m src.baseline
```

### Análise dos pesos pós-treino

```bash
uv run python -m src.analyze_weights
```

Gera `outputs/analise_pesos.png` com heatmap, magnitude por grupo de features e distribuição de pesos.

---

## Features de entrada (54 dims)

Para o sorteio $D_t \subseteq \{1,\ldots,25\}$, $|D_t|=15$:

| # | Feature | Dims | Normalização |
|---|---------|------|--------------|
| 1 | Multi-hot do sorteio | 25 | — (já em [0,1]) |
| 2 | Atraso por número | 25 | `atraso / (atraso + 1)` ∈ [0,1) |
| 3 | Soma total | 1 | `(soma − 195) / 18.0` |
| 4 | Paridade (qtd de pares) | 1 | `(pares − 7.2) / 1.25` |
| 5 | Primos | 1 | `(primos − 5.4) / 1.2` |
| 6 | Consecutivos | 1 | `(consec − 8.4) / 1.6` |

Constantes de normalização das features 3–5 derivadas da distribuição hipergeométrica teórica (N=25, K=15, n=15) — independentes do dataset.

O **atraso** de um número é quantos sorteios consecutivos ele está ausente. Atualizado a cada sorteio:

```
atraso[i] = 0        se (i+1) ∈ D_t
atraso[i] += 1       caso contrário
```

---

## Arquitetura e treinamento

```
MLP: 54 → 64 → 32 → 25   (ReLU nas camadas ocultas, logits na saída)
Loss: BCEWithLogitsLoss
Otimizador: Adam (lr=1e-3)
```

O loop segue o **protocolo prequencial** (Dawid): a predição para o sorteio $t+1$ é feita *antes* de observá-lo, eliminando vazamento de dados.

```
atraso = zeros(25)
para t = 1..N:
    x_t  = features(sorteio_t, atraso)
    pred = modelo(x_t)                    # predição para t+1

    se t < N:
        loss = BCE(pred, multi_hot(sorteio_{t+1}))
        loss.backward(); step()
        registrar acertos

    atraso = atualizar_atraso(atraso, sorteio_t)

sugestão = top15(sigmoid(modelo(x_N)))    # para o sorteio N+1
```

---

## Baseline e avaliação

A baseline representa qualquer conjunto de 15 números escolhido ao acaso:

$$X \sim \text{Hipergeométrica}(N=25,\, K=15,\, n=15)$$

$$E[X] = 9{,}0 \qquad \text{SD}[X] \approx 1{,}22$$

Todo resultado é reportado em relação a essa baseline. **9/15 já é o acaso** — nunca interpretar isoladamente como sucesso.

### Resultado sobre 3.709 sorteios

| Métrica | Valor |
|---|---|
| Média modelo | 9,117 ± 1,248 |
| Baseline (acaso) | 9,000 ± 1,220 |
| Diferença | +0,10σ — efeito negligenciável |

A análise dos pesos da primeira camada confirma a hipótese i.i.d.: todos os grupos de features têm magnitude próxima da inicialização aleatória, sem estrutura diferenciada para as features temporais (multi-hot e atraso).

---

## Estrutura do projeto

```
.
├── CLAUDE.md                   # especificação técnica do projeto
├── pyproject.toml
├── uv.lock
├── data/
│   └── Lotofácil.xlsx          # ignorado pelo git
├── src/
│   ├── data_loader.py          # leitura e validação do histórico
│   ├── features.py             # 6 features + estado de atraso
│   ├── model.py                # MLP configurável
│   ├── train_online.py         # loop prequencial
│   ├── baseline.py             # baseline hipergeométrica + comparação
│   └── analyze_weights.py      # análise dos pesos pós-treino
└── outputs/                    # ignorado pelo git
    ├── historico_acertos.csv
    ├── sugestao_proximo_sorteio.json
    ├── modelo_final.pt
    └── analise_pesos.png
```

---

## Stack

- Python 3.12+, PyTorch ≥ 2.2, pandas, numpy, matplotlib, openpyxl
- CPU é suficiente; `--device auto` usa CUDA se disponível