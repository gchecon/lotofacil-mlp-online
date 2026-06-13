# Lotofácil — Heurísticas + MLP Online (projeto exploratório)

**Repositório:** https://github.com/gchecon/lotofacil-mlp-online

## Visão geral

Rede neural (MLP) treinada de forma **online** (incremental, sorteio a sorteio) sobre o histórico de sorteios da Lotofácil (15 números de 1-25), com entrada enriquecida por features heurísticas clássicas de apostador (atraso, soma, paridade, primos, sequências consecutivas).

**Premissa de fundo:** se os sorteios forem i.i.d., não existe dependência t→t+1 a ser aprendida. O experimento serve para (a) testar se as features heurísticas dão algum "gancho" além das frequências marginais, e (b) servir de demonstração concreta da diferença entre padrão real e padrão percebido.

**Tom do projeto:** exploratório/analítico. Evitar em código, comentários, docstrings e logs qualquer linguagem que sugira "previsão real" ou "sistema de apostas". Resultados sempre relativos à baseline de acaso (seção "Avaliação e baseline").

## Dados de entrada

- Planilha (CSV ou XLSX), uma linha por sorteio, **em ordem cronológica** (concurso 1, 2, 3, ...).
- Cada linha: 15 inteiros distintos em [1,25].
- **Nunca embaralhar os dados** — a ordem é parte do protocolo de avaliação.
- Validar no carregamento: 15 valores únicos por linha, todos em [1,25].

## Feature engineering — vetor de entrada (54 dims)

Para o sorteio t, D_t ⊆ {1,...,25}, |D_t| = 15.

| # | Feature | Dims | Definição | Normalização |
|---|---------|------|-----------|--------------|
| 1 | Multi-hot do sorteio | 25 | posição i = 1 se (i+1) ∈ D_t | — (já em [0,1]) |
| 2 | Atraso por número (**estado**) | 25 | atraso_t[i] = 0 se (i+1) ∈ D_t, senão atraso_{t-1}[i]+1. Init: atraso_0[i]=0 | log1p(atraso) ou atraso/(atraso+1) |
| 3 | Soma total | 1 | Σ D_t ∈ [120,270] | (soma − 195) / 18.0 |
| 4 | Paridade | 1 | \|D_t ∩ {pares 2..24}\| | (pares − 7.2) / 1.25 |
| 5 | Primos | 1 | \|D_t ∩ {2,3,5,7,11,13,17,19,23}\| | (primos − 5.4) / 1.2 |
| 6 | Consecutivos | 1 | \|{k∈[1,24] : k∈D_t e k+1∈D_t}\| | E≈8.4; SD empírico (não-trivial analiticamente) |

Constantes de normalização das features 3-5 vêm da distribuição hipergeométrica (N=25, n=15) — **teóricas, não dependem do dataset**.

**Alvo:** multi-hot (25 dims) do sorteio t+1.

## Arquitetura

- MLP: `54 → 64 → 32 → 25`, ReLU nas camadas ocultas, saída em **logits** (sem ativação).
- Loss: `nn.BCEWithLogitsLoss`.
- Sugestão final: `topk(sigmoid(logits), k=15)`.
- Otimizador: Adam, lr ~1e-3 (hiperparâmetro a explorar).
- Framework: **PyTorch** (controle fino sobre o loop online).

## Loop de treinamento (avaliação prequencial)

```
atraso = zeros(25)
para t de 1..N:
    X_t = features(sorteio_t, atraso)            # 54 dims
    logits_t = modelo(X_t)
    pred_t = sigmoid(logits_t)                    # previsão p/ t+1

    se t < N:
        Y = multi_hot(sorteio_{t+1})
        loss = BCEWithLogitsLoss(logits_t, Y)
        loss.backward(); otimizador.step(); otimizador.zero_grad()
        registrar(acertos = |topk(pred_t,15) ∩ sorteio_{t+1}|)

    atraso = atualizar_atraso(atraso, sorteio_t)

sugestao_final = topk(sigmoid(modelo(X_N)), 15)   # para o sorteio N+1
```

Regra de ouro: `pred_t` é calculada **antes** de usar `sorteio_{t+1}` (princípio prequencial, Dawid) — sem isso há vazamento de dados.

## Avaliação e baseline — CRÍTICO

Baseline de acaso (hipergeométrica N=25, K=15, n=15), válida para QUALQUER conjunto de 15 números, mesmo não-informativo:

```
E[acertos] = 15 × (15/25) = 9
SD[acertos] ≈ 1.22
```

**Regra obrigatória:** todo resultado de `acertos` deve ser reportado relativo a essa baseline (ex.: "média 9.1±1.3 sobre N sorteios; baseline 9.0±1.22 → 0.1σ, sem sinal"). 9/15 já É o acaso — nunca reportar isolado como sucesso.

Teste de sanidade (fase final): verificar se os pesos de entrada que processam as features 1-2 (hipótese de dependência temporal) ficam próximos da inicialização — esperado sob i.i.d.

## Execução (alvo)

```bash
python -m src.train_online --data data/sorteios.csv --output outputs/
```

## Estrutura de diretórios

```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   └── sorteios.csv
├── src/
│   ├── data_loader.py    # leitura + validação
│   ├── features.py       # 6 features + normalizações + estado de atraso
│   ├── model.py           # MLP (54→64→32→25)
│   ├── train_online.py    # loop prequencial
│   └── baseline.py         # baseline hipergeométrica + comparação
└── outputs/
    ├── historico_acertos.csv
    ├── sugestao_proximo_sorteio.json
    └── modelo_final.pt
```

## Stack

- Python 3.11+, torch, pandas, numpy, openpyxl (se XLSX)
- CPU é suficiente — sem dependência de GPU/CUDA.

## Convenções

- Type hints em funções públicas; docstrings (Google/NumPy) em funções com estado (`atualizar_atraso`, loop de treino).
- `torch.manual_seed()` fixo para reprodutibilidade.
- `atraso` mantém nome em português (termo de domínio sem equivalente conciso); demais identificadores em inglês.

## Roadmap

- [x] `data_loader.py` — leitura + validação (15 únicos, 1-25)
- [x] `features.py` — 6 features + normalizações + estado de atraso
- [x] `model.py` — MLP configurável (dims como parâmetro)
- [x] `train_online.py` — loop prequencial + registro de acertos
- [x] `baseline.py` — baseline hipergeométrica + comparação automática
- [x] Análise final: distribuição dos pesos de entrada pós-treino
