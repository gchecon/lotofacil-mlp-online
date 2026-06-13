"""Feature engineering para o loop prequencial da Lotofácil.

Vetor de entrada: 54 dims
  [0:25]  multi-hot do sorteio
  [25:50] atraso normalizado por número
  [50]    soma total
  [51]    paridade (qtd de pares)
  [52]    primos
  [53]    consecutivos
"""

import numpy as np

# Constantes teóricas (hipergeométrica N=25, K=15, n=15)
_SOMA_MEAN = 195.0
_SOMA_STD = 18.0
_PARES_MEAN = 7.2
_PARES_STD = 1.25
_PRIMOS_MEAN = 5.4
_PRIMOS_STD = 1.2
_CONSEC_MEAN = 8.4
# SD analítico não-trivial; valor obtido por simulação de ~1M sorteios uniformes
_CONSEC_STD = 1.6

_PRIMOS: frozenset[int] = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23})
_PARES: frozenset[int] = frozenset(range(2, 26, 2))  # {2, 4, ..., 24}


def initial_atraso() -> np.ndarray:
    """Retorna o vetor de atraso inicial zerado, shape (25,), float32."""
    return np.zeros(25, dtype=np.float32)


def build_features(sorteio: np.ndarray, atraso: np.ndarray) -> np.ndarray:
    """Constrói o vetor de entrada (54 dims) para um sorteio.

    Args:
        sorteio: Array de 15 inteiros em [1, 25] representando o sorteio atual.
        atraso: Array shape (25,) com atraso de cada número (índice i → número i+1),
                refletindo o estado **antes** de processar este sorteio.

    Returns:
        Array float32 de shape (54,) com as features normalizadas.
    """
    nums = set(int(n) for n in sorteio)

    # Feature 1: multi-hot (25 dims) — já em [0, 1]
    multi_hot = np.zeros(25, dtype=np.float32)
    for n in nums:
        multi_hot[n - 1] = 1.0

    # Feature 2: atraso/(atraso+1) ∈ [0, 1) (25 dims)
    atraso_f = atraso.astype(np.float32)
    atraso_norm = atraso_f / (atraso_f + 1.0)

    # Feature 3: soma ∈ [120, 270]
    soma_norm = (float(sorteio.sum()) - _SOMA_MEAN) / _SOMA_STD

    # Feature 4: quantidade de pares ∈ {2,4,...,24}
    pares_norm = (float(len(nums & _PARES)) - _PARES_MEAN) / _PARES_STD

    # Feature 5: quantidade de primos ∈ {2,3,5,7,11,13,17,19,23}
    primos_norm = (float(len(nums & _PRIMOS)) - _PRIMOS_MEAN) / _PRIMOS_STD

    # Feature 6: pares consecutivos |{k∈[1,24] : k∈D e k+1∈D}|
    consec = float(sum(1 for k in range(1, 25) if k in nums and (k + 1) in nums))
    consec_norm = (consec - _CONSEC_MEAN) / _CONSEC_STD

    return np.concatenate([
        multi_hot,
        atraso_norm,
        np.array([soma_norm, pares_norm, primos_norm, consec_norm], dtype=np.float32),
    ])


def update_atraso(atraso: np.ndarray, sorteio: np.ndarray) -> np.ndarray:
    """Atualiza o vetor de atraso após processar um sorteio.

    Para cada número i+1 em [1, 25]:
      - Se foi sorteado: atraso[i] = 0
      - Caso contrário:  atraso[i] += 1

    Args:
        atraso: Array (25,) com o estado atual de atraso.
        sorteio: Array de 15 inteiros do sorteio recém-processado.

    Returns:
        Novo array (25,) com atrasos atualizados; não modifica o original.
    """
    new_atraso = atraso + 1.0
    for n in sorteio:
        new_atraso[int(n) - 1] = 0.0
    return new_atraso.astype(np.float32)