"""Baseline hipergeométrica e comparação estatística com resultados do modelo.

A baseline representa qualquer conjunto de 15 números escolhido ao acaso de {1..25}:
    X ~ Hipergeométrica(N=25, K=15, n=15)
    E[X] = 9.0
    Var[X] = 1.5  →  SD ≈ 1.2247

Uso:
    python -m src.baseline --historico outputs/historico_acertos.csv
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Hipergeométrica(N=25, K=15, n=15): escolher 15 de 25 onde 15 são "acerto"
_N, _K, _n = 25, 15, 15

BASELINE_MEAN: float = _n * _K / _N                                         # 9.0
BASELINE_VAR: float = _n * (_K / _N) * (1 - _K / _N) * (_N - _n) / (_N - 1)  # 1.5
BASELINE_STD: float = math.sqrt(BASELINE_VAR)                               # ≈ 1.2247


def baseline_stats() -> tuple[float, float]:
    """Retorna (média, desvio padrão) teóricos da baseline hipergeométrica."""
    return BASELINE_MEAN, BASELINE_STD


def _pmf(k: int) -> float:
    """P(X = k) para X ~ Hipergeométrica(N=25, K=15, n=15)."""
    from math import comb
    return comb(_K, k) * comb(_N - _K, _n - k) / comb(_N, _n)


def _normal_pvalue(z: float) -> float:
    """p-value bicaudal aproximado via função de erro (sem scipy)."""
    return math.erfc(abs(z) / math.sqrt(2))


def compare(historico_path: str | Path) -> dict:
    """Compara os acertos do modelo com a baseline hipergeométrica.

    Args:
        historico_path: Caminho para historico_acertos.csv gerado pelo train_online.

    Returns:
        Dicionário com estatísticas do modelo, baseline e teste z.
    """
    df = pd.read_csv(historico_path)
    acertos = df["acertos"].to_numpy(dtype=float)
    n = len(acertos)

    model_mean = float(acertos.mean())
    model_std = float(acertos.std())

    # Teste z: diferença de médias (válido para n >> 1; aqui n ≈ 3700)
    se = BASELINE_STD / math.sqrt(n)
    z = (model_mean - BASELINE_MEAN) / se
    p_value = _normal_pvalue(z)

    counts = np.bincount(acertos.astype(int), minlength=16)

    return {
        "n_sorteios": n,
        "model_mean": model_mean,
        "model_std": model_std,
        "baseline_mean": BASELINE_MEAN,
        "baseline_std": BASELINE_STD,
        "delta": model_mean - BASELINE_MEAN,
        "delta_sigma": (model_mean - BASELINE_MEAN) / BASELINE_STD,
        "z_score": z,
        "p_value_bicaudal": p_value,
        "distribuicao_acertos": {str(k): int(counts[k]) for k in range(16) if counts[k] > 0},
    }


def print_report(stats: dict) -> None:
    """Imprime relatório formatado de comparação."""
    n = stats["n_sorteios"]
    print()
    print("=" * 58)
    print("  COMPARAÇÃO MODELO vs. BASELINE HIPERGEOMÉTRICA")
    print("=" * 58)
    print(f"  Sorteios avaliados : {n}")
    print(f"  Média modelo       : {stats['model_mean']:.4f}  ± {stats['model_std']:.4f}")
    print(f"  Baseline (acaso)   : {stats['baseline_mean']:.4f}  ± {stats['baseline_std']:.4f}")
    print(f"  Diferença          : {stats['delta']:+.4f}  ({stats['delta_sigma']:+.2f}σ)")
    print(f"  Teste z            : z = {stats['z_score']:+.3f},  p = {stats['p_value_bicaudal']:.4f}")
    print()
    print("  Distribuição de acertos (modelo vs. teórica):")
    print(f"  {'k':>3}  {'obs':>6}  {'obs%':>7}  {'teo%':>7}")
    print("  " + "-" * 28)
    for k_str, obs in sorted(stats["distribuicao_acertos"].items(), key=lambda x: int(x[0])):
        k = int(k_str)
        obs_pct = 100 * obs / n
        teo_pct = 100 * _pmf(k)
        print(f"  {k:>3}  {obs:>6}  {obs_pct:>6.2f}%  {teo_pct:>6.2f}%")
    print("=" * 58)
    p = stats["p_value_bicaudal"]
    effect = abs(stats["delta_sigma"])
    # Com n≈3700, o z-teste tem poder para detectar efeitos ínfimos.
    # O tamanho do efeito (delta em σ da baseline) é o indicador primário.
    if effect < 0.2:
        verdict = (
            f"Efeito negligenciável ({stats['delta_sigma']:+.2f}σ da baseline). "
            f"p={p:.4f} reflete n={n} grande, não sinal prático."
        )
    elif effect < 0.5:
        verdict = f"Efeito pequeno ({stats['delta_sigma']:+.2f}σ, p={p:.4f}); interpretar com cautela."
    else:
        verdict = f"Efeito relevante ({stats['delta_sigma']:+.2f}σ, p={p:.4f})."
    print(f"  {verdict}")
    print("=" * 58)
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparação com baseline hipergeométrica")
    parser.add_argument(
        "--historico",
        default="outputs/historico_acertos.csv",
        help="Caminho para historico_acertos.csv (padrão: outputs/historico_acertos.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stats = compare(args.historico)
    print_report(stats)