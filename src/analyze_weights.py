"""Análise dos pesos da primeira camada pós-treinamento.

Teste de sanidade: se os sorteios forem i.i.d., os pesos que processam as
features 1-2 (multi-hot e atraso — hipótese de dependência temporal) devem
permanecer próximos da inicialização aleatória, sem estrutura aprendida.

Grupos de features na primeira camada (W: 64 × 54):
  [0:25]  multi-hot do sorteio      (feature 1 — temporal)
  [25:50] atraso por número         (feature 2 — temporal)
  [50]    soma total                 (feature 3 — escalar)
  [51]    paridade                   (feature 4 — escalar)
  [52]    primos                     (feature 5 — escalar)
  [53]    consecutivos               (feature 6 — escalar)

Uso:
    python -m src.analyze_weights --model outputs/modelo_final.pt --output outputs/
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import LotofacilMLP

# Limites de inicialização kaiming_uniform (padrão do nn.Linear com ReLU)
# bound = sqrt(1 / fan_in)  onde fan_in = 54
_INIT_BOUND = math.sqrt(1 / 54)          # ≈ 0.136
_INIT_MEAN_ABS = _INIT_BOUND / 2         # E[|w|] ~ bound/2 para Uniform(-b, b)
_INIT_STD = _INIT_BOUND / math.sqrt(3)   # std de Uniform(-b, b)

_GROUPS = {
    "multi-hot\n[0:25]": (0, 25),
    "atraso\n[25:50]": (25, 50),
    "soma\n[50]": (50, 51),
    "paridade\n[51]": (51, 52),
    "primos\n[52]": (52, 53),
    "consecutivos\n[53]": (53, 54),
}

_TEMPORAL_COLS = slice(0, 50)   # features 1-2
_SCALAR_COLS = slice(50, 54)    # features 3-6


def load_first_layer_weights(model_path: str | Path) -> np.ndarray:
    """Carrega o modelo e retorna os pesos da primeira camada, shape (64, 54)."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    w = state["net.0.weight"].numpy()   # Linear(54→64) → shape (64, 54)
    return w


def analyze(w: np.ndarray) -> dict:
    """Calcula estatísticas por grupo de features."""
    results = {}
    for label, (start, end) in _GROUPS.items():
        cols = w[:, start:end].ravel()
        results[label] = {
            "mean_abs": float(np.abs(cols).mean()),
            "std": float(cols.std()),
            "max_abs": float(np.abs(cols).max()),
            "n": len(cols),
        }

    temporal = w[:, _TEMPORAL_COLS].ravel()
    scalar = w[:, _SCALAR_COLS].ravel()
    results["_temporal"] = {"mean_abs": float(np.abs(temporal).mean()), "std": float(temporal.std())}
    results["_scalar"] = {"mean_abs": float(np.abs(scalar).mean()), "std": float(scalar.std())}
    return results


def print_report(stats: dict) -> None:
    print()
    print("=" * 62)
    print("  ANÁLISE DOS PESOS DA PRIMEIRA CAMADA  (W: 64 × 54)")
    print("=" * 62)
    print(f"  Inicialização kaiming_uniform: bound ≈ ±{_INIT_BOUND:.4f}")
    print(f"  E[|w|] init ≈ {_INIT_MEAN_ABS:.4f}   std init ≈ {_INIT_STD:.4f}")
    print()
    print(f"  {'Grupo':<20} {'E[|w|]':>8} {'std':>8} {'max|w|':>8}")
    print("  " + "-" * 46)
    for label, s in stats.items():
        if label.startswith("_"):
            continue
        clean = label.replace("\n", " ")
        print(f"  {clean:<20} {s['mean_abs']:>8.4f} {s['std']:>8.4f} {s['max_abs']:>8.4f}")
    print("  " + "-" * 46)
    print(f"  {'[init referência]':<20} {_INIT_MEAN_ABS:>8.4f} {_INIT_STD:>8.4f}")
    print()

    t = stats["_temporal"]
    sc = stats["_scalar"]
    ratio = sc["mean_abs"] / t["mean_abs"] if t["mean_abs"] > 0 else float("inf")
    print(f"  Temporal (feat 1-2): E[|w|] = {t['mean_abs']:.4f}")
    print(f"  Escalar  (feat 3-6): E[|w|] = {sc['mean_abs']:.4f}")
    print(f"  Razão escalar/temporal: {ratio:.2f}×")
    print()
    if ratio < 1.3:
        verdict = "Sem estrutura diferenciada — pesos uniformemente próximos da init."
    elif ratio < 2.0:
        verdict = "Leve preferência pelas features escalares; sinal fraco."
    else:
        verdict = "Pesos escalares significativamente maiores — rede priorizou features marginais."
    print(f"  {verdict}")
    print("=" * 62)
    print()


def plot(w: np.ndarray, stats: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Pesos da primeira camada pós-treinamento (64 × 54)", fontsize=12)

    # --- Painel 1: heatmap W ---
    ax = axes[0]
    im = ax.imshow(w, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_title("Heatmap W (linhas=neurônios, colunas=features)")
    ax.set_xlabel("Feature (coluna)")
    ax.set_ylabel("Neurônio oculto")
    ax.axvline(24.5, color="k", lw=0.8, ls="--", label="multi-hot|atraso")
    ax.axvline(49.5, color="gray", lw=0.8, ls="--", label="atraso|escalares")
    ax.legend(fontsize=7, loc="upper right")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # --- Painel 2: E[|w|] por grupo vs. init ---
    ax = axes[1]
    labels = [l.replace("\n", " ") for l in _GROUPS]
    mean_abs = [stats[l]["mean_abs"] for l in _GROUPS]
    colors = ["steelblue"] * 2 + ["darkorange"] * 4
    bars = ax.bar(range(len(labels)), mean_abs, color=colors)
    ax.axhline(_INIT_MEAN_ABS, color="red", ls="--", lw=1.2, label=f"init E[|w|]≈{_INIT_MEAN_ABS:.3f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("E[|w|]")
    ax.set_title("Magnitude média por grupo de features")
    ax.legend(fontsize=8)
    # legenda de cores
    from matplotlib.patches import Patch
    ax.legend(handles=[
        bars[0], bars[2],
        plt.Line2D([0], [0], color="red", ls="--", lw=1.2),
    ], labels=["temporal (feat 1-2)", "escalar (feat 3-6)", f"init ≈{_INIT_MEAN_ABS:.3f}"], fontsize=8)

    # --- Painel 3: distribuição de pesos temporal vs. escalar ---
    ax = axes[2]
    temporal_w = w[:, _TEMPORAL_COLS].ravel()
    scalar_w = w[:, _SCALAR_COLS].ravel()
    ax.hist(temporal_w, bins=40, alpha=0.6, color="steelblue", label="temporal (feat 1-2)", density=True)
    ax.hist(scalar_w, bins=20, alpha=0.6, color="darkorange", label="escalar (feat 3-6)", density=True)
    x = np.linspace(-_INIT_BOUND, _INIT_BOUND, 100)
    ax.plot(x, np.ones_like(x) / (2 * _INIT_BOUND), "r--", lw=1.2, label="Uniform init")
    ax.set_xlabel("Valor do peso")
    ax.set_ylabel("Densidade")
    ax.set_title("Distribuição: temporal vs. escalar")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "analise_pesos.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo em: {out_path.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análise dos pesos da primeira camada")
    parser.add_argument("--model", default="outputs/modelo_final.pt", help="Caminho para modelo_final.pt")
    parser.add_argument("--output", default="outputs/", help="Diretório para salvar o gráfico")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    w = load_first_layer_weights(args.model)
    stats = analyze(w)
    print_report(stats)
    plot(w, stats, Path(args.output))