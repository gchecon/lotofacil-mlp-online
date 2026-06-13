"""Loop de treinamento prequencial (online) para a Lotofácil.

Uso:
    python -m src.train_online --data data/Lotofácil.xlsx --output outputs/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_loader import load_sorteios
from src.features import build_features, initial_atraso, update_atraso
from src.model import build_model, predict_top15


def _multi_hot(sorteio: np.ndarray, device: torch.device) -> torch.Tensor:
    y = torch.zeros(25, device=device)
    for n in sorteio:
        y[int(n) - 1] = 1.0
    return y


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_online(
    data_path: str | Path,
    output_dir: str | Path,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
    log_every: int = 500,
) -> None:
    """Executa o loop prequencial e salva resultados em output_dir.

    Para cada sorteio t (1..N):
      1. Constrói features com o estado de atraso atual.
      2. Gera predição logits_t (sem ver sorteio_{t+1}).
      3. Se t < N: calcula loss contra sorteio_{t+1}, atualiza pesos e registra acertos.
      4. Atualiza atraso com sorteio_t.

    A predição para o sorteio N+1 (sugestão final) é feita após o último update.

    Args:
        data_path: Caminho para CSV ou XLSX com o histórico.
        output_dir: Diretório onde serão salvos os três artefatos de saída.
        lr: Learning rate do Adam.
        seed: Seed para reprodutibilidade.
        device: "auto", "cpu" ou "cuda".
        log_every: Intervalo de concursos para exibir progresso no terminal.
    """
    dev = _select_device(device)
    print(f"Device: {dev}")
    if dev.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(dev)}")

    sorteios = load_sorteios(data_path)
    N = len(sorteios)
    print(f"Sorteios carregados: {N}")

    model = build_model(seed=seed).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atraso = initial_atraso()
    historico: list[dict] = []
    t0 = time.perf_counter()

    model.train()
    for t in range(N):
        x = torch.tensor(build_features(sorteios[t], atraso), device=dev)
        logits = model(x)

        if t < N - 1:
            y = _multi_hot(sorteios[t + 1], dev)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                top15 = set(predict_top15(model, x))
                acertos = len(top15 & set(sorteios[t + 1].tolist()))

            historico.append({"concurso": t + 1, "acertos": acertos, "loss": loss.item()})

            if (t + 1) % log_every == 0:
                janela = [h["acertos"] for h in historico[-log_every:]]
                media = np.mean(janela)
                elapsed = time.perf_counter() - t0
                print(
                    f"  Concurso {t + 1:>5}/{N} | "
                    f"acertos (últimos {log_every}): {media:.3f} | "
                    f"loss: {loss.item():.4f} | "
                    f"tempo: {elapsed:.1f}s"
                )

        atraso = update_atraso(atraso, sorteios[t])

    # Sugestão para o próximo sorteio (N+1)
    model.eval()
    x_final = torch.tensor(build_features(sorteios[-1], atraso), device=dev)
    sugestao = predict_top15(model, x_final)

    # --- Relatório final ---
    acertos_arr = np.array([h["acertos"] for h in historico])
    media_total = acertos_arr.mean()
    std_total = acertos_arr.std()
    baseline_mean, baseline_std = 9.0, 1.22
    sigma = (media_total - baseline_mean) / baseline_std

    print()
    print("=" * 55)
    print(f"Acertos — média: {media_total:.3f} ± {std_total:.3f}")
    print(f"Baseline acaso:  {baseline_mean:.3f} ± {baseline_std:.3f}")
    print(f"Diferença:       {media_total - baseline_mean:+.3f} ({sigma:+.2f}σ)")
    print(f"Sugestão sorteio {N + 1}: {sugestao}")
    print("=" * 55)

    # --- Persistência ---
    pd.DataFrame(historico).to_csv(output_dir / "historico_acertos.csv", index=False)

    with open(output_dir / "sugestao_proximo_sorteio.json", "w") as f:
        json.dump(
            {
                "proximo_concurso": N + 1,
                "numeros_sugeridos": sugestao,
                "media_acertos_treino": round(float(media_total), 4),
                "baseline_esperado": baseline_mean,
                "diferenca_sigma": round(float(sigma), 4),
            },
            f,
            indent=2,
        )

    torch.save(model.state_dict(), output_dir / "modelo_final.pt")
    print(f"\nArtefatos salvos em: {output_dir.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinamento online prequencial — Lotofácil")
    parser.add_argument("--data", required=True, help="Caminho para CSV ou XLSX com sorteios")
    parser.add_argument("--output", default="outputs/", help="Diretório de saída (padrão: outputs/)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate Adam (padrão: 1e-3)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device de execução (padrão: auto — usa CUDA se disponível)",
    )
    parser.add_argument("--log-every", type=int, default=500, help="Intervalo de log (padrão: 500)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_online(
        data_path=args.data,
        output_dir=args.output,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )