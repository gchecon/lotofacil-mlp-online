"""MLP para o problema de previsão prequencial da Lotofácil."""

import torch
import torch.nn as nn


class LotofacilMLP(nn.Module):
    """MLP com camadas ocultas ReLU e saída em logits (sem ativação final).

    Args:
        dims: Sequência com as dimensões de cada camada, incluindo entrada e saída.
              Padrão: (54, 64, 32, 25) conforme especificação do projeto.
    """

    def __init__(self, dims: tuple[int, ...] = (54, 64, 32, 25)) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits de shape (..., 25)."""
        return self.net(x)


def predict_top15(model: LotofacilMLP, x: torch.Tensor) -> list[int]:
    """Retorna os 15 números com maior probabilidade (em ordem crescente).

    Args:
        model: Modelo treinado.
        x: Tensor de shape (54,) ou (1, 54).

    Returns:
        Lista de 15 inteiros em [1, 25] ordenados de forma crescente.
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0) if x.dim() == 1 else x)
        probs = torch.sigmoid(logits.squeeze(0))
        indices = torch.topk(probs, k=15).indices.tolist()
    return sorted(i + 1 for i in indices)


def build_model(seed: int = 42) -> LotofacilMLP:
    """Instancia o modelo padrão com seed fixo para reprodutibilidade."""
    torch.manual_seed(seed)
    return LotofacilMLP()