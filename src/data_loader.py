"""Leitura e validação do histórico de sorteios da Lotofácil."""

from pathlib import Path

import numpy as np
import pandas as pd

# Colunas de bolas no formato oficial da CEF (Bola1..Bola15)
_BOLA_COLS = [f"Bola{i}" for i in range(1, 16)]


def load_sorteios(path: str | Path) -> np.ndarray:
    """Carrega o histórico de sorteios e retorna array (N, 15) de inteiros.

    Suporta o formato oficial da CEF (XLSX/XLS com cabeçalho e colunas extras)
    e CSVs simples sem cabeçalho (15 colunas de números por linha).

    Args:
        path: Caminho para arquivo CSV ou XLSX em ordem cronológica.

    Returns:
        Array numpy de shape (N, 15), dtype int32, valores em [1, 25],
        ordenado do concurso mais antigo ao mais recente.

    Raises:
        ValueError: Se alguma linha não contiver exatamente 15 valores únicos em [1, 25].
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=0)
        if all(c in df.columns for c in _BOLA_COLS):
            df = df[_BOLA_COLS]
    else:
        df = pd.read_csv(path, header=None)

    data = df.to_numpy(dtype=np.int32)
    _validate(data)
    return data


def _validate(data: np.ndarray) -> None:
    """Valida que cada linha tem 15 inteiros únicos em [1, 25]."""
    if data.ndim != 2 or data.shape[1] != 15:
        raise ValueError(
            f"Esperado (N, 15) colunas, encontrado shape {data.shape}. "
            "Verifique se o arquivo tem exatamente 15 colunas de números."
        )

    for i, row in enumerate(data):
        unique = np.unique(row)
        if len(unique) != 15:
            raise ValueError(
                f"Linha {i + 1}: esperados 15 valores únicos, "
                f"encontrados {len(unique)} ({row.tolist()})."
            )
        if unique[0] < 1 or unique[-1] > 25:
            raise ValueError(
                f"Linha {i + 1}: todos os valores devem estar em [1, 25], "
                f"encontrado {row.tolist()}."
            )