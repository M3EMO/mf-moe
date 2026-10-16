"""Checkpoints para retomar una corrida después de un retiro de la VM de Colab.

Guardan el modelo, el optimizador y el estado del bucle en un archivo, escrito de forma atómica, y después
un marcador `<archivo>.step` que cambia en cada guardado: el runner de Vigía baja el checkpoint cuando el
marcador cambia y lo sube a la VM nueva. Se leen con `weights_only=True`: nada de pickle arbitrario.
Cada checkpoint lleva la clave de su corrida; retomar con otra clave es un error, nunca un arranque en blanco.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch


def save_state(path: str, key: str, model, opt, state: dict) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    torch.save({"key": key, "model": model.state_dict(), "opt": opt.state_dict(), "state": state,
                "torch_rng": torch.get_rng_state(),  # el ruido de los clones usa el generador de torch
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, tmp)  # fmt: skip
    tmp.replace(p)
    p.with_name(p.name + ".step").write_text(f"{time.time():.3f}\n")


def read_state(path: str | None, key: str) -> dict | None:
    """El checkpoint entero, sin aplicarlo; None si no hay. Un modelo que crece necesita leer cuántos
    expertos tiene cada capa antes de cargar los pesos."""
    if not path or not Path(path).exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["key"] != key:
        raise ValueError(f"el checkpoint {path} es de la corrida {ck['key']!r}, no de {key!r}")
    return ck


def apply_state(ck: dict, model, opt) -> dict:
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["torch_rng"])
    if ck["cuda_rng"]:
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck["state"]


def load_state(path: str | None, key: str, model, opt) -> dict | None:
    """El estado guardado, con modelo y optimizador ya cargados; None si no hay checkpoint."""
    ck = read_state(path, key)
    return None if ck is None else apply_state(ck, model, opt)


def rng_state(rng) -> list:
    v, internal, gauss = rng.getstate()
    return [v, list(internal), gauss]


def set_rng_state(rng, s: list) -> None:
    rng.setstate((s[0], tuple(s[1]), s[2]))
