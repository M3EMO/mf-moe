"""Ticket 05 del mapa: olvido de referencia, velocidad real y prueba de capacidad (MoE fijo).

Uso (VM con GPU): python -m mf_moe.reference OUT_DIR [olvido,capacidad12,capacidad6]
- olvido: MoE fijo de 12 expertos aprende suma y después mult, en los dos escenarios; mide cuánto
  cae la suma. Sin olvido medible, las tareas o la escala cambian.
- capacidadN: MoE fijo de N expertos aprende las 12 tareas juntas hasta 88 % en todas, o hasta que
  salta la alarma. La dificultad sirve si 6 expertos NO llegan y 12 sí.
Cada experimento escribe OUT_DIR/<nombre>.json con configuración, semilla, historia y resultado.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

from mf_moe.model import build
from mf_moe.tasks import DEFAULT_DIGITS, TASKS, eval_set
from mf_moe.train import accuracy, train_until

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _opt(model):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)


def olvido(seed: int = 0) -> dict:
    out = {}
    for named in (True, False):
        model = build(12, seed=seed).to(DEVICE)
        opt, rng = _opt(model), random.Random(seed)
        ev = {t: eval_set(t, named) for t in ("suma", "mult")}
        r1 = train_until(model, opt, ["suma"], ev, named=named, rng=rng, device=DEVICE)
        before = accuracy(model, ev["suma"], DEVICE)
        r2 = train_until(model, opt, ["mult"], ev, named=named, rng=rng, device=DEVICE)
        after = accuracy(model, ev["suma"], DEVICE)
        out["con_nombre" if named else "sin_nombre"] = {
            "suma_antes": before, "suma_despues": after, "olvido": before - after,
            "mult": accuracy(model, ev["mult"], DEVICE),
            "pasos": [r1.steps, r2.steps], "estancada": [r1.stalled, r2.stalled],
            "tok_s": round((r1.tokens + r2.tokens) / max(r1.seconds + r2.seconds, 1e-9)),
            "historia": [r1.history, r2.history],
        }  # fmt: skip
    return out


def capacidad(n_experts: int, seed: int = 0) -> dict:
    model = build(n_experts, seed=seed).to(DEVICE)
    ev = {t: eval_set(t, True) for t in TASKS}
    r = train_until(model, _opt(model), list(TASKS), ev, named=True, rng=random.Random(seed), device=DEVICE)
    return {"expertos": n_experts, "pasos": r.steps, "estancada": r.stalled,
            "exactitud": r.history[-1]["acc"] if r.history else {},
            "tok_s": round(r.tokens / max(r.seconds, 1e-9)), "historia": r.history}  # fmt: skip


def main(out: str, which: str = "olvido,capacidad12,capacidad6") -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    meta = {"device": DEVICE, "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None,
            "digitos": DEFAULT_DIGITS, "semilla": 0}  # fmt: skip
    for name in which.split(","):
        res = olvido() if name == "olvido" else capacidad(int(name.removeprefix("capacidad")))
        Path(out, f"{name}.json").write_text(json.dumps({"meta": meta, name: res}, ensure_ascii=False))
        print(f"== {name} listo", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
