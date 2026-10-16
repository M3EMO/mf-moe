"""Bucle secuencial del experimento: las doce tareas en fila, cada una sola hasta 88 % o hasta la alarma.

Después de cada tarea se evalúan las doce y se llena una fila de la matriz R: R[i][t] es la exactitud en la
tarea t al terminar la i-ésima tarea del orden. De R salen la exactitud media final, el BWT y el olvido; los
pasos de cada tarea, comparados con los de la tarea sola, dan la transferencia hacia adelante (ticket 09).

Orden (ticket 04): uno al azar por semilla, el mismo para todas las condiciones de esa semilla.
Repetición (ticket 08): al terminar cada tarea se guardan 1.000 ejemplos suyos, con entradas fuera de la
evaluación; desde la segunda tarea, el 10 % de cada lote sale de ese buffer, parejo entre tareas viejas.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from mf_moe.model import build
from mf_moe.tasks import TASKS, eval_set
from mf_moe.train import accuracy, adamw, sample, train_until

BUFFER = 1000  # ejemplos guardados por tarea (ticket 08)


@dataclass(frozen=True)
class Condition:
    name: str
    n_experts: int = 12
    replay: bool = False


FIJO = Condition("fijo")
FIJO_REPETICION = Condition("fijo+repetición", replay=True)


def task_order(seed: int) -> list[str]:
    order = list(TASKS)
    random.Random(f"orden-{seed}").shuffle(order)
    return order


def metrics(R: list[dict[str, float]], order: list[str]) -> dict[str, float]:
    """Exactitud media final, BWT y olvido (Chaudhry et al. 2018) sobre la matriz R en el orden de entrenamiento."""
    last, T = R[-1], len(order)
    old = range(T - 1)
    return {
        "exactitud_media": sum(last[t] for t in order) / T,
        "bwt": sum(last[order[j]] - R[j][order[j]] for j in old) / (T - 1),
        "olvido": sum(max(R[i][order[j]] for i in range(j, T - 1)) - last[order[j]] for j in old) / (T - 1),
    }


def run(cond: Condition, seed: int, named: bool, device: str, eval_n: int = 1000, **train_kw) -> dict:
    model = build(cond.n_experts, seed=seed).to(device)
    opt, rng, order = adamw(model), random.Random(seed), task_order(seed)
    evals = {t: eval_set(t, named, n=eval_n) for t in TASKS}
    buffer: dict[str, list] = {}
    R, steps, stalled = [], [], []
    for t in order:
        r = train_until(model, opt, [t], {t: evals[t]}, named=named, rng=rng, device=device,
                        replay=buffer if cond.replay else None, **{"patience": 40, **train_kw})  # fmt: skip
        R.append({u: accuracy(model, evals[u], device) for u in TASKS})
        steps.append(r.steps)
        stalled.append(r.stalled)
        if cond.replay:
            held = {e.prompt for e in evals[t]}
            buffer[t] = [sample([t], rng, named, None, held) for _ in range(BUFFER)]
    return {"condicion": asdict(cond), "semilla": seed, "con_nombre": named, "orden": order, "R": R,
            "pasos": steps, "estancada": stalled, **metrics(R, order)}  # fmt: skip
