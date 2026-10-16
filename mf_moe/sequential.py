"""Bucle secuencial del experimento: las doce tareas en fila, cada una sola hasta 88 % o hasta la alarma.

Después de cada tarea se evalúan las doce y se llena una fila de la matriz R: R[i][t] es la exactitud en la
tarea t al terminar la i-ésima tarea del orden. De R salen la exactitud media final, el BWT y el olvido; los
pasos de cada tarea, comparados con los de la tarea sola, dan la transferencia hacia adelante (ticket 09).

Orden (ticket 04): uno al azar por semilla, el mismo para todas las condiciones de esa semilla.
Repetición (ticket 08): al terminar cada tarea se guardan 1.000 ejemplos suyos, con entradas fuera de la
evaluación; desde la segunda tarea, el 10 % de cada lote sale de ese buffer, parejo entre tareas viejas.
Condiciones (ticket 09): las 14 de `CONDITIONS`. La división revisa cada 500 pasos dentro de la tarea; la
fusión consolida al terminar la tarea, antes de evaluar; Lifelong-MoE crece al empezar la tarea.
"""

from __future__ import annotations

import random
from dataclasses import asdict, astuple, dataclass

from mf_moe.division import Divider, resize
from mf_moe.lifelong import Lifelong
from mf_moe.merging import Merger
from mf_moe.model import build, routers
from mf_moe.resume import apply_state, read_state, rng_state, save_state, set_rng_state
from mf_moe.tasks import TASKS, Example, eval_set
from mf_moe.train import accuracy, adamw, encode, sample, train_until

BUFFER = 1000  # ejemplos guardados por tarea (ticket 08)
CALIB = 32  # entradas por tarea aprendida en el lote fijo con que la fusión compara salidas


@dataclass(frozen=True)
class Condition:
    name: str
    n_experts: int = 12  # expertos ruteados por capa al empezar
    replay: bool = False
    divide: str | None = None  # señal del disparador: "duda", "perdida" o "ambas"
    merge: str | None = None  # criterio de fusión: "salidas", "pesos" o "ambas"
    lifelong: bool = False


FIJO = Condition("fijo")
FIJO_REPETICION = Condition("fijo+repetición", replay=True)
PRINCIPAL = Condition("fractal/duda+salidas", 6, replay=True, divide="duda", merge="salidas")
CONDITIONS = {c.name: c for c in [
    FIJO,
    FIJO_REPETICION,
    Condition("lifelong-moe", 6, lifelong=True),
    *[Condition(f"fractal/{d}+{m}", 6, replay=True, divide=d, merge=m)
      for d in ("duda", "perdida", "ambas") for m in ("salidas", "pesos", "ambas")],
    Condition("fractal/duda, sin fusión", 6, replay=True, divide="duda"),
    Condition("fractal/duda+salidas, sin repetición", 6, divide="duda", merge="salidas"),
]}  # fmt: skip


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


def _calibration(tasks: list[str], seed: int, named: bool, evals: dict, device: str):
    """Lote fijo de entradas de entrenamiento de las tareas aprendidas, fuera de la evaluación."""
    exs = []
    for t in tasks:
        rng, held = random.Random(f"calib-{seed}-{t}"), {e.prompt for e in evals[t]}
        exs += [sample([t], rng, named, None, held) for _ in range(CALIB)]
    return encode(exs, device)[0]


def run(cond: Condition, seed: int, named: bool, device: str, eval_n: int = 1000, ckpt: str | None = None,
        order: list[str] | None = None, **train_kw) -> dict:  # fmt: skip
    """Con `ckpt`, guarda cada 10.000 pasos y al cerrar cada tarea, y retoma desde ahí si el archivo existe.
    `order` reemplaza el orden de la semilla, por ejemplo en un piloto con menos tareas."""
    model = build(cond.n_experts, seed=seed).to(device)
    rng, order = random.Random(seed), order or task_order(seed)
    evals = {t: eval_set(t, named, n=eval_n) for t in TASKS}
    key = f"secuencial/{cond.name}/semilla{seed}/{'con' if named else 'sin'}_nombre/eval{eval_n}/{'-'.join(order)}"
    div = Divider(signal=cond.divide) if cond.divide else None
    mer = Merger(criterion=cond.merge) if cond.merge else None
    ll = Lifelong() if cond.lifelong else None
    buffer: dict[str, list[Example]] = {}
    R, steps, stalled, experts, start, inner = [], [], [], [], 0, None
    ck = read_state(ckpt, key)
    if ck is not None:  # un modelo que creció necesita sus capas del tamaño guardado antes de cargar
        st = ck["state"]
        resize(model, st["expertos_por_capa"])
        opt = adamw(model)
        apply_state(ck, model, opt)
        R, steps, stalled, experts = st["R"], st["pasos"], st["estancada"], st["expertos"]
        start, inner = st["tarea"], st["dentro"]
        buffer = {t: [Example(*e) for e in exs] for t, exs in st["buffer"].items()}
        if div:
            div.events, div._acc = st["division"], st["division_acumulada"]
        if mer:
            mer.events = st["fusion"]
        if ll and st["maestro"] is not None:
            teacher = build(cond.n_experts, seed=seed).to(device)
            resize(teacher, st["maestro_expertos"])
            teacher.load_state_dict(st["maestro"])
            ll.teacher, ll.frozen = teacher.eval().requires_grad_(False), st["congeladas"]
            ll.refreeze(model, opt)
        set_rng_state(rng, st["rng"])
    else:
        opt = adamw(model)
    if div:
        div.attach(model)

    def save(i: int, inside: dict | None) -> None:
        teacher = ll.teacher if ll else None
        save_state(ckpt, key, model, opt, {
            "R": R, "pasos": steps, "estancada": stalled, "expertos": experts, "tarea": i, "dentro": inside,
            "buffer": {t: [astuple(e) for e in exs] for t, exs in buffer.items()}, "rng": rng_state(rng),
            "expertos_por_capa": [r.num_experts for r in routers(model)],
            "division": div.events if div else [], "division_acumulada": div._acc if div and inside else [],
            "fusion": mer.events if mer else [],
            "maestro": teacher.state_dict() if teacher else None,
            "maestro_expertos": [r.num_experts for r in routers(teacher)] if teacher else None,
            "congeladas": ll.frozen if ll else [],
        })  # fmt: skip

    for i in range(start, len(order)):
        t, resumed = order[i], i == start and inner is not None
        if ll and not resumed:
            ll.start_task(model, opt, i)
        if div and not resumed:
            div._acc = []  # ventana nueva en cada tarea: la fusión y el crecimiento cambian las capas
        r = train_until(model, opt, [t], {t: evals[t]}, named=named, rng=rng, device=device,
                        replay=buffer if cond.replay else None, resume=inner if resumed else None,
                        save=(lambda inside, i=i: save(i, inside)) if ckpt else None,
                        on_step=div, aux_loss=ll.aux_loss if ll else None,
                        **{"patience": 40, **train_kw})  # fmt: skip
        if mer:
            mer(model, opt, _calibration(order[: i + 1], seed, named, evals, device), task=t)
        R.append({u: accuracy(model, evals[u], device) for u in TASKS})
        steps.append(r.steps)
        stalled.append(r.stalled)
        experts.append([x.num_experts for x in routers(model)])
        if cond.replay:
            held = {e.prompt for e in evals[t]}
            buffer[t] = [sample([t], rng, named, None, held) for _ in range(BUFFER)]
        if ckpt:
            save(i + 1, None)
    return {"condicion": asdict(cond), "semilla": seed, "con_nombre": named, "orden": order, "R": R,
            "pasos": steps, "pasos_totales": sum(steps), "estancada": stalled, "expertos": experts,
            "eventos_division": div.events if div else [], "eventos_fusion": mer.events if mer else [],
            **metrics(R, order)}  # fmt: skip


def main(out: str, name: str, seed: str, named: str, ckpt: str | None = None) -> None:
    """Uso (VM con GPU): python -m mf_moe.sequential OUT_DIR CONDICION SEMILLA con|sin [CHECKPOINT]."""
    import json
    from pathlib import Path

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = run(CONDITIONS[name], int(seed), named == "con", device, ckpt=ckpt)
    res["meta"] = {"device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None}
    slug = f"{name.replace('/', '_').replace(', ', '_').replace(' ', '_')}_s{seed}_{named}"
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, f"{slug}.json").write_text(json.dumps(res, ensure_ascii=False))
    print(f"== {slug}: bwt={res['bwt']:.4f} exactitud={res['exactitud_media']:.4f} pasos={res['pasos_totales']}",
          flush=True)  # fmt: skip


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:6])
