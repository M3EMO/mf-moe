"""Entrenamiento hasta dominar una tarea (88 %) y evaluación exacta (tickets 04 y 05 del mapa).

Exactitud = respuesta exacta. Con teacher forcing, una respuesta es exacta si el argmax acierta cada
token de la respuesta (EOS incluido): equivale a decodificar greedy, en una sola pasada.
No hay tope de pasos (decisión del usuario). La alarma pausa y avisa si ni la exactitud ni el acierto por
dígito mejoran durante `patience` evaluaciones seguidas en ninguna tarea: devuelve `stalled=True` y el llamador decide.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from mf_moe.model import PAD
from mf_moe.tasks import Example, make_example

MASTERY = 0.88


def encode(examples: Sequence[Example], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """input_ids y labels con -100 fuera de la respuesta; relleno a la derecha con PAD."""
    seqs = [(list(e.prompt.encode()), list(e.answer.encode())) for e in examples]
    width = max(len(p) + len(a) for p, a in seqs)
    ids = torch.full((len(seqs), width), PAD, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, (p, a) in enumerate(seqs):
        ids[i, : len(p) + len(a)] = torch.tensor(p + a)
        labels[i, len(p) : len(p) + len(a)] = torch.tensor(a)
    return ids.to(device), labels.to(device)


@torch.no_grad()
def scores(model, examples: Sequence[Example], device: str, batch: int = 250) -> tuple[float, float, list[float]]:
    """Respuesta exacta, acierto por byte de respuesta y acierto en cada posición de la respuesta."""
    model.eval()
    ok, pos, hit_at = 0, [], []
    for i in range(0, len(examples), batch):
        ids, labels = encode(examples[i : i + batch], device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model(input_ids=ids).logits.argmax(-1)
        target, mask = labels[:, 1:], labels[:, 1:] != -100
        hit = pred[:, :-1] == target
        ok += int((hit | ~mask).all(dim=1).sum())
        pos.append((mask.cumsum(1) - 1)[mask].cpu())  # posición dentro de la respuesta
        hit_at.append(hit[mask].float().cpu())
    model.train()
    pos, hit_at = torch.cat(pos), torch.cat(hit_at)
    per_pos = torch.bincount(pos, weights=hit_at) / torch.bincount(pos)
    return ok / len(examples), float(hit_at.mean()), per_pos.tolist()


def accuracy(model, examples: Sequence[Example], device: str, batch: int = 250) -> float:
    return scores(model, examples, device, batch)[0]


def sample(tasks: Sequence[str], rng: random.Random, named: bool, digits: dict | None, held: set[str]) -> Example:
    """Un ejemplo de entrenamiento cuya entrada no está en la evaluación: la exactitud mide entradas nuevas."""
    for _ in range(1000):
        ex = make_example(rng.choice(tasks), rng, named, digits)
        if ex.prompt not in held:
            return ex
    raise RuntimeError(f"la evaluación cubre casi todo el espacio de entradas de {tasks}")


@dataclass
class Result:
    steps: int = 0
    tokens: int = 0
    seconds: float = 0.0  # solo entrenamiento, sin evaluar
    stalled: bool = False
    history: list[dict] = field(default_factory=list)


def train_until(
    model,
    opt: torch.optim.Optimizer,
    tasks: Sequence[str],
    evals: dict[str, list[Example]],
    *,
    named: bool,
    rng: random.Random,
    device: str,
    batch: int = 128,
    eval_every: int = 250,
    patience: int = 20,
    log: Callable[[dict], None] = lambda r: print(json.dumps(r), flush=True),
    digits: dict | None = None,
) -> Result:
    """Entrena con lotes parejos de `tasks` hasta que TODAS lleguen al 88 % (o salte la alarma)."""
    res, best, best_dig, since = Result(), dict.fromkeys(tasks, -1.0), dict.fromkeys(tasks, -1.0), 0
    held = {e.prompt for exs in evals.values() for e in exs}  # bin y modmult vieron toda la evaluación (2026-10-15)
    model.train()
    while True:
        t0 = time.perf_counter()
        exs = [sample(tasks, rng, named, digits, held) for _ in range(batch)]
        ids, labels = encode(exs, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        res.steps += 1
        res.tokens += int((ids != PAD).sum())
        res.seconds += time.perf_counter() - t0
        if res.steps % eval_every:
            continue
        sc = {t: scores(model, evals[t], device) for t in tasks}
        accs, digs = {t: s[0] for t, s in sc.items()}, {t: round(s[1], 4) for t, s in sc.items()}
        worst = min(accs.values())
        rec = {"step": res.steps, "loss": round(loss.item(), 4), "min_acc": worst, "acc": accs, "dig": digs,
               "tok_s": round(res.tokens / res.seconds)}  # fmt: skip
        res.history.append(rec)
        log(rec)
        if worst >= MASTERY:
            return res
        # Alarma: ninguna tarea mejoró 1 punto. Mirar solo la peor la disparaba con una tarea trabada
        # en 0 % aunque las demás avanzaran (ref 2026-10-14: 12 tareas cortadas a los 5.250 pasos).
        # También cuenta el acierto por dígito: la respuesta exacta no ve el progreso previo a un salto.
        improved = [t for t in tasks if accs[t] > best[t] + 0.01 or digs[t] > best_dig[t] + 0.01]
        for t in improved:  # la referencia sube solo al mejorar: una deriva lenta también cuenta
            best[t], best_dig[t] = max(best[t], accs[t]), max(best_dig[t], digs[t])
        if improved:
            since = 0
        else:
            since += 1
            if since >= patience:
                res.stalled = True
                return res
