"""B10 (mf-moe-5tq.23): ¿importa el orden del dataset? Idea del usuario: el conocimiento nuevo necesita raíces.

Mismo modelo y tareas que B9b (`colab/b9_bits.py`): 5 primitivas y 10 pares. Tres órdenes:
- raices: primitivas, después pares;
- azar: todo mezclado en una etapa;
- reves: pares, después primitivas.
Cada etapa entrena hasta el 88 % o el tope de pasos, con 10 % de repetición de las etapas anteriores (sin
repetición, B9b P3 dio olvido total y el olvido no separa nada). Al final:
- olvido: exactitud en cada tarea entrenada;
- generalización: exactitud en los 10 pares nunca vistos (el nombre «b+a» compone dos primitivas conocidas);
- información mutua experto × primitiva (`mutual_info` de B9b).
Pendiente: el orden por k-means jerárquico (Vo et al.) y las composiciones de 3.
Uso: .venv/bin/python -m colab.b10_curriculum SALIDA.json [pasos máximos por etapa] [semilla]
"""

import json
import random
import sys
import time

import numpy as np
import torch

from colab.b9_bits import PAIRS, SINGLES, eval_sets, fresh, mutual_info, routing, train
from mf_moe.train import scores

UNSEEN = [f"{a}+{b}" for a in SINGLES for b in SINGLES if a != b and f"{a}+{b}" not in PAIRS]
ORDERS = {"raices": [SINGLES, PAIRS], "azar": [SINGLES + PAIRS], "reves": [PAIRS, SINGLES]}


def main(path: str, max_steps: int = 20000, seed: int = 0) -> None:
    torch.set_num_threads(4)
    t0, res = time.time(), {"semilla": seed, "pasos_maximos": max_steps, "no_vistos": UNSEEN, "ordenes": {}}
    ev = eval_sets(SINGLES + PAIRS + UNSEEN)
    held = {e.prompt for exs in ev.values() for e in exs}
    for name, stages in ORDERS.items():
        model, opt = fresh(seed)
        rng, seen, out = random.Random(seed), [], {"etapas": []}
        for i, tasks in enumerate(stages):
            r = train(model, opt, tasks, {t: ev[t] for t in tasks}, max_steps, rng, f"{name}:{i}", replay=seen, held=held)
            out["etapas"].append({k: r[k] for k in ("pasos", "domino", "historia")})
            seen = seen + tasks
        acc = {t: round(scores(model, ev[t], "cpu")[0], 3) for t in SINGLES + PAIRS + UNSEEN}
        out |= {"pasos_totales": sum(e["pasos"] for e in out["etapas"]), "exactitud": acc,
                "entrenadas_media": round(float(np.mean([acc[t] for t in SINGLES + PAIRS])), 3),
                "no_vistos_media": round(float(np.mean([acc[t] for t in UNSEEN])), 3),
                "info_mutua": mutual_info(routing(model, {t: ev[t] for t in SINGLES}), SINGLES)}  # fmt: skip
        res["ordenes"][name] = out
        print(json.dumps({"orden": name, **{k: out[k] for k in ("pasos_totales", "entrenadas_media", "no_vistos_media")}}), flush=True)
        json.dump(res, open(path, "w"), ensure_ascii=False)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
