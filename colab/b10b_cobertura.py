"""B10b: ¿por qué B10 no generaliza a pares nuevos? Dos causas posibles, dos factores:
- cobertura: 10 o 15 pares entrenados de los 20 pares distintos (B10 usó 10);
- pasos: la respuesta escribe solo el resultado, o también el resultado intermedio («raíces» en los datos).

Los 5 pares no vistos son fijos y forman un ciclo (not→andor→rot→xor→par→not): cada primitiva aparece una vez
primera y una vez segunda. Ninguno equivale como función a otra tarea, así que acertarlos exige componer.
Cobertura 10 saca un segundo ciclo (not→xor→andor→par→rot→not); cada primitiva queda dos veces en cada lugar.
Orden «raices» de B10: primitivas, después pares, con 10 % de repetición.
Uso: .venv/bin/python -m colab.b10b_cobertura SALIDA.json [pasos máximos por etapa] [semilla]
"""

import json
import random
import sys
import time

import numpy as np
import torch

from colab.b9_bits import SINGLES, eval_sets, fresh, train
from mf_moe.train import scores


def _cycle(order: str) -> list[str]:
    p = order.split()
    return [f"{a}+{b}" for a, b in zip(p, p[1:] + p[:1])]


UNSEEN = _cycle("not andor rot xor par")
DROPPED = _cycle("not xor andor par rot")
POOL = [f"{a}+{b}" for a in SINGLES for b in SINGLES if a != b and f"{a}+{b}" not in UNSEEN]
COVERAGE = {10: [t for t in POOL if t not in DROPPED], 15: POOL}
assert len(UNSEEN) == len(DROPPED) == 5 and len(POOL) == 15 and not set(DROPPED) & set(UNSEEN)


def main(path: str, max_steps: int = 20000, seed: int = 0) -> None:
    torch.set_num_threads(4)
    t0, res = time.time(), {"semilla": seed, "pasos_maximos": max_steps, "no_vistos": UNSEEN, "condiciones": {}}
    for steps in (False, True):
        ev = eval_sets(SINGLES + POOL + UNSEEN, steps=steps)
        held = {e.prompt for exs in ev.values() for e in exs}
        for n, pairs in COVERAGE.items():
            name = f"{n}{'_pasos' if steps else ''}"
            model, opt = fresh(seed)
            rng, seen, out = random.Random(seed), [], {"pares": pairs, "etapas": []}
            for i, tasks in enumerate((SINGLES, pairs)):
                r = train(model, opt, tasks, {t: ev[t] for t in tasks}, max_steps, rng, f"{name}:{i}",
                          replay=seen, held=held, steps=steps)  # fmt: skip
                out["etapas"].append({k: r[k] for k in ("pasos", "domino", "historia")})
                seen = seen + tasks
            acc = {t: round(scores(model, ev[t], "cpu")[0], 3) for t in SINGLES + pairs + UNSEEN}
            out |= {"pasos_totales": sum(e["pasos"] for e in out["etapas"]), "exactitud": acc,
                    "entrenadas_media": round(float(np.mean([acc[t] for t in SINGLES + pairs])), 3),
                    "no_vistos_media": round(float(np.mean([acc[t] for t in UNSEEN])), 3)}  # fmt: skip
            res["condiciones"][name] = out
            print(json.dumps({"condicion": name, **{k: out[k] for k in ("pasos_totales", "entrenadas_media", "no_vistos_media")}}), flush=True)
            json.dump(res, open(path, "w"), ensure_ascii=False)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
