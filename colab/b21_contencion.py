"""B21 (mapa mf-moe-generalidad, ticket 01): con la misma capacidad, ¿contener o consolidar?

Banco y conjuntos de B20. Sus controles ya están pagados: acá solo corren los brazos y unas repeticiones del control.
- cont8: entrenan los 8 expertos que más usan los hechos nuevos. Son los populares: casi los mismos que usan las anclas;
- esp8: entrenan los 8 más específicos del tema (uso del tema / uso de las anclas). Es la contención más justa sin cambiar el ruteo;
- cons: se congelan los que más usan las anclas, lo viejo que el banco protege (la regla de §4.12);
- azar: los mismos números, al azar;
- prec24: como cons24, pero ordena por sensibilidad de lo viejo a cada experto (Fisher diagonal), no por uso. Secundario;
- cont8_tronco: como cont8 y además congela todo lo que no es experto. Es el techo de la contención.
Uso: .venv/bin/python -m colab.b21_contencion SALIDA.json
"""

import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab import b19_gradientes as b19
from colab import b20_matriz as b20
from colab.b16_conceptos import DEVICE, TRAIN_FROM, name, probes, wordnet
from mf_moe import granite
from mf_moe.model import moe
from mf_moe.phase2 import amp, set_trainable

# brazo: (regla, fracción de expertos entrenables, congelar lo que no es experto)
ARMS = {"control": ("todos", 1.0, False), "cont8": ("nuevo", 0.25, False), "esp8": ("tema", 0.25, False), "cons8": ("viejo", 0.25, False), "azar8": ("azar", 0.25, False),
        "cont8_tronco": ("nuevo", 0.25, True), "cons24": ("viejo", 0.75, False), "azar24": ("azar", 0.75, False), "prec24": ("viejo", 0.75, False)}  # fmt: skip
SETS, N_USE, MAX_HOURS, GIVE_UP = (0, 1), 64, 3.0, 3  # GIVE_UP: corridas seguidas sin aprender tras las que un brazo se abandona


def frozen_rows(arm: str, U_new: np.ndarray, U_old: np.ndarray, rng, U_topic: np.ndarray | None = None) -> list[list[int]]:
    """Expertos congelados por capa. U: [capas, expertos]. Empates fijos: argsort estable sobre el uso."""
    rule, share, _ = ARMS[arm]
    n = U_new.shape[1]
    keep = round(n * share)
    out = []
    for i, (new, old) in enumerate(zip(U_new, U_old)):
        if rule == "todos":
            rows = []
        elif rule == "nuevo":
            rows = np.argsort(-new, kind="stable")[keep:]  # todo menos lo que más usa lo nuevo
        elif rule == "tema":
            rows = np.argsort(-(U_topic[i] + 1e-3) / (old + 1e-3), kind="stable")[keep:]  # 1e-3: un experto sin uso no da cociente infinito
        elif rule == "viejo":
            rows = np.argsort(-old, kind="stable")[: n - keep]  # lo que más usa lo viejo
        else:
            rows = rng.permutation(n)[keep:]
        out.append(sorted(int(x) for x in rows))
    return out


def sensitivity(model, tok, items: list, layers) -> np.ndarray:
    """Suma de gradientes al cuadrado por experto, ejemplo por ejemplo: [capas, expertos]. Excluye al router."""
    model.eval()
    model.zero_grad(set_to_none=True)
    params = [(e.gate_up_proj, e.down_proj) for e, _ in (moe(model, i) for i in layers)]
    out = np.zeros((len(params), params[0][0].shape[0]))
    for item in items:
        ids, mask, labels = b17.batch(tok, [item])
        with amp(DEVICE):
            model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
        for j, pair in enumerate(params):
            for p in pair:
                out[j] += p.grad.float().pow(2).flatten(1).sum(1).cpu().numpy()
        model.zero_grad(set_to_none=True)
    model.train()
    return out


class Mask:
    """Se engancha a `b18.run_set` como el probe de B19. Congela antes de crear el optimizador y lo deshace después."""

    def __init__(self, arm: str, seed: int, topic: list | None = None):
        self.arm, self.seed, self.topic = arm, seed, topic

    def pre(self, model, tok, new, keep) -> None:
        layers = range(TRAIN_FROM, len(model.model.layers))
        U_new = b19.usage(model, tok, new).mean(0)[TRAIN_FROM:]
        self.rows = self.choose(model, tok, new, keep[:N_USE], layers, U_new)
        free =[np.setdiff1d(np.arange(U_new.shape[1]), rows) for rows in self.rows]
        self.share = float(np.mean([u[f].sum() for u, f in zip(U_new, free)]))  # cuánto del uso de lo nuevo cae en lo que puede moverse
        self.hooks, self.watch, self.off = [], [], []
        names = {id(p): n for n, p in model.named_parameters()}
        for i, rows in zip(layers, self.rows):
            experts, router = moe(model, i)
            for p in (experts.gate_up_proj, experts.down_proj, router.weight):
                if rows:
                    mask = torch.ones(p.shape[0], *[1] * (p.dim() - 1), device=p.device, dtype=p.dtype)
                    mask[rows] = 0
                    self.hooks.append(p.register_hook(lambda g, m=mask: g * m))
                    self.watch.append((names[id(p)], p, rows))
        if ARMS.get(self.arm, ("", 0.0, False))[2]:  # un brazo de una subclase (B23) no congela el tronco
            self.off =[p for n, p in model.named_parameters() if p.requires_grad and b19.block(n) != "expertos"]
            for p in self.off:
                p.requires_grad_(False)

    def choose(self, model, tok, new, old, layers, U_new) -> list[list[int]]:
        """Expertos congelados por capa. B23 cambia solo esta regla."""
        U_old = sensitivity(model, tok, old, layers) if self.arm == "prec24" else b19.usage(model, tok, old).mean(0)[TRAIN_FROM:]
        U_topic = b19.usage(model, tok, self.topic).mean(0)[TRAIN_FROM:] if ARMS[self.arm][0] == "tema" else None
        return frozen_rows(self.arm, U_new, U_old, np.random.default_rng(self.seed), U_topic)

    def post(self, model, state: dict) -> None:
        self.moved = max((float((p.detach()[rows].float() - state[n][rows].to(p.device).float()).abs().max()) for n, p, rows in self.watch), default=0.0)
        for h in self.hooks:
            h.remove()
        for p in self.off:
            p.requires_grad_(True)

    def final(self, model, tok) -> dict:
        return {"brazo": self.arm, "congelados": self.rows, "fuga": self.moved, "uso_entrenable": self.share}


def given_up(runs: dict, arm: str) -> bool:
    """Un brazo que no aprende ya dio su resultado: seguirlo hasta MAX_STEPS en cada tema solo gasta unidades."""
    mine = [r["nuevos_final"] for k, r in runs.items() if k.endswith(f"|{arm}")]
    return len(mine) >= GIVE_UP and all(x >= b17.STOP_LOSS for x in mine)


def plan(areas=None) -> list[tuple]:
    """Por conjunto y por tema, todos los brazos: si el tope de tiempo corta, los temas ya corridos quedan pareados."""
    areas = list(b20.TOPICS) if areas is None else areas
    first = {}
    for a, parent in b20.TOPICS.items():
        first.setdefault(parent, a)
    checks = [(a, 0, "control") for a in first.values() if a in areas]  # ¿reutilizar el control de B20 es válido?
    return checks + [(a, s, arm) for s in SETS for a in areas for arm in ARMS if arm != "control"]


def main(path: str, model=None, tok=None) -> None:
    res = json.load(open(path)) if os.path.exists(path) else {"areas": {}}
    wn = wordnet()
    if model is None:
        from transformers import AutoTokenizer

        model, tok = granite.load(DEVICE), AutoTokenizer.from_pretrained(granite.MODEL_ID)
        set_trainable(model, TRAIN_FROM)
    ev, chosen, pools = b20.split(wn)
    res |= {"temas": b20.TOPICS, "evaluacion": chosen, "config": {"brazos": ARMS, "conjuntos": SETS, "n_uso": N_USE, "train_from": TRAIN_FROM, "max_horas": MAX_HOURS}}
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    H, E = probes(model, tok, [name(x) for x in ev])
    antes = b17.def_losses(model, tok, [b18.fact(x, x.definition()) for x in ev]).round(4).tolist()
    t0 = time.time()
    for area, s, arm in plan():
        key = f"{area}|s{s}|{arm}"
        if key in res["areas"] or given_up(res["areas"], arm):
            continue
        if time.time() - t0 > MAX_HOURS * 3600:
            res["corte"] = f"tope de {MAX_HOURS} h"
            break
        root = wn.synset(area)
        base = {"H": H, "E": E, "antes": antes, "raiz": [round(x.wup_similarity(root) or 0.0, 4) for x in ev]}
        topic = [b18.fact(x, x.definition()) for x in random.Random(f"tema:{area}").sample(pools[area], min(N_USE, len(pools[area])))]
        probe = Mask(arm, seed=sum(map(ord, area)) + s, topic=topic)  # hash() cambia entre procesos: el azar tiene que poder repetirse
        res["areas"][key] = r = b18.run_set(model, tok, wn, area, ev, pools[area], base, state, s, 0, probe=probe)
        print(json.dumps({"corrida": key, "subida_media": round(float(np.mean(r["subida"])), 3), "nuevos": round(r["nuevos_final"], 3),
                          "pasos": r["pasos"], "fuga": r["fuga"], "segundos": r["segundos"]}), flush=True)  # fmt: skip
        res["corte"] = b18.gate(r) if len(res["areas"]) == 1 else None
        json.dump(res, open(path, "w"), ensure_ascii=False)
        if res["corte"]:
            print(json.dumps({"corte": res["corte"]}), flush=True)
            return
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:2])
