"""B14 (ticket 15): tamiz en CPU de reglas de congelado. ¿Qué regla elige la fase 3 (B5 graduada o B8 TF-IDF)?

Banco: la fila de B9b sin repetición (5 primitivas, después 10 pares; las primitivas caen a 0-1 %). La etapa A
corre una vez por semilla hasta el dominio; cada rama copia ese modelo con su AdamW, congela según su regla y
entrena los pares STAGE_B pasos fijos (misma exposición para todas). Ramas:
- control: nada;
- uso: los k expertos por capa más usados en las primitivas (la consolidación por uso de la fase 2);
- azar: k expertos por capa al azar (¿importa cuáles?);
- tfidf: uso en las primitivas / uso en ejemplos sin nombre de tarea, el «texto general» del banco (`_tfidf`);
- graduada: lr de cada experto ∝ 1/(1+c·Ω), con Ω de la etapa A (`_graded_step` de la fase 2);
- abajo: congela la capa 0 entera («Spurious Forgetting», ICLR 2025). Los embeddings quedan libres: están atados
  a la salida, y congelarlos congelaría también la cabeza.
Medidas: exactitud y pérdida de cada primitiva al cerrar A y al final (la fase 2 mide el olvido en pérdida: la
exactitud exacta puede quedar en 0 % aunque la pérdida mejore), y exactitud de los pares al final y su historia.
Uso: .venv/bin/python -m colab.b14_congelado SALIDA.json SEMILLA explora|confirma [k c]
"""

import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

from colab.b9_bits import DEVICE, PAIRS, SINGLES, eval_sets, fresh, routing, train
from mf_moe.bits import make_bits_example
from mf_moe.lifelong import freeze_rows
from mf_moe.model import moe, routers
from mf_moe.phase2 import _graded_step, _tfidf
from mf_moe.train import encode, scores

STAGE_A_MAX, STAGE_B, N_GENERAL = 8000, 4000, 300
EXPLORE = {"control": {}, "uso_k3": {"k": 3}, "uso_k8": {"k": 8},
           "graduada_c1": {"c": 1.0}, "graduada_c10": {"c": 10.0}, "graduada_c100": {"c": 100.0}}  # fmt: skip


def confirm(k: int, c: float) -> dict:
    return {"control": {}, "uso": {"k": k}, "azar": {"k": k, "azar": True}, "tfidf": {"k": k, "tfidf": True},
            "graduada": {"c": c}, "abajo": {"abajo": True}}  # fmt: skip


def usage(model, evals: dict) -> np.ndarray:
    """[capas, expertos]: elecciones del router en las posiciones de la respuesta."""
    idx = np.concatenate([v["idx"] for v in routing(model, evals).values()])
    n = routers(model)[0].num_experts
    return np.stack([np.bincount(idx[:, L].reshape(-1), minlength=n) for L in range(idx.shape[1])])


@torch.no_grad()
def losses(model, evals: dict) -> dict:
    """Pérdida media por token de la respuesta, por tarea."""
    model.eval()
    out = {}
    for t, exs in evals.items():
        ids, labels = encode(exs, DEVICE)
        out[t] = round(model(input_ids=ids, labels=labels).loss.item(), 4)
    model.train()
    return out


def choose(arm: dict, u: np.ndarray, bg: np.ndarray, seed: int) -> list[list[int]]:
    """Los k expertos que congela cada capa."""
    if arm.get("azar"):
        return [sorted(random.Random(f"{seed}/{L}").sample(range(u.shape[1]), arm["k"])) for L in range(len(u))]
    score = [_tfidf(torch.tensor(a), torch.tensor(b)).numpy() for a, b in zip(u, bg)] if arm.get("tfidf") else u
    return [sorted(np.argsort(-s, kind="stable")[: arm["k"]].tolist()) for s in score]


def branch(model, opt, arm: dict, omega: dict, prims: dict, general: dict, pairs: dict, seed: int) -> dict:
    """Congela según `arm` y entrena los pares sobre este modelo (main le pasa una copia de la etapa A)."""
    layers, out, after = list(range(len(model.model.layers))), {}, None
    if arm.get("abajo"):
        for p in model.model.layers[0].parameters():
            p.requires_grad_(False)  # AdamW salta los parámetros sin gradiente
    if "k" in arm:
        out["congelados"] = choose(arm, usage(model, prims), usage(model, general), seed)
        for L, rows in zip(layers, out["congelados"]):
            experts, router = moe(model, L)
            for p in (experts.gate_up_proj, experts.down_proj, router.weight):
                freeze_rows(opt, p, rows)
    if "c" in arm:
        scratch = {L: torch.zeros_like(omega[L]) for L in layers}  # Ω queda fija en la etapa B: un solo cierre
        after = lambda: _graded_step(opt, model, layers, omega, scratch, arm["c"])  # noqa: E731
    out["etapa_b"] = train(model, opt, PAIRS, pairs, STAGE_B, random.Random(seed + 2), "B", after_step=after,
                           until_mastery=False)  # fmt: skip
    out["primitivas_despues"] = {t: round(scores(model, prims[t], DEVICE)[0], 3) for t in SINGLES}
    out["perdida_primitivas"] = losses(model, prims)
    return out


def main(path: str, seed: int, mode: str, k: int = 3, c: float = 10.0) -> None:
    torch.set_num_threads(4)
    res = json.load(open(path)) if os.path.exists(path) else {"semilla": seed, "modo": mode, "ramas": {}}
    save = lambda: json.dump(res, open(path, "w"), ensure_ascii=False)  # noqa: E731
    ev = eval_sets(SINGLES + PAIRS)
    prims, pairs = {t: ev[t] for t in SINGLES}, {t: ev[t] for t in PAIRS}
    pick = random.Random(f"general:{seed}")
    general = {"general": [make_bits_example(pick.choice(SINGLES), random.Random(f"general:{i}"))
                           for i in range(N_GENERAL)]}  # fmt: skip
    model, opt = fresh(seed)
    layers = list(range(len(model.model.layers)))
    acc = {L: torch.zeros(routers(model)[L].num_experts) for L in layers}
    ckpt = path + ".etapa_a.pt"
    if os.path.exists(ckpt):  # retoma: la etapa A es la misma para todas las ramas
        st = torch.load(ckpt)
        model.load_state_dict(st["modelo"])
        opt.load_state_dict(st["opt"])
        acc = st["acc"]
    else:
        # con Ω = 0 el paso graduado no cambia nada (escala 1): solo suma la importancia -g·Δθ de cada experto
        zero = {L: torch.zeros_like(acc[L]) for L in layers}
        res["etapa_a"] = train(model, opt, SINGLES, prims, STAGE_A_MAX, random.Random(seed + 1), "A",
                               after_step=lambda: _graded_step(opt, model, layers, zero, acc, 1.0))  # fmt: skip
        torch.save({"modelo": model.state_dict(), "opt": opt.state_dict(), "acc": acc}, ckpt)
        save()
    res["perdida_cierre_a"] = losses(model, prims)
    omega = {L: acc[L].clamp_min(0) for L in layers}
    for name, arm in (EXPLORE if mode == "explora" else confirm(k, c)).items():
        if name in res["ramas"]:
            continue
        t0 = time.time()
        res["ramas"][name] = branch(*copy.deepcopy((model, opt)), arm, omega, prims, general, pairs, seed) | {"regla": arm}
        res["ramas"][name]["segundos"] = round(time.time() - t0)
        print(json.dumps({"rama": name, "primitivas": res["ramas"][name]["primitivas_despues"],
                          "perdida": res["ramas"][name]["perdida_primitivas"],
                          "pares": res["ramas"][name]["etapa_b"]["historia"][-1]["media"]}), flush=True)  # fmt: skip
        save()


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), a[2], *((int(a[3]), float(a[4])) if len(a) > 3 else ()))
