"""Piloto 5 de la fase 2 (ticket 12): dominios nuevos (idiomas fuera de Granite) y fila del hijo al dividir.

El piloto 4 mostró que Granite ya sabía matemática, código, biomedicina y derecho (entrenarlos subía su pérdida),
y que dividir con la fila del hijo en la dirección de la duda daña al modelo preentrenado (+2,5 nats con 5
divisiones por capa). Este piloto mide:
1. Pérdida del preentrenado en cada idioma (cuánto queda por aprender).
2. Daño y uso de 16 divisiones por capa sin entrenar, con la fila del hijo `mix` = 1 (fase 1), 0,1 y 0 (clon).
3. Aprendizaje y olvido sin crecer: finés y después turco, 1.600 secuencias cada uno, lote 64, lr 2e-5 y 6e-5.
4. La división con el umbral real (z 3,5, revisión cada 5 pasos) y `mix` 0,1 y 1, en finés y turco.
Uso: python pilot5_phase2.py DATOS OUT
"""

import json
import sys
import time
from pathlib import Path

import torch

import mf_moe.phase2 as P
from mf_moe.division import Divider
from mf_moe.granite import load
from mf_moe.model import routers

DOMS = ("fines", "turco")


def main(data_dir: str, out: str) -> None:
    data = P.load_data(data_dir)
    Path(out).mkdir(parents=True, exist_ok=True)
    res = {}

    def emit(key, value):
        res[key] = value
        (Path(out) / "pilot5.json").write_text(json.dumps(res, ensure_ascii=False))
        print("== " + json.dumps({key: value}), flush=True)

    def losses(model, doms=(*P.DOMAINS, P.GENERAL)):
        return {d: round(P.eval_loss(model, data[d]["eval"][:200], "cuda"), 4) for d in doms}

    m = load(attention_dropout=0.0)
    P.set_trainable(m, 12)
    emit("preentrenado", losses(m))
    del m
    torch.cuda.empty_cache()

    def ids(dom, n, eval_=False):
        return torch.as_tensor(data[dom]["eval" if eval_ else "train"][:n].astype("int64"), device="cuda")

    for mix in (1.0, 0.1, 0.0):
        m = load(attention_dropout=0.0)
        layers = P.set_trainable(m, 12)
        div = Divider(signal="duda", z=-1e9, max_experts=48, every=1, layers=tuple(layers), mix=mix)
        div.attach(m)
        opt, x = P._adamw(m, 0.0), ids("fines", 8)
        for _ in range(16):
            with torch.no_grad(), P.amp("cuda"):
                logits = m(input_ids=x).logits
            div.observe(m, x, logits, x)
            div.check(m, opt, 0)
            del logits
        with torch.no_grad(), P.amp("cuda"):
            m(input_ids=ids("fines", 8, eval_=True))
        uso = [round(float((routers(m)[i].last[2] >= 32).float().mean()), 4) for i in layers]
        for r in routers(m):
            r.record = False
        emit(f"division_estatica_mix_{mix}", {"perdida": losses(m, ("fines", P.GENERAL)), "uso_hijos_fines": uso,
                                               "expertos": routers(m)[layers[0]].num_experts})  # fmt: skip
        del m, div, opt
        torch.cuda.empty_cache()

    def run(name, cond, **cfg):
        torch.cuda.reset_peak_memory_stats()
        sub = {d: {"train": v["train"][:1600], "eval": v["eval"][:200]} for d, v in data.items()}
        t0 = time.time()
        r = P.run2(cond, 0, load(attention_dropout=0.0), sub, "cuda", P.Cfg2(domains=DOMS, batch=64, warmup=5, **cfg),
                   log=lambda x: print(json.dumps(x), flush=True))  # fmt: skip
        z = r["z_max_revisiones"] or []
        emit(name, {"L_inicial": r["L_inicial"], "L": r["L"], "aprendido": r["aprendido"], "olvido": r["olvido"],
                    "expertos_finales": r["expertos_finales"], "divisiones": len(r["eventos_division"] or []),
                    "z_max_por_revision": [max(c["z"] for c in z if c["paso"] == s) for s in sorted({c["paso"] for c in z})],
                    "uso_expertos_nuevos": r["uso_expertos_nuevos"], "segundos": round(time.time() - t0),
                    "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})  # fmt: skip
        torch.cuda.empty_cache()

    fixed = P.Cond2("sin_crecer", grow_at_start=False, divide=False, merge_align=False, freeze_parent=False)
    for lr in (2e-5, 6e-5):
        run(f"sin_crecer_lr_{lr}", fixed, lr=lr)
    for mix in (0.1, 1.0):
        run(f"division_mix_{mix}", P.CONDS2["division+padre_congelado"], lr=2e-5, check_every=5, split_mix=mix)
    print("== fin del piloto 5", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
