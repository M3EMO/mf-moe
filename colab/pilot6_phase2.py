"""Piloto 6 de la fase 2 (ticket 12): hijos y clones silenciosos, reclutados por el balanceo.

El piloto 5 mostró que cualquier hijo que el router elige desde el primer paso daña al modelo preentrenado (16
divisiones por capa: +0,53 nats en general aun como clon puro) y que finés y turco aprenden y olvidan (lr 6e-5:
finés 3,20 → 2,97 y olvida 0,62 al aprender turco). Ahora cada experto nuevo nace con sesgo -1 y el balanceo de
los nacidos lo recluta de a poco. Corre finés y turco (1.600 secuencias cada uno, lote 64, lr 6e-5) con:
1. el control (16 clones silenciosos por capa al inicio);
2. la división con padre congelado, hijo silencioso, `mix` 0 y 0,1, revisión cada 5 pasos y división forzada en
   cada revisión (el piloto 5 no superó z 3,5 en ninguna revisión: z máximo entre 2,5 y 3,1).
La referencia sin crecer es la del piloto 5 (mismos datos: lo comprueba el sha256 de datos.json).
Uso: python pilot6_phase2.py DATOS OUT
"""

import json
import sys
import time
from pathlib import Path

import torch

import mf_moe.phase2 as P
from mf_moe.granite import load
from mf_moe.model import routers

DOMS = ("fines", "turco")
# El pilotito tiene 25 pasos × 64 pasadas por dominio: u recluta en ~1/4 de dominio (1 / (0,25 · 1.600) = 2,5e-3);
# el decay de 0,99 deja que el balanceo se pase de largo ~0,25 en el sesgo.
BAL = {"balance_u": 2.5e-3, "load_decay": 0.99, "silent": True}


def main(data_dir: str, out: str) -> None:
    data = P.load_data(data_dir)
    Path(out).mkdir(parents=True, exist_ok=True)
    res = {}

    def emit(key, value):
        res[key] = value
        (Path(out) / "pilot6.json").write_text(json.dumps(res, ensure_ascii=False))
        print("== " + json.dumps({key: value}), flush=True)

    def run(name, cond, **cfg):
        torch.cuda.reset_peak_memory_stats()
        sub = {d: {"train": v["train"][:1600], "eval": v["eval"][:200]} for d, v in data.items()}
        m, t0 = load(attention_dropout=0.0), time.time()
        r = P.run2(cond, 0, m, sub, "cuda", P.Cfg2(domains=DOMS, batch=64, lr=6e-5, warmup=5, **cfg),
                   log=lambda x: print(json.dumps(x), flush=True))  # fmt: skip
        z = r["z_max_revisiones"] or []
        layers = range(cfg.get("trainable_from", 12), len(m.model.layers))
        emit(name, {"L_inicial": r["L_inicial"], "L": r["L"], "aprendido": r["aprendido"], "olvido": r["olvido"],
                    "expertos_finales": r["expertos_finales"], "divisiones": len(r["eventos_division"] or []),
                    "z_max_por_revision": [max(c["z"] for c in z if c["paso"] == s) for s in sorted({c["paso"] for c in z})],
                    "uso_expertos_nuevos": r["uso_expertos_nuevos"],
                    "sesgo_nacidos": [[round(float(routers(m)[i].bias[routers(m)[i].n_born :].min()), 3),
                                       round(float(routers(m)[i].bias[routers(m)[i].n_born :].max()), 3)]
                                      if routers(m)[i].num_experts > routers(m)[i].n_born else None for i in layers],
                    "segundos": round(time.time() - t0), "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})  # fmt: skip
        del m
        torch.cuda.empty_cache()

    run("control_silencioso", P.CONDS2["control"], grow_row="silencioso", **BAL)
    for mix in (0.0, 0.1):
        run(f"division_silenciosa_mix_{mix}", P.CONDS2["division+padre_congelado"], check_every=5, split_mix=mix, z_split=-1e9, **BAL)
    print("== fin del piloto 6", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
