"""Piloto 4 de la fase 2 (ticket 12): ¿el modelo aprende el dominio que entrena, y la división ocurre?

El piloto 3 mostró que 100 pasos de matemática (lote 8, lr 2e-5, dropout 0) suben la pérdida de matemática, y
que ninguna revisión supera z 3,5. Este piloto mide:
1. Si la evaluación representa al entrenamiento: pérdida del preentrenado en la evaluación y en las últimas
   200 secuencias de entrenamiento, que ninguna corrida entrena.
2. Lote y learning rate: 3.200 secuencias de matemática (3,3 M tokens) con lotes de 8, 64 y 128, sin crecer.
3. La división y la fusión alineada en la GPU: umbral -1e9 (divide en cada revisión) y revisión cada 10 pasos;
   guarda el z más alto de cada revisión para calibrar el umbral.
Uso: python pilot4_phase2.py DATOS OUT
"""

import json
import sys
import time
from pathlib import Path

import torch

import mf_moe.phase2 as P
from mf_moe.granite import load


def main(data_dir: str, out: str) -> None:
    data = P.load_data(data_dir)
    Path(out).mkdir(parents=True, exist_ok=True)
    res = {}

    def emit(key, value):
        res[key] = value
        (Path(out) / "pilot4.json").write_text(json.dumps(res, ensure_ascii=False))
        print("== " + json.dumps({key: value}), flush=True)

    def tail(model, dom):
        return round(P.eval_loss(model, data[dom]["train"][-200:], "cuda"), 4)

    m = load(attention_dropout=0.0)
    P.set_trainable(m, 12)
    emit("evaluacion_contra_cola", {d: {"evaluacion": round(P.eval_loss(m, v["eval"][:200], "cuda"), 4),
                                        "cola_entrenamiento": tail(m, d)} for d, v in data.items()})  # fmt: skip
    del m
    torch.cuda.empty_cache()

    def run(name, cond, doms, n, **cfg):
        torch.cuda.reset_peak_memory_stats()
        sub = {d: {"train": v["train"][:n], "eval": v["eval"][:200]} for d, v in data.items()}
        m, t0 = load(attention_dropout=0.0), time.time()
        r = P.run2(cond, 0, m, sub, "cuda", P.Cfg2(domains=doms, **cfg), log=lambda x: print(json.dumps(x), flush=True))
        z = r["z_max_revisiones"] or []
        emit(name, {"L_inicial": r["L_inicial"], "aprendido": r["aprendido"], "olvido": r["olvido"],
                    "cola_despues": {d: tail(m, d) for d in doms}, "expertos_finales": r["expertos_finales"],
                    "divisiones": len(r["eventos_division"] or []), "fusiones": len(r["eventos_fusion"] or []),
                    "z_max_por_revision": [max(c["z"] for c in z if c["paso"] == s) for s in sorted({c["paso"] for c in z})],
                    "uso_expertos_nuevos": r["uso_expertos_nuevos"], "segundos": round(time.time() - t0),
                    "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})  # fmt: skip
        del m
        torch.cuda.empty_cache()

    fixed = P.Cond2("sin_crecer", grow_at_start=False, divide=False, merge_align=False, freeze_parent=False)
    # lr de Adam escala ~ raíz del lote: 8 -> 64 multiplica por 2,8 (2e-5 -> 6e-5)
    for batch, lr, warmup in ((8, 2e-5, 100), (64, 2e-5, 10), (64, 6e-5, 10), (128, 8e-5, 5)):
        run(f"lote_{batch}_lr_{lr}", fixed, ("matematica",), 3200, batch=batch, lr=lr, warmup=warmup)
    for name in ("division+padre_congelado", "division+fusion_alineada"):
        run(name, P.CONDS2[name], ("matematica", "codigo"), 400, check_every=10, z_split=-1e9)
    print("== fin del piloto 4", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
