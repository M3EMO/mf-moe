"""Piloto 3 de la fase 2 (ticket 12): resuelve lo que dejó abierto el piloto 2 antes de encolar las 12 corridas.

1. Daño del crecimiento: pérdida de evaluación del modelo preentrenado (32 expertos) contra el control crecido a
   48 con la fila del router clonada (piloto 2: +0,27 nats en matemática) y con la fila en cero.
2. Dropout: la config de Granite trae attention_dropout 0,1; se mide la pérdida en modo entrenamiento y si 100
   pasos de matemática bajan la pérdida de matemática con dropout 0,1 y con 0.
3. La división y la fusión alineada en la GPU: revisión cada 10 pasos para que ocurran en un piloto corto.
4. Memoria pico del control a 48 expertos con micro-lotes de 2 y de 1.
Uso: python pilot3_phase2.py DATOS OUT
"""

import json
import sys
import time
from pathlib import Path

import torch

import mf_moe.phase2 as P
from mf_moe.granite import load
from mf_moe.model import routers


def main(data_dir: str, out: str) -> None:
    data = P.load_data(data_dir)
    ev = {d: data[d]["eval"][:200] for d in ("matematica", P.GENERAL)}
    Path(out).mkdir(parents=True, exist_ok=True)
    res = {}

    def emit(key, value):
        res[key] = value
        (Path(out) / "pilot3.json").write_text(json.dumps(res, ensure_ascii=False))
        print("== " + json.dumps({key: value}), flush=True)

    def losses(model):
        return {d: round(P.eval_loss(model, x, "cuda"), 4) for d, x in ev.items()}

    # 1 y 2a: sin entrenar
    m = load(attention_dropout=0.0)
    layers = P.set_trainable(m, 12)
    emit("preentrenado", losses(m))
    m.config.attention_dropout = 0.1
    for mod in m.modules():
        if hasattr(mod, "attention_dropout"):
            mod.attention_dropout = 0.1
    m.train()
    with torch.no_grad(), P.amp("cuda"):
        x = torch.as_tensor(ev["matematica"][:8].astype("int64"), device="cuda")
        emit("modo_entrenamiento_dropout_0.1", round(float(m(input_ids=x, labels=x).loss), 4))
    del m
    calib = torch.as_tensor(data[P.GENERAL]["train"][:8].astype("int64"), device="cuda")
    for row in ("clon", "cero"):
        m = load(attention_dropout=0.0)
        P.set_trainable(m, 12)
        P._grow_control(m, P._adamw(m, 0.0), layers, 48, calib, "cuda", row)
        emit(f"crecido_{row}", losses(m))
        del m
        torch.cuda.empty_cache()

    def run(name, cond, doms, n, dropout, **cfg):
        torch.cuda.reset_peak_memory_stats()
        sub = {d: {"train": v["train"][:n], "eval": v["eval"][:200]} for d, v in data.items()}
        t0 = time.time()
        r = P.run2(cond, 0, load(attention_dropout=dropout), sub, "cuda", P.Cfg2(domains=doms, **cfg), log=lambda _: None)
        emit(name, {"L_inicial": r["L_inicial"], "aprendido": r["aprendido"], "olvido": r["olvido"],
                    "expertos_finales": r["expertos_finales"], "divisiones": len(r["eventos_division"] or []),
                    "fusiones": len(r["eventos_fusion"] or []), "segundos": round(time.time() - t0),
                    "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})  # fmt: skip
        torch.cuda.empty_cache()

    # 2b: ¿100 pasos de matemática bajan la pérdida de matemática?
    fixed = P.Cond2("sin_crecer", grow_at_start=False, divide=False, merge_align=False, freeze_parent=False)
    for dropout in (0.1, 0.0):
        run(f"matematica_100_pasos_dropout_{dropout}", fixed, ("matematica",), 800, dropout)
    # 3: la división y la fusión alineada en la GPU
    for name in ("division+padre_congelado", "division+fusion_alineada"):
        run(name, P.CONDS2[name], ("matematica", "codigo"), 400, 0.0, check_every=10)
    # 4: memoria pico a 48 expertos
    for micro in (2, 1):
        run(f"control_cero_micro_{micro}", P.CONDS2["control"], ("matematica",), 160, 0.0, micro=micro, grow_row="cero")
    print("== fin del piloto 3", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
