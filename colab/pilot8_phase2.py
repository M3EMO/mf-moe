"""Piloto 8 de la fase 2 (ticket 14, etapa C): el árbol fractal en la GPU.

Corre `arbol+consolidacion` (mitosis con hija de bajo rango, hija vieja congelada, consolidación por uso) con el lr
de los deltas U·V × 1, × 3 y × 10 (V8 de la spec), `control_bajo_rango+consolidacion` × 1 y × 10 (V1: los mismos
parámetros nuevos, ruteo plano) y, como referencia, `control+consolidacion_uso` (etapa A). Configuración del
piloto 7: finés y turco, 1.600 secuencias cada uno, lote 64, lr 6e-5, revisión cada 5 pasos y mitosis forzada en
cada revisión. El múltiplo ganador (menor pérdida aprendida del árbol, no menor olvido) pasa a las 3 semillas.
Referencia del piloto 7: `division+consolidacion_uso` finés aprendido 2,975, olvido 0,271.
Reglas de corte (spec R1-R3): más del 25 % de hojas muertas en alguna capa, una compuerta con más del 95 % a una
hija, coincidencia train/eval menor que 0,99 o bf16/fp32 menor que 0,985 (el piloto 8 midió 0,990 en el control
plano: el bf16 cambia ~1 % de los top-8 en cualquier brazo), o más del doble de tiempo que el piloto 7 (~1.050 s)
frenan las corridas largas del árbol. Cada corrida trae su veredicto en `corte`.
Uso: python pilot8_phase2.py DATOS OUT
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


def cut(r: dict, seconds: float) -> list[str]:
    """Reglas de corte que falla la corrida (vacío = pasa)."""
    rep, out = r["ruteo"], []
    if "error" in rep:
        return [f"reporte: {rep['error']}"]
    if max(rep["hojas_muertas"]) > 0.25:
        out.append("hojas muertas > 25 %")
    if any((f.get("compuerta_reparto_max") or 0) > 0.95 for f in rep["forma"]):
        out.append("compuerta > 95 % a una hija")
    if rep["train_vs_eval"] < 0.99 or rep["bf16_vs_fp32"] < 0.985:
        out.append("inferencia dispar < 0,99")
    if seconds > 2 * 1050:
        out.append("más del doble de tiempo que el piloto 7")
    return out


def main(data_dir: str, out: str, only: str = "") -> None:
    data = P.load_data(data_dir)
    Path(out).mkdir(parents=True, exist_ok=True)
    res = {}

    def emit(key, value):
        res[key] = value
        (Path(out) / "pilot8.json").write_text(json.dumps(res, ensure_ascii=False))
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
                    "uso_expertos_nuevos": r["uso_expertos_nuevos"], "ruteo": r["ruteo"], "corte": cut(r, time.time() - t0),
                    "parametros_entrenables_expertos": r["parametros_entrenables_expertos"], "curva": r["curva"],
                    "sesgo_nacidos": [[round(float(routers(m)[i].bias[routers(m)[i].n_born :].min()), 3),
                                       round(float(routers(m)[i].bias[routers(m)[i].n_born :].max()), 3)]
                                      if len(routers(m)[i].bias) > routers(m)[i].n_born else None for i in layers],
                    "segundos": round(time.time() - t0), "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})  # fmt: skip
        del m
        torch.cuda.empty_cache()

    for name, mult in (("arbol+consolidacion", 1.0), ("arbol+consolidacion", 3.0), ("arbol+consolidacion", 10.0),
                       ("control_bajo_rango+consolidacion", 1.0), ("control_bajo_rango+consolidacion", 10.0),
                       ("control+consolidacion_uso", 1.0)):  # fmt: skip
        if not only or name.startswith(only):  # el piloto 8b repite solo el árbol, después del arreglo de la compuerta
            run(f"{name} x{mult:g}", P.CONDS2[name], check_every=5, split_mix=0.1, z_split=-1e9, lowrank_lr_mult=mult, **BAL)
    print("== fin del piloto 8", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(*sys.argv[1:4])
