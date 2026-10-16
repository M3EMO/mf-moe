"""Ticket 05 del mapa: olvido de referencia, velocidad real y prueba de capacidad (MoE fijo).

Uso (VM con GPU): python -m mf_moe.reference OUT_DIR [olvido,capacidad12,capacidad6] [CHECKPOINT]
- olvido: MoE fijo de 12 expertos aprende suma y después mult, en los dos escenarios; mide cuánto
  cae la suma. Sin olvido medible, las tareas o la escala cambian.
- capacidadN: MoE fijo de N expertos aprende las 12 tareas juntas hasta 88 % en todas, o hasta que
  salta la alarma. La dificultad sirve si 6 expertos NO llegan y 12 sí.
Cada experimento escribe OUT_DIR/<nombre>.json con configuración, semilla, historia y resultado.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from mf_moe.model import build
from mf_moe.resume import load_state, rng_state, save_state, set_rng_state
from mf_moe.tasks import DEFAULT_DIGITS, LSB_FIRST, STEPS, TASKS, eval_set
from mf_moe.train import accuracy, adamw, scores, train_until

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _save(model, path: Path) -> None:
    """Checkpoint en bf16 (alcanza para diagnosticar y pesa la mitad), escrito de forma atómica: el runner
    lo baja apenas aparece."""
    tmp = path.with_name(path.name + ".tmp")
    save_file({k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()}, str(tmp))
    tmp.replace(path)


def olvido(seed: int = 0) -> dict:
    out = {}
    for named in (True, False):
        model = build(12, seed=seed).to(DEVICE)
        opt, rng = adamw(model), random.Random(seed)
        ev = {t: eval_set(t, named) for t in ("suma", "mult")}
        r1 = train_until(model, opt, ["suma"], ev, named=named, rng=rng, device=DEVICE)
        before = accuracy(model, ev["suma"], DEVICE)
        r2 = train_until(model, opt, ["mult"], ev, named=named, rng=rng, device=DEVICE)
        after = accuracy(model, ev["suma"], DEVICE)
        out["con_nombre" if named else "sin_nombre"] = {
            "suma_antes": before, "suma_despues": after, "olvido": before - after,
            "mult": accuracy(model, ev["mult"], DEVICE),
            "pasos": [r1.steps, r2.steps], "estancada": [r1.stalled, r2.stalled],
            "tok_s": round((r1.tokens + r2.tokens) / max(r1.seconds + r2.seconds, 1e-9)),
            "historia": [r1.history, r2.history],
        }  # fmt: skip
    return out


# Pasos hasta 88 % de cada tarea sola (calibraciones 2 a 5, una semilla; mult de la corrida con bf16).
STEPS_ALONE = {"suma": 1000, "mult": 15000, "inv": 250, "ord": 250, "bin": 11750, "mod": 250, "sumamult": 36000,
               "invsuma": 2250, "binsuma": 2500, "modmult": 15000, "digmult": 1750, "ordsuma": 1750}  # fmt: skip


def capacidad(n_experts: int, seed: int = 0, ckpt_dir: str | None = None, ckpt: str | None = None) -> dict:
    """Las doce tareas juntas; cada una recibe una parte del lote proporcional a sus pasos sola (decisión del
    usuario, 2026-10-15): con partes iguales, sumamult sola obligaba a unas 18 h por corrida.
    Con `ckpt`, guarda cada 10.000 pasos y retoma desde ahí si el archivo existe."""
    model = build(n_experts, seed=seed).to(DEVICE)
    opt, rng, key = adamw(model), random.Random(seed), f"capacidad{n_experts}/semilla{seed}"
    ev = {t: eval_set(t, True) for t in TASKS}
    inner = load_state(ckpt, key, model, opt)
    if inner:
        set_rng_state(rng, inner.pop("rng"))

    def save(state: dict) -> None:
        save_state(ckpt, key, model, opt, {**state, "rng": rng_state(rng)})

    r = train_until(model, opt, list(TASKS), ev, named=True, rng=rng, device=DEVICE, patience=40,
                    weights=STEPS_ALONE, resume=inner, save=save if ckpt else None)  # fmt: skip
    if ckpt_dir:
        _save(model, Path(ckpt_dir, f"capacidad{n_experts}.safetensors"))
    return {"expertos": n_experts, "pasos": r.steps, "estancada": r.stalled,
            "exactitud": r.history[-1]["acc"] if r.history else {},
            "tok_s": round(r.tokens / max(r.seconds, 1e-9)), "historia": r.history}  # fmt: skip


# Corrida 5 de calibración: las dos tareas de módulo con restos parciales. La 4 dejó aprendidas
# mult, sumamult, binsuma y digmult; mod y modmult fallaron en el resto (registro en el paper).
CALIBRACION_5 = [("mod", DEFAULT_DIGITS["mod"]), ("modmult", DEFAULT_DIGITS["modmult"])]


def aprendibilidad(seed: int = 0, variants: list | None = None, ckpt_dir: str | None = None) -> dict:
    """Cada tarea sola, modelo nuevo de 12 expertos: pasos hasta 88 % o alarma (paciencia 40 = 10.000 pasos)."""
    out = {}
    for t, dig in variants or [(t, DEFAULT_DIGITS[t]) for t in TASKS]:
        digits = {**DEFAULT_DIGITS, t: tuple(dig)}
        model = build(12, seed=seed).to(DEVICE)
        ev = {t: eval_set(t, True, digits=digits)}
        r = train_until(model, adamw(model), [t], ev, named=True, rng=random.Random(seed), device=DEVICE,
                        patience=40, digits=digits)  # fmt: skip
        key = f"{t}{'-'.join(map(str, dig))}"
        exact, digit, per_pos = scores(model, ev[t], DEVICE)
        out[key] = {"pasos": r.steps, "estancada": r.stalled, "exactitud": exact, "por_digito": round(digit, 4),
                    "por_posicion": [round(x, 3) for x in per_pos], "digitos": dig,
                    "desde_unidad": t in LSB_FIRST, "paso_intermedio": t in STEPS,
                    "tok_s": round(r.tokens / max(r.seconds, 1e-9))}  # fmt: skip
        if ckpt_dir:
            _save(model, Path(ckpt_dir, f"{key}.safetensors"))
        print(f"== {key}: {out[key]}", flush=True)
    return out


def main(out: str, which: str = "olvido,capacidad12,capacidad6", ckpt: str | None = None) -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    meta = {"device": DEVICE, "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else None,
            "digitos": DEFAULT_DIGITS, "semilla": 0}  # fmt: skip
    for name in which.split(","):
        if name == "olvido":
            res = olvido()
        elif name == "aprendibilidad":
            res = aprendibilidad()
        elif name == "calibracion5":
            res = aprendibilidad(variants=CALIBRACION_5, ckpt_dir=out)
        else:
            res = capacidad(int(name.removeprefix("capacidad")), ckpt_dir=out, ckpt=ckpt)
        Path(out, f"{name}.json").write_text(json.dumps({"meta": meta, name: res}, ensure_ascii=False))
        print(f"== {name} listo", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
