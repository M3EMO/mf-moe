"""Diagnóstico de la fase 2 (ticket 12): ¿dónde vive el olvido de Granite?

El piloto 6 mostró que congelar 10 de 42 padres no cambia el olvido del finés. Si el olvido vive fuera de los
expertos (atención o router), los cinco brazos de la fase 2 empatan por construcción. Protocolo, sin crecer, con
la configuración del piloto 5 (lote 64, lr 6e-5, 1.600 secuencias por idioma):
1. Entrena finés (estado θ1) y después turco (estado θ2).
2. Parte de θ2 y devuelve un grupo de parámetros de las capas entrenables a θ1: expertos, router, atención o
   normas; también "todo menos el grupo". Mide la pérdida de finés, turco y general.
3. Mide cuánto se movió cada grupo (norma relativa de θ2 − θ1) y cuánto cambió el ruteo de los tokens de finés
   (fracción del top-8 que se conserva, por capa).
Recuperación de un grupo = (finés(θ2) − finés(θ2 con el grupo de θ1)) / (finés(θ2) − finés(θ1)).
Uso: python diag_phase2.py DATOS OUT
"""

import json
import sys
from pathlib import Path

import torch

import mf_moe.phase2 as P
from mf_moe.granite import load
from mf_moe.model import routers

GROUPS = {"expertos": "block_sparse_moe.experts", "router": "block_sparse_moe.router",
          "atencion": "self_attn", "normas": "layernorm"}  # fmt: skip


def main(data_dir: str, out: str, n: int = 1600, device: str = "cuda") -> None:
    data = P.load_data(data_dir)
    sub = {d: {"train": v["train"][:n], "eval": v["eval"][:200]} for d, v in data.items()}
    Path(out).mkdir(parents=True, exist_ok=True)
    res = {}

    def emit(key, value):
        res[key] = value
        (Path(out) / "diag.json").write_text(json.dumps(res, ensure_ascii=False))
        print("== " + json.dumps({key: value}), flush=True)

    def losses(m):
        return {d: round(P.eval_loss(m, sub[d]["eval"], device), 4) for d in ("fines", "turco", P.GENERAL)}

    cfg = dict(batch=64, lr=6e-5, warmup=5)
    fixed = P.Cond2("sin_crecer")
    m = load(attention_dropout=0.0)
    P.run2(fixed, 0, m, sub, device, P.Cfg2(domains=("fines",), **cfg), log=lambda x: None)
    theta1 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    keys = [k for k, p in m.named_parameters() if p.requires_grad]

    def top8(model):
        for r in routers(model):
            r.record = True
        with torch.no_grad(), P.amp(device):
            model.model(input_ids=torch.as_tensor(sub["fines"]["eval"][:4].astype("int64"), device=device))
        idx = [r.last[2].clone() for r in routers(model)]
        for r in routers(model):
            r.record = False
        return idx

    idx1 = top8(m)
    emit("theta1", losses(m))
    P.run2(fixed, 0, m, sub, device, P.Cfg2(domains=("turco",), **cfg), log=lambda x: None)
    theta2 = {k: v.detach().clone() for k, v in m.state_dict().items()}
    base = losses(m)
    emit("theta2", base)
    idx2 = top8(m)
    keep = [round(float((a.sort(-1).values.unsqueeze(-1) == b.sort(-1).values.unsqueeze(-2)).any(-1).float().mean()), 3)
            for a, b in zip(idx1, idx2)]  # fmt: skip
    emit("ruteo_finés_conservado_por_capa", keep)
    drift = {}
    for g, pat in GROUPS.items():
        ks = [k for k in keys if pat in k]
        num = sum(float((theta2[k].float() - theta1[k].float()).norm() ** 2) for k in ks) ** 0.5
        den = sum(float(theta1[k].float().norm() ** 2) for k in ks) ** 0.5
        drift[g] = {"parametros": sum(theta1[k].numel() for k in ks), "cambio_relativo": round(num / den, 5)}
    emit("deriva_por_grupo", drift)
    gap = base["fines"] - res["theta1"]["fines"]
    for g, pat in GROUPS.items():
        for mode, pick in (("solo", lambda k: pat in k), ("todo_menos", lambda k: pat not in k)):
            m.load_state_dict({**theta2, **{k: theta1[k] for k in keys if pick(k)}})
            L = losses(m)
            emit(f"restaura_{mode}_{g}", L | {"recupera": round((base["fines"] - L["fines"]) / gap, 3) if gap else None})
    m.load_state_dict(theta2)
    print("== fin del diagnóstico", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
