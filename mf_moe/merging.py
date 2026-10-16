"""Fusión de expertos (ticket 07 del mapa): al terminar cada tarea, une pares redundantes de cada capa.

Criterio, una condición por variante:
- "salidas": coseno entre las salidas medias de los expertos sobre un set fijo de tokens (estilo HC-SMoE);
- "pesos": coseno entre los pesos aplanados de los expertos;
- "ambas": fusiona solo si los dos criterios pasan el umbral.
Umbral relativo: un par se fusiona si su similitud supera en `z` desvíos a la media de los pares de su capa.
Los pares se toman disjuntos, de mayor a menor similitud, y la capa nunca baja de `min_experts`.

Operador (HC-SMoE): promedio de pesos ponderado por la frecuencia de uso (la carga del router). La fila
del router y el sesgo quedan los del experto más usado, que suma la carga del otro; los momentos de
AdamW se promedian con los mismos pesos, así el experto fusionado sigue entrenando sin saltos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from mf_moe.model import PAD, routers

CRITERIA = ("salidas", "pesos", "ambas")


def _shrink_param(opt, module: nn.Module, name: str, keep: int, drop: int, wk: float, wd: float, mix: bool) -> None:
    """Quita la fila `drop` del parámetro `name`; con `mix`, la fila `keep` pasa a ser el promedio ponderado."""
    old = getattr(module, name)
    rows = [i for i in range(old.shape[0]) if i != drop]

    def fold(v: torch.Tensor) -> torch.Tensor:
        v = v.clone()
        if mix:
            v[keep] = wk * v[keep] + wd * v[drop]
        return v[rows]

    new = nn.Parameter(fold(old.data))
    for g in opt.param_groups:
        g["params"] = [new if q is old else q for q in g["params"]]
    st = opt.state.pop(old, None)
    if st:
        opt.state[new] = {k: fold(v) if torch.is_tensor(v) and v.dim() else v for k, v in st.items()}
    setattr(module, name, new)


@torch.no_grad()
def merge_pair(model, opt, layer: int, a: int, b: int) -> tuple[int, int]:
    """Fusiona los expertos `a` y `b` de la capa `layer`; devuelve (conservado, quitado) en índices previos."""
    mlp = model.model.layers[layer].mlp
    experts, router = mlp.experts, mlp.gate
    keep, drop = (a, b) if router.load[a] >= router.load[b] else (b, a)
    total = float(router.load[keep] + router.load[drop])
    wk = float(router.load[keep]) / total if total > 0 else 0.5
    for name in ("gate_up_proj", "down_proj"):
        _shrink_param(opt, experts, name, keep, drop, wk, 1 - wk, mix=True)
    _shrink_param(opt, router, "weight", keep, drop, 1.0, 0.0, mix=False)  # fila del más usado
    rows = [i for i in range(router.num_experts) if i != drop]
    router.load[keep] = total
    router.bias, router.load = router.bias[rows], router.load[rows]
    experts.num_experts -= 1
    router.num_experts -= 1
    return keep, drop


def _pair_z(sim: torch.Tensor) -> torch.Tensor:
    sd = sim.std()
    return torch.zeros_like(sim) if not torch.isfinite(sd) or sd == 0 else (sim - sim.mean()) / sd


@torch.no_grad()
def _similarities(experts, hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Matrices de coseno entre expertos: salidas medias sobre `hs` y pesos aplanados."""
    outs = []
    for e in range(experts.num_experts):
        gate, up = F.linear(hs, experts.gate_up_proj[e]).chunk(2, dim=-1)
        outs.append(F.linear(experts.act_fn(gate) * up, experts.down_proj[e]).mean(0))
    o = F.normalize(torch.stack(outs).float(), dim=-1)
    w = torch.cat([experts.gate_up_proj.flatten(1), experts.down_proj.flatten(1)], dim=1).float()
    w = F.normalize(w, dim=-1)
    return o @ o.T, w @ w.T


@dataclass
class Merger:
    criterion: str = "salidas"
    z: float = 2.0
    min_experts: int = 6
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        assert self.criterion in CRITERIA, self.criterion

    @torch.no_grad()
    def __call__(self, model, opt, calib_ids: torch.Tensor, task: str = "") -> None:
        """Fusiona usando `calib_ids` (lote fijo de entradas de entrenamiento) para las salidas medias."""
        rs = routers(model)
        was = [r.record for r in rs]
        for r in rs:
            r.record = True
        model.eval()
        model(input_ids=calib_ids)
        model.train()
        valid = (calib_ids != PAD).reshape(-1)
        hidden = [r.last[0][valid].float() for r in rs]
        for r, w in zip(rs, was):
            r.record = w
        for layer, (r, hs) in enumerate(zip(rs, hidden)):
            experts = model.model.layers[layer].mlp.experts
            so, sw = _similarities(experts, hs.to(experts.gate_up_proj.dtype))
            i, j = torch.triu_indices(r.num_experts, r.num_experts, offset=1)
            zo, zw = _pair_z(so[i, j]), _pair_z(sw[i, j])
            score = {"salidas": zo, "pesos": zw, "ambas": torch.minimum(zo, zw)}[self.criterion]
            alive, used = list(range(r.num_experts)), set()  # índices originales todavía presentes
            for p in score.argsort(descending=True).tolist():
                a, b = int(i[p]), int(j[p])
                if score[p] <= self.z or r.num_experts <= self.min_experts:
                    break
                if a in used or b in used:
                    continue
                _, drop = merge_pair(model, opt, layer, alive.index(a), alive.index(b))
                used |= {a, b}
                orig_drop = alive.pop(drop)
                self.events.append({"tarea": task, "capa": layer, "par": (a, b), "quitado": orig_drop,
                                    "z": round(float(score[p]), 3), "expertos": r.num_experts})  # fmt: skip
