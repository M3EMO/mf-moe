"""Condición Lifelong-MoE (Chen et al., ICML 2023), fiel al paper (decisión del usuario, 2026-10-16).

- Crece con calendario manual: un experto por capa al empezar las tareas 2, 4, 6, 8, 10 y 12 (de 6 a 12).
- El experto nuevo copia al más usado de su capa (Net2WiderNet), con ruido chico y la fila del padre.
- Al crecer, los expertos viejos y sus filas del router quedan congelados; entre dos crecimientos sigue
  entrenando el más nuevo, así las tareas impares tienen algún experto ruteado para aprender.
- La pérdida suma λ·KL(maestro ‖ alumno) con λ = 0,1, contra el modelo del final de la tarea anterior.
- Sin repetición ni fusión.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from mf_moe.division import split
from mf_moe.model import PAD, moe, routers

GROW_AT = (1, 3, 5, 7, 9, 11)  # índice de la tarea en el orden: tareas 2, 4, ..., 12
KL_LAMBDA = 0.1


def _freeze_rows(opt, param: torch.nn.Parameter, n: int) -> None:
    """Congela las primeras `n` filas: anula su gradiente y sus momentos en AdamW, así no se mueven."""
    mask = torch.ones(param.shape[0], *[1] * (param.dim() - 1), device=param.device)
    mask[:n] = 0
    param.register_hook(lambda g: g * mask)
    for v in opt.state.get(param, {}).values():
        if torch.is_tensor(v) and v.dim():
            v[:n] = 0


class Lifelong:
    def __init__(self, lam: float = KL_LAMBDA):
        self.lam = lam
        self.teacher = None
        self.frozen: list[int] = []  # filas congeladas por capa

    @torch.no_grad()
    def start_task(self, model, opt, index: int) -> None:
        if index == 0:
            return
        self.teacher = copy.deepcopy(model).eval().requires_grad_(False)
        if index not in GROW_AT:
            return
        self.frozen = []
        for layer, r in enumerate(routers(model)):
            self.frozen.append(r.num_experts)
            split(model, opt, layer, int(r.load.argmax()), row_dir=None)
        self.refreeze(model, opt)

    def refreeze(self, model, opt) -> None:
        """Aplica el congelado guardado en `frozen`; también sirve al retomar desde un checkpoint."""
        for layer, n_old in enumerate(self.frozen):
            experts, router = moe(model, layer)
            for p in (experts.gate_up_proj, experts.down_proj, router.weight):
                _freeze_rows(opt, p, n_old)

    def aux_loss(self, ids: torch.Tensor, out) -> torch.Tensor | float:
        """λ·KL(maestro ‖ alumno) sobre los tokens que no son relleno; 0 en la primera tarea."""
        if self.teacher is None:
            return 0.0
        with torch.no_grad():
            t = F.log_softmax(self.teacher(input_ids=ids).logits.float(), -1)
        s = F.log_softmax(out.logits.float(), -1)
        kl = (t.exp() * (t - s)).sum(-1)
        valid = ids != PAD
        return self.lam * kl[valid].mean()
