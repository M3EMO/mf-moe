"""Árbol fractal sobre Granite (ticket 14, etapa C): "la duda asigna, el uso protege" en cada escala.

- Cada experto de Granite es la raíz de un subárbol. El router elige top-k raíces con la fórmula de Granite y
  cada nodo interno tiene una compuerta de dos hijas (top-1, con gradiente por paso directo aditivo: 1 + p − p.detach()
  vale 1, así una mitosis no cambia la salida). Las compuertas tienen su propio sesgo de balanceo (solo elige,
  acotado a ±1), para que una rama no se quede con todos los tokens (colapso del enrutador).
- Mitosis con hoja de bajo rango (`split`): la hoja queda como hija vieja y nace una hija nueva = copia de la
  madre + U·V (rango `RANK`, U en cero). La compuerta parte los tokens de la madre por la mitad a lo largo de la
  duda; los dudosos van a la nueva. La copia base de la hija nueva queda congelada: lo nuevo se aprende solo en
  U·V. Una mitosis sobre una hija de bajo rango materializa su peso efectivo y suma otro U'·V' (nieta).
- `freeze` congela las hojas de la lista (filas base y su U·V) más todas las copias base.
El árbol vive en buffers y parámetros de los módulos, así entra al checkpoint; `resize` rearma las formas.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mf_moe.division import _grow_param
from mf_moe.granite import RecordingGraniteRouter
from mf_moe.lifelong import freeze_rows
from mf_moe.model import moe, routers
from mf_moe.tree import _append

RANK = 16
NAMES = ("gate_up_proj", "down_proj")


class LowRank(nn.Module):
    """Deltas U·V de las hojas de bajo rango de una capa; `rows` dice a qué fila de expertos va cada uno."""

    def __init__(self, experts: nn.Module, rank: int = RANK):
        super().__init__()
        for name in NAMES:
            _, out, inp = getattr(experts, name).shape
            self.register_parameter(f"u_{name}", nn.Parameter(torch.zeros(0, out, rank)))
            self.register_parameter(f"v_{name}", nn.Parameter(torch.zeros(0, rank, inp)))
        self.register_buffer("rows", torch.zeros(0, dtype=torch.long))

    def add_to(self, name: str, w: torch.Tensor) -> torch.Tensor:
        if not len(self.rows):
            return w
        delta = getattr(self, f"u_{name}") @ getattr(self, f"v_{name}")
        return w.index_add(0, self.rows, delta.to(w.dtype))


class GraniteTreeRouter(RecordingGraniteRouter):
    def __init__(self, config, decay: float = 0.999, u: float = 0.0):
        super().__init__(config, decay=decay, u=u)
        n, h = self.num_experts, config.hidden_size
        self.gates = nn.Parameter(torch.zeros(0, 2, h))
        self.gate_b = nn.Parameter(torch.zeros(0, 2))
        self.register_buffer("gate_bias", torch.zeros(0, 2))  # balanceo de cada compuerta: solo elige
        self.register_buffer("gate_load", torch.zeros(0, 2))
        self.register_buffer("node_leaf", torch.arange(n))  # nodo -> hoja (fila de expertos), -1 si es interno
        self.register_buffer("node_gate", torch.full((n,), -1))  # nodo -> compuerta, -1 si es hoja
        self.register_buffer("node_kids", torch.full((n, 2), -1))

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight).float()
        probs = torch.softmax(logits, dim=-1)
        _, roots = (probs + self.bias).topk(self.top_k, dim=-1)
        weights = torch.softmax(logits.gather(-1, roots), dim=-1)  # la fórmula de Granite, sobre las raíces
        node, scale = roots, torch.ones_like(weights)
        while True:
            t, k = (self.node_gate[node] >= 0).nonzero(as_tuple=True)
            if not len(t):
                break
            g = self.node_gate[node[t, k]]
            gp = torch.softmax(torch.einsum("nh,nch->nc", hidden_states[t].float(), self.gates[g].float()) + self.gate_b[g], -1)
            kid = (gp + self.gate_bias[g]).argmax(-1)
            p = gp.gather(-1, kid[:, None]).squeeze(-1)
            # vale 1 y lleva el gradiente. Nunca p / p.detach(): si el sesgo elige una hija con p ~ 0, el gradiente es
            # 1/p y la corrida da NaN (piloto 8, 2026-10-22)
            scale = scale.index_put((t, k), scale[t, k] * (1 + p - p.detach()))
            node = node.index_put((t, k), self.node_kids[node[t, k], kid])
            if self.training and self.u:
                with torch.no_grad():
                    frac = torch.zeros_like(self.gate_load).index_put_((g, kid), torch.ones_like(p), accumulate=True)
                    seen = frac.sum(1, keepdim=True) > 0
                    frac = frac / frac.sum(1, keepdim=True).clamp_min(1)
                    self.gate_load.copy_(torch.where(seen, self.decay * self.gate_load + (1 - self.decay) * frac, self.gate_load))
                    self.gate_bias += torch.where(seen, self.u * torch.sign(0.5 - self.gate_load), 0.0)
                    self.gate_bias.clamp_(-1.0, 1.0)  # p vive en [0, 1]: más allá, el sesgo ya decide solo
        idx = self.node_leaf[node]
        if self.training:
            with torch.no_grad():
                frac = torch.bincount(roots.reshape(-1), minlength=len(self.bias)).float() / roots.numel()
                self.load.mul_(self.decay).add_(frac, alpha=1 - self.decay)  # las raíces son las de Granite: sin sesgo
        if self.record:
            self.last = (hidden_states.detach(), probs.detach(), idx)
        return idx, (weights * scale).type_as(hidden_states), logits


def install_lowrank(model) -> None:
    """Deltas vacíos en cada capa. Van antes de crear el optimizador: crecer los reemplaza dentro de su grupo."""
    model.config._experts_implementation = "grouped_mm_bf16"  # los deltas U·V entran solo por esa implementación
    for i in range(len(model.model.layers)):
        experts, router = moe(model, i)
        experts.lowrank = LowRank(experts).to(router.weight.device)


def install(model) -> None:
    """Router en árbol y deltas vacíos en cada capa; sin mitosis, rutea igual que Granite."""
    install_lowrank(model)
    for i, layer in enumerate(model.model.layers):
        experts, old = moe(model, i)
        new = GraniteTreeRouter(model.config, decay=old.decay, u=old.u).to(old.weight.device)
        new.weight.data.copy_(old.weight.data)
        new.bias.copy_(old.bias)
        new.load.copy_(old.load)
        layer.block_sparse_moe.router = new


@torch.no_grad()
def split(model, opt, layer: int, parent: int, row_dir: torch.Tensor | None, **_) -> int:
    """Mitosis de la hoja `parent`: queda como hija vieja; nace una hija de bajo rango. Devuelve su fila."""
    experts, router = moe(model, layer)
    lr = experts.lowrank
    for name in NAMES:  # la copia base es el peso efectivo de la madre (con su U·V, si lo tiene)
        w = lr.add_to(name, getattr(experts, name).detach())[parent]
        _grow_param(opt, experts, name, parent, w)
    child = experts.num_experts
    experts.num_experts += 1
    to_lowrank(model, opt, layer, [child])
    h = router.weight.shape[1]
    d = row_dir if row_dir is not None and row_dir.norm() > 0 else torch.randn(h, device=router.weight.device)
    d = (d / d.norm()).float()
    tau, kappa = 0.0, 1.0  # la compuerta parte los tokens de la madre por la mitad a lo largo de la duda
    if hasattr(router, "last"):
        hs, _, idx = router.last
        mine = hs[(idx == parent).any(-1)].float()
        if len(mine) > 1:
            s = mine @ d
            tau, kappa = float(s.median()), 4.0 / (float(s.std()) + 1e-6)
    _append(opt, router, "gates", torch.stack([torch.zeros(h, device=d.device), kappa * d]))
    _append(opt, router, "gate_b", torch.tensor([0.0, -kappa * tau], device=d.device))
    dev = router.node_leaf.device
    router.gate_bias = torch.cat([router.gate_bias, torch.zeros(1, 2, device=dev)])
    router.gate_load = torch.cat([router.gate_load, torch.full((1, 2), 0.5, device=dev)])
    node, n = int((router.node_leaf == parent).nonzero()[0]), len(router.node_leaf)
    router.node_leaf = torch.cat([router.node_leaf, torch.tensor([parent, child], device=dev)])
    router.node_leaf[node] = -1
    router.node_gate = torch.cat([router.node_gate, torch.tensor([-1, -1], device=dev)])
    router.node_gate[node] = len(router.gates) - 1
    router.node_kids = torch.cat([router.node_kids, torch.full((2, 2), -1, device=dev)])
    router.node_kids[node] = torch.tensor([n, n + 1], device=dev)
    router.num_experts += 1
    return child


@torch.no_grad()
def to_lowrank(model, opt, layer: int, rows) -> None:
    """Las filas pasan a ser hojas de bajo rango: su copia base queda fija (ver `freeze`) y aprenden en U·V,
    con U en cero (la salida no cambia). El control con los mismos parámetros que el árbol (V1) lo usa con sus clones."""
    experts, _ = moe(model, layer)
    lr = experts.lowrank
    for row in rows:
        for name in NAMES:
            _, out, inp = getattr(experts, name).shape
            _append(opt, lr, f"u_{name}", torch.zeros(out, RANK, device=lr.rows.device))
            _append(opt, lr, f"v_{name}", torch.randn(RANK, inp, device=lr.rows.device) / inp**0.5)
        lr.rows = torch.cat([lr.rows, torch.tensor([row], device=lr.rows.device)])


def leaf_depths(router) -> torch.Tensor:
    """Profundidad de cada hoja (fila de expertos): 0 para un experto de Granite que nunca hizo mitosis."""
    out = torch.zeros(router.num_experts, dtype=torch.long)
    todo = [(i, 0) for i in range(router.weight.shape[0])]
    while todo:
        node, d = todo.pop()
        kids = router.node_kids[node].tolist()
        if kids[0] < 0:
            out[int(router.node_leaf[node])] = d
        else:
            todo += [(k, d + 1) for k in kids]
    return out


def below(max_depth: int):
    """Para `Divider.eligible`: solo hacen mitosis las hojas por encima de `max_depth` (H10: un solo nivel = 1)."""
    return lambda model, layer: (leaf_depths(moe(model, layer)[1]) < max_depth).to(moe(model, layer)[1].weight.device)


def shape(model, layer: int) -> dict:
    """Forma del árbol de la capa (F3): hojas por profundidad. Sin router en árbol, todo está en la profundidad 0."""
    router = moe(model, layer)[1]
    if not hasattr(router, "node_kids"):
        return {"hojas_por_profundidad": [router.num_experts]}
    depth, todo = {}, [(i, 0) for i in range(router.weight.shape[0])]
    while todo:
        node, d = todo.pop()
        kids = router.node_kids[node].tolist()
        if kids[0] < 0:
            depth[d] = depth.get(d, 0) + 1
        else:
            todo += [(k, d + 1) for k in kids]
    load = router.gate_load
    return {"hojas_por_profundidad": [depth.get(d, 0) for d in range(max(depth) + 1)],
            "compuerta_reparto_max": round(float(load.max(1).values.max()), 4) if len(load) else None}  # fmt: skip


def freeze(model, opt, layer: int, leaves) -> None:
    """Congela esas hojas (filas base y sus U·V) y todas las copias base. Vuelve a registrar los hooks: una
    mitosis reemplaza los Parameters."""
    experts, _ = moe(model, layer)
    lr = experts.lowrank
    rows = sorted(set(leaves) | set(lr.rows.tolist()))
    if rows:
        for name in NAMES:
            freeze_rows(opt, getattr(experts, name), rows)
    own = [j for j, r in enumerate(lr.rows.tolist()) if r in set(leaves)]
    if own:
        for name in NAMES:
            for p in (getattr(lr, f"u_{name}"), getattr(lr, f"v_{name}")):
                freeze_rows(opt, p, own)


def resize(model, counts: list[int]) -> None:
    """Rearma las formas de un árbol que creció (hojas por capa) para cargar después su checkpoint."""
    scratch = torch.optim.SGD(model.parameters(), lr=0.0)
    for layer, n in enumerate(counts):
        while routers(model)[layer].num_experts < n:
            split(model, scratch, layer, 0, None)
