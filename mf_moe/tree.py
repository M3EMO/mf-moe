"""MoE fractal jerárquico (ticket 13): el router baja por un árbol y una división es una mitosis con partición.

La fase 1 probó un MoE plano: dividir solo agregaba un experto más a la lista que elige el router, y padre e hijo
competían por los mismos tokens. Acá cada capa tiene un árbol:
- El router de la capa elige top-k entre las raíces, con el mismo sesgo de balanceo que `BiasBalancedRouter`.
- Una raíz o un nodo interno tiene una compuerta lineal que elige a uno de sus dos hijos (top-1); las hojas son
  los expertos. El peso del token es el de la raíz: la compuerta pasa su gradiente con un truco de paso directo
  (p / p.detach() vale 1), así una división no cambia la salida.
- Mitosis: una hoja que duda se vuelve nodo interno con dos hijas, la vieja (los mismos pesos) y la nueva (clon
  con ruido). La compuerta reparte los tokens del padre por la mitad a lo largo de la dirección de la duda: los
  tokens dudosos van a la nueva. Con `freeze_old` la hija vieja queda congelada y guarda lo aprendido.
El árbol vive en buffers, así entra al checkpoint; `resize` rearma las formas antes de cargarlo.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mf_moe.division import _grow_param
from mf_moe.lifelong import freeze_rows
from mf_moe.model import moe, routers


class TreeRouter(nn.Module):
    def __init__(self, config, u: float = 1e-3, decay: float = 0.999):
        super().__init__()
        n, h = config.num_experts, config.hidden_size
        self.top_k, self.hidden_dim, self.norm_topk_prob = config.num_experts_per_tok, h, config.norm_topk_prob
        self.u, self.decay, self.record = u, decay, False
        self.weight = nn.Parameter(torch.empty(n, h))  # filas de las raíces
        self.gates = nn.Parameter(torch.zeros(0, 2, h))  # una compuerta por nodo interno
        self.gate_b = nn.Parameter(torch.zeros(0, 2))
        self.register_buffer("bias", torch.zeros(n))  # sesgo de balanceo de las raíces: solo elige
        self.register_buffer("load", torch.full((n,), 1.0 / n))
        self.register_buffer("node_leaf", torch.arange(n))  # nodo -> experto, -1 si es interno
        self.register_buffer("node_gate", torch.full((n,), -1))  # nodo -> compuerta, -1 si es hoja
        self.register_buffer("node_kids", torch.full((n, 2), -1))
        self.register_buffer("frozen", torch.zeros(n, dtype=torch.bool))  # expertos congelados

    @property
    def num_experts(self) -> int:  # hojas: la división y los expertos cuentan hojas
        return int((self.node_leaf >= 0).sum())

    def forward(self, hidden_states: torch.Tensor):
        hs = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hs, self.weight)
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        _, roots = torch.topk(probs + self.bias, self.top_k, dim=-1)
        top, node = probs.gather(-1, roots), roots
        if self.norm_topk_prob:
            top = top / top.sum(dim=-1, keepdim=True)
        scale = torch.ones_like(top)
        while True:
            t, k = (self.node_gate[node] >= 0).nonzero(as_tuple=True)
            if not len(t):
                break
            g = self.node_gate[node[t, k]]
            gp = torch.softmax(torch.einsum("nh,nch->nc", hs[t].float(), self.gates[g].float()) + self.gate_b[g], -1)
            kid = gp.argmax(-1)
            p = gp.gather(-1, kid[:, None]).squeeze(-1)
            scale = scale.index_put((t, k), scale[t, k] * p / p.detach())  # vale 1; lleva el gradiente
            node = node.index_put((t, k), self.node_kids[node[t, k], kid])
        idx = self.node_leaf[node]
        if self.training:
            with torch.no_grad():
                frac = torch.bincount(roots.reshape(-1), minlength=len(self.bias)).float() / roots.numel()
                self.load.mul_(self.decay).add_(frac, alpha=1 - self.decay)
                self.bias.add_(self.u * torch.sign(self.load.mean() - self.load))
        if self.record:
            self.last = (hs.detach(), probs.detach(), idx)
        return logits, (top * scale).to(logits.dtype), idx


def install(model) -> None:
    """Reemplaza el router de cada capa por un `TreeRouter` con las mismas filas: al empezar, cada raíz es una hoja."""
    for layer in model.model.layers:
        old = layer.mlp.gate
        new = TreeRouter(model.config).to(old.weight.device)
        new.weight.data.copy_(old.weight.data)
        layer.mlp.gate = new


def _append(opt, module: nn.Module, name: str, row: torch.Tensor) -> None:
    """Agrega una fila a un parámetro con momentos de AdamW en cero (una compuerta nueva no hereda nada)."""
    old = getattr(module, name)
    new = nn.Parameter(torch.cat([old.data, row.unsqueeze(0).to(old.dtype)]))
    for g in opt.param_groups:
        g["params"] = [new if q is old else q for q in g["params"]]
    st = opt.state.pop(old, None)
    if st:
        opt.state[new] = {k: torch.cat([v, torch.zeros_like(v[:1]) if len(v) else torch.zeros_like(row)[None]])
                          if torch.is_tensor(v) and v.dim() else v for k, v in st.items()}  # fmt: skip
    setattr(module, name, new)


def refreeze(model, opt, layer: int) -> None:
    """Vuelve a congelar los expertos marcados: crecer reemplaza los Parameters y con ellos sus hooks."""
    experts, router = moe(model, layer)
    rows = router.frozen.nonzero().flatten().tolist()
    if rows:
        for p in (experts.gate_up_proj, experts.down_proj):
            freeze_rows(opt, p, rows)


@torch.no_grad()
def split(model, opt, layer: int, parent: int, row_dir: torch.Tensor | None, noise: float = 0.01,
          freeze_old: bool = False, **_) -> int:  # fmt: skip
    """Mitosis de la hoja `parent`: queda como hija vieja y nace una hija nueva; devuelve el índice de la nueva."""
    experts, router = moe(model, layer)
    for name in ("gate_up_proj", "down_proj"):
        w = getattr(experts, name)[parent]
        _grow_param(opt, experts, name, parent, w + noise * w.std() * torch.randn_like(w))
    child = experts.num_experts
    experts.num_experts += 1
    h = router.hidden_dim
    d = row_dir if row_dir is not None and row_dir.norm() > 0 else torch.randn(h, device=router.weight.device)
    d = (d / d.norm()).float()
    # la compuerta parte los tokens del padre por la mitad a lo largo de la duda (último lote registrado)
    tau, kappa = 0.0, 1.0
    if hasattr(router, "last"):
        hs, _, idx = router.last
        mine = hs[(idx == parent).any(-1)].float()
        if len(mine) > 1:
            s = mine @ d
            tau, kappa = float(s.median()), 4.0 / (float(s.std()) + 1e-6)
    _append(opt, router, "gates", torch.stack([torch.zeros(h, device=d.device), kappa * d]))
    _append(opt, router, "gate_b", torch.tensor([0.0, -kappa * tau], device=d.device))
    node = int((router.node_leaf == parent).nonzero()[0])
    n = len(router.node_leaf)
    router.node_leaf = torch.cat([router.node_leaf, torch.tensor([parent, child], device=d.device)])
    router.node_leaf[node] = -1
    router.node_gate = torch.cat([router.node_gate, torch.tensor([-1, -1], device=d.device)])
    router.node_gate[node] = len(router.gates) - 1
    router.node_kids = torch.cat([router.node_kids, torch.full((2, 2), -1, device=d.device)])
    router.node_kids[node] = torch.tensor([n, n + 1], device=d.device)
    router.frozen = torch.cat([router.frozen, torch.tensor([False], device=d.device)])
    if freeze_old:
        router.frozen[parent] = True
    refreeze(model, opt, layer)
    return child


def resize(model, counts: list[int]) -> None:
    """Rearma las formas de un árbol que creció (hojas por capa) para cargar después su checkpoint."""
    scratch = torch.optim.SGD(model.parameters(), lr=0.0)
    for layer, n in enumerate(counts):
        while routers(model)[layer].num_experts < n:
            split(model, scratch, layer, 0, None, noise=0.0)


@torch.no_grad()
def specialization(model, batches: dict[str, torch.Tensor]) -> list[float]:
    """Información mutua normalizada I(hoja; tarea) / H(tarea) por capa, con las hojas que eligen los tokens.
    1 = cada hoja atiende una sola tarea; 0 = el reparto de hojas no dice nada de la tarea."""
    rs = routers(model)
    was = [r.record for r in rs]
    counts = [torch.zeros(len(batches), r.num_experts) for r in rs]
    model.eval()
    for r in rs:
        r.record = True
    for ti, ids in enumerate(batches.values()):
        model(input_ids=ids)
        for c, r in zip(counts, rs):
            c[ti] += torch.bincount(r.last[2].reshape(-1).cpu(), minlength=c.shape[1]).float()
    for r, w in zip(rs, was):
        r.record = w
    model.train()
    out = []
    for c in counts:
        p = c / c.sum()
        pt, pl = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
        mi = float((p * (p / (pt * pl)).log()).nan_to_num().sum())
        out.append(round(mi / math.log(len(batches)), 4))
    return out
