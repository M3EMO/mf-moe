"""División de expertos (ticket 06 del mapa): el disparador, el seguimiento de la señal y el clon.

Señal por experto, acumulada entre revisiones sobre los tokens que el router le manda (top-k):
- "duda": entropía del router sin sesgo (la señal de Vigía; condición principal);
- "perdida": pérdida de los tokens de respuesta que recibe (comparador);
- "ambas": divide solo si las dos superan el umbral.
Cada `every` pasos, en cada capa, el experto con la señal más alta se divide si su z robusto sobre la capa
(mediana y desviación absoluta mediana, Iglewicz y Hoaglin) supera `z` = 3,5; como máximo una división por
capa y revisión, hasta `max_experts` ruteados por capa. El z-score común no sirve con pocos expertos: con 6
su máximo posible es 2,24, así que el corte original de 2 casi nunca dispararía (usuario, 2026-10-16).

El hijo copia los pesos del padre más un ruido chico; su fila del router nace con la dirección media de
los estados ocultos de los tokens de alta duda del padre (idea de DynMoE), con la norma de la fila del
padre para no desbordar los logits; su sesgo de balanceo arranca igual al del padre y la carga se reparte.
AdamW crece en el lugar: el hijo hereda los momentos del padre, así su primer paso no es gigante.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from mf_moe.model import PAD, moe, routers

SIGNALS = ("duda", "perdida", "ambas")


def _grow_param(opt: torch.optim.Optimizer, module: nn.Module, name: str, parent: int, child: torch.Tensor) -> None:
    """Agrega `child` como fila nueva del parámetro `name` (dim 0) y extiende sus momentos en AdamW."""
    old = getattr(module, name)
    new = nn.Parameter(torch.cat([old.data, child.unsqueeze(0).to(old.dtype)]))
    for g in opt.param_groups:
        g["params"] = [new if q is old else q for q in g["params"]]
    st = opt.state.pop(old, None)
    if st:
        opt.state[new] = {k: torch.cat([v, v[parent : parent + 1]]) if torch.is_tensor(v) and v.dim() else v
                          for k, v in st.items()}  # fmt: skip
    setattr(module, name, new)


@torch.no_grad()
def split(model, opt: torch.optim.Optimizer, layer: int, parent: int, row_dir: torch.Tensor | None,
          noise: float = 0.01, mix: float = 1.0, silent: bool = False) -> int:  # fmt: skip
    """Clona el experto `parent` de la capa `layer`; devuelve el índice del hijo.
    La fila del router del hijo mezcla la del padre con `row_dir` (con la norma del padre): `mix=1` es la dirección
    de la duda (fase 1); un `mix` chico deja al hijo cerca del padre y cambia menos el ruteo de un modelo preentrenado.
    Con `silent` el hijo nace con sesgo -1 (la softmax nunca pasa de 1): el router no lo elige hasta que el balanceo
    le sube el sesgo."""
    experts, router = moe(model, layer)
    for name in ("gate_up_proj", "down_proj"):
        w = getattr(experts, name)[parent]
        _grow_param(opt, experts, name, parent, w + noise * w.std() * torch.randn_like(w))
    w = router.weight[parent]
    row = w if row_dir is None or row_dir.norm() == 0 else (1 - mix) * w + mix * row_dir / row_dir.norm() * w.norm()
    _grow_param(opt, router, "weight", parent, row)
    router.bias = torch.cat([router.bias, router.bias[parent : parent + 1] - float(silent)])
    router.load[parent] /= 2
    router.load = torch.cat([router.load, router.load[parent : parent + 1]])
    experts.num_experts += 1
    router.num_experts += 1
    return experts.num_experts - 1


def resize(model, counts: list[int]) -> None:
    """Hace crecer cada capa hasta `counts` expertos, para cargar después el checkpoint de un modelo que creció."""
    scratch = torch.optim.SGD(model.parameters(), lr=0.0)
    for layer, n in enumerate(counts):
        while routers(model)[layer].num_experts < n:
            split(model, scratch, layer, 0, row_dir=None, noise=0.0)


@dataclass
class Divider:
    """Acumula la señal por experto en cada paso y divide en cada revisión. Se usa como `on_step` de train_until."""

    signal: str = "duda"
    z: float = 3.5
    every: int = 500
    max_experts: int = 12
    mix: float = 1.0  # fila del hijo: ver `split`
    silent: bool = False  # el hijo nace sin que el router lo elija: ver `split`
    layers: tuple[int, ...] | None = None  # capas que pueden dividir; None = todas
    events: list[dict] = field(default_factory=list)
    z_max: list[dict] = field(default_factory=list)  # z más alto de cada capa en cada revisión, divida o no
    _acc: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self):
        assert self.signal in SIGNALS, self.signal

    def attach(self, model) -> None:
        for r in routers(model):
            r.record = True

    def _reset(self, model) -> None:
        self._acc = [{k: torch.zeros(r.num_experts, device=r.weight.device) for k in ("ent", "n", "loss", "nl", "nhi")}
                     | {"hi": torch.zeros(r.num_experts, r.weight.shape[1], device=r.weight.device)}
                     for r in routers(model)]  # fmt: skip

    @torch.no_grad()
    def observe(self, model, ids: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        # Una fusión (u otro cambio de fuera) achica capas: la señal acumulada para el tamaño viejo ya no sirve
        # (corrida con recuperación, 2026-10-16: la división buscó un experto que ya no existía).
        if not self._acc or any(len(a["n"]) != r.num_experts for r, a in zip(routers(model), self._acc)):
            self._reset(model)
        valid = (ids != PAD).reshape(-1)
        tok_loss = torch.full(ids.shape, float("nan"), device=ids.device)  # pérdida del token que predice cada posición
        mask = labels[:, 1:] != -100
        ce = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:].clamp(min=0), reduction="none")
        tok_loss[:, :-1] = torch.where(mask, ce, float("nan"))
        tok_loss = tok_loss.reshape(-1)
        for r, a in zip(routers(model), self._acc):
            hs, probs, idx = r.last
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
            hi = valid & (ent > ent[valid].mean())  # alta duda: sobre la media de la capa en este lote
            for k in range(idx.shape[1]):
                e = idx[:, k]
                a["ent"].index_add_(0, e[valid], ent[valid])
                a["n"].index_add_(0, e[valid], torch.ones_like(ent[valid]))
                has = valid & ~tok_loss.isnan()
                a["loss"].index_add_(0, e[has], tok_loss[has])
                a["nl"].index_add_(0, e[has], torch.ones_like(tok_loss[has]))
                a["hi"].index_add_(0, e[hi], hs[hi].float())
                a["nhi"].index_add_(0, e[hi], torch.ones_like(ent[hi]))

    @staticmethod
    def _z(x: torch.Tensor) -> torch.Tensor:
        med = x.median()
        mad = (x - med).abs().median()
        return torch.zeros_like(x) if mad == 0 else 0.6745 * (x - med) / mad

    def check(self, model, opt, step: int) -> None:
        for layer, (r, a) in enumerate(zip(routers(model), self._acc)):
            if r.num_experts >= self.max_experts or (self.layers is not None and layer not in self.layers):
                continue
            doubt, loss = a["ent"] / a["n"].clamp_min(1), a["loss"] / a["nl"].clamp_min(1)
            zd, zl = self._z(doubt), self._z(loss)
            score = {"duda": zd, "perdida": zl, "ambas": torch.minimum(zd, zl)}[self.signal]
            parent = int(score.argmax())
            self.z_max.append({"paso": step, "capa": layer, "z": round(float(score[parent]), 3)})
            if score[parent] > self.z:
                child = split(model, opt, layer, parent, a["hi"][parent] / a["nhi"][parent].clamp_min(1),
                              mix=self.mix, silent=self.silent)  # fmt: skip
                self.events.append({"paso": step, "capa": layer, "padre": parent, "hijo": child,
                                    "z": round(float(score[parent]), 3), "expertos": r.num_experts})  # fmt: skip
        self._reset(model)

    def __call__(self, model, opt, step: int, ids, logits, labels) -> None:
        self.observe(model, ids, logits, labels)
        if step % self.every == 0:
            self.check(model, opt, step)
