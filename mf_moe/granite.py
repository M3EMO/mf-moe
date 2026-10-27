"""Adaptador de Granite-3.1-1B-A400M para la fase 2 (ticket 12 del mapa): MoE preentrenado.

En transformers 5, los expertos de Granite usan el mismo formato que nuestro MoE de la fase 1 (`gate_up_proj`
y `down_proj` por capa), así que la división, la fusión alineada y el crecimiento de AdamW sirven igual.
El router de Granite se reemplaza por uno que agrega dos buffers y la opción de registrar su señal:
- `bias`: sesgo que solo elige expertos (sobre la softmax), como en la fase 1; en cero, el ruteo es el de Granite.
  Con `u > 0` el router balancea el uso de los expertos nacidos en la corrida; un hijo con sesgo -1 nace sin que el router lo elija
  y entra al ruteo de a poco (piloto 5: un hijo elegido desde el primer paso desplaza a otros expertos y daña).
- `load`: media móvil del uso de cada experto; la fusión la usa para ponderar el promedio.
Con `record=True` guarda (estados, probabilidades, top-k) del último forward, que la división lee.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import GraniteMoeConfig, GraniteMoeForCausalLM
from transformers.models.granitemoe import modeling_granitemoe as gm

from mf_moe.model import moe  # importar mf_moe.model registra los expertos grouped_mm_bf16

MODEL_ID = "ibm-granite/granite-3.1-1b-a400m-base"


class RecordingGraniteRouter(gm.GraniteMoeTopKRouter):
    def __init__(self, config: GraniteMoeConfig, decay: float = 0.999, u: float = 0.0):
        super().__init__(config)
        self.decay, self.record, self.u = decay, False, u
        self.n_born = self.num_experts  # los expertos desde este índice nacieron en la corrida
        self.register_buffer("bias", torch.zeros(self.num_experts))
        self.register_buffer("load", torch.full((self.num_experts,), 1.0 / self.num_experts))

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight).float()
        probs = torch.softmax(logits, dim=-1)
        _, idx = self.restrict(probs, probs + self.bias).topk(self.top_k, dim=-1)  # el sesgo solo elige; en cero, es el top-k de Granite
        weights = torch.softmax(logits.gather(-1, idx), dim=-1).type_as(hidden_states)
        if self.training:
            with torch.no_grad():
                frac = torch.bincount(idx.reshape(-1), minlength=self.num_experts).float() / idx.numel()
                self.load.mul_(self.decay).add_(frac, alpha=1 - self.decay)
                if self.u:  # balanceo sin pérdida auxiliar (fase 1), solo para los nacidos: los de Granite
                    # conservan su ruteo preentrenado
                    err = torch.sign(self.load.mean() - self.load[self.n_born :])
                    self.bias[self.n_born :] += self.u * err
        if self.record:
            self.last = (hidden_states.detach(), probs.detach(), idx)
        return idx, weights, logits

    def restrict(self, probs: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Puntajes de elección del top-k; EMO (`mf_moe.phase2.EmoRouter`) enmascara los expertos fuera del grupo."""
        return scores


def patch_routers(model: GraniteMoeForCausalLM) -> GraniteMoeForCausalLM:
    """Reemplaza el router de cada capa por `RecordingGraniteRouter`, con los mismos pesos."""
    for layer in range(len(model.model.layers)):
        old = moe(model, layer)[1]
        new = RecordingGraniteRouter(model.config).to(device=old.weight.device, dtype=old.weight.dtype)
        new.bias, new.load = new.bias.float(), new.load.float()
        new.weight.data.copy_(old.weight.data)
        model.model.layers[layer].block_sparse_moe.router = new
    return model


def tiny(n_experts: int = 6, top_k: int = 2, seed: int = 0) -> GraniteMoeForCausalLM:
    """Granite diminuto armado desde la config, para tests en CPU sin bajar el modelo."""
    torch.manual_seed(seed)
    cfg = GraniteMoeConfig(vocab_size=256, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=4, num_local_experts=n_experts,
                           num_experts_per_tok=top_k, max_position_embeddings=128)  # fmt: skip
    return patch_routers(GraniteMoeForCausalLM(cfg))


def load(device: str = "cuda", **config) -> GraniteMoeForCausalLM:
    """Pesos en fp32 (en bf16, AdamW pierde las actualizaciones chicas) y expertos que multiplican en bf16.
    `config` reemplaza campos de la config, por ejemplo `attention_dropout` (0,1 en la de Granite)."""
    model = GraniteMoeForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, experts_implementation="grouped_mm_bf16",
                                                  **config).to(device)  # fmt: skip
    return patch_routers(model)
