"""MoE base sobre Qwen2-MoE (ticket 03 del mapa): d = 512, 8 capas, 1 compartido + ruteados finos, top-3.

El router reemplaza la pérdida auxiliar por el balanceo de DeepSeek-V3: un sesgo por experto, sin
gradiente, elige el top-k; los pesos salen del softmax sin sesgo, así la entropía del router conserva
su sentido como sensor de duda. La carga se cuenta con una media móvil larga (`decay`), nunca por lote:
en tareas en fila cada lote trae una sola tarea, y balancear por lote recluta expertos de tareas viejas.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.integrations.moe import ExpertsInterface, grouped_mm_experts_forward
from transformers.models.qwen2_moe import modeling_qwen2_moe as qm

VOCAB = 256  # bytes
PAD = 0
EOS_ID = 10  # "\n"


class _Bf16Weights:
    """Los expertos con los pesos en bf16 solo para multiplicar: el gradiente vuelve a los pesos fp32. Con
    `lowrank` (hojas del árbol fractal, `mf_moe.fractal_tree`) suma los deltas U·V a sus filas."""

    def __init__(self, experts: torch.nn.Module, dtype: torch.dtype | None = torch.bfloat16):
        self._experts, self._dtype = experts, dtype

    def __getattr__(self, name: str):
        value = getattr(self._experts, name)
        if name not in ("gate_up_proj", "down_proj"):
            return value
        value = value.to(self._dtype) if self._dtype else value
        lr = getattr(self._experts, "lowrank", None)
        return lr.add_to(name, value) if lr is not None else value


def _grouped_mm_bf16(self, hidden_states, top_k_index, top_k_weights):
    # grouped_mm ignora autocast y con pesos fp32 multiplica en fp32 (perfil en L4, 2026-10-15: 183 contra 110 ms
    # por paso). Guardar los pesos en bf16 perdería las actualizaciones chicas de AdamW.
    if not torch.is_autocast_enabled(hidden_states.device.type):
        exp = _Bf16Weights(self, None) if getattr(self, "lowrank", None) is not None else self
        return grouped_mm_experts_forward(exp, hidden_states, top_k_index, top_k_weights)
    return grouped_mm_experts_forward(_Bf16Weights(self), hidden_states.to(torch.bfloat16), top_k_index, top_k_weights)


ExpertsInterface.register("grouped_mm_bf16", _grouped_mm_bf16)


class BiasBalancedRouter(qm.Qwen2MoeTopKRouter):
    def __init__(self, config: Qwen2MoeConfig, u: float = 1e-3, decay: float = 0.999):
        super().__init__(config)
        self.u, self.decay = u, decay
        self.record = False  # con True guarda (estados, probs, top-k) del último forward: la división los lee
        self.register_buffer("bias", torch.zeros(self.num_experts))
        self.register_buffer("load", torch.full((self.num_experts,), 1.0 / self.num_experts))

    def forward(self, hidden_states: torch.Tensor):
        hs = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hs, self.weight)
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        _, idx = torch.topk(probs + self.bias, self.top_k, dim=-1)  # el sesgo solo elige
        top = probs.gather(-1, idx)
        if self.record:
            self.last = (hs.detach(), probs.detach(), idx)
        if self.norm_topk_prob:
            top = top / top.sum(dim=-1, keepdim=True)
        if self.training:
            with torch.no_grad():
                frac = torch.bincount(idx.reshape(-1), minlength=self.num_experts).float() / idx.numel()
                self.load.mul_(self.decay).add_(frac, alpha=1 - self.decay)
                # shortcut: actualización por signo (DeepSeek-V3 la define sobre sigmoide); con softmax
                # el ticket de DeepSeek sugiere u·error si el sesgo oscila; revisar en la calibración.
                self.bias.add_(self.u * torch.sign(self.load.mean() - self.load))
        return logits, top.to(logits.dtype), idx


def config(n_experts: int, top_k: int = 3, d: int = 512, layers: int = 8, ctx: int = 512) -> Qwen2MoeConfig:
    return Qwen2MoeConfig(
        vocab_size=VOCAB,
        hidden_size=d,
        num_hidden_layers=layers,
        num_attention_heads=d // 64,
        num_key_value_heads=d // 64,
        intermediate_size=4 * d,
        moe_intermediate_size=d,  # 0,25 de la FFN densa: expertos finos
        shared_expert_intermediate_size=d,
        num_experts=n_experts,
        num_experts_per_tok=top_k,
        decoder_sparse_step=1,
        max_position_embeddings=ctx,
        tie_word_embeddings=True,
        output_router_logits=False,
        router_aux_loss_coef=0.0,
        pad_token_id=PAD,
        eos_token_id=EOS_ID,
        bos_token_id=None,
        experts_implementation="grouped_mm_bf16",
    )


def build(n_experts: int, seed: int = 0, **kw) -> Qwen2MoeForCausalLM:
    torch.manual_seed(seed)
    cfg = config(n_experts, **kw)
    model = Qwen2MoeForCausalLM(cfg)
    for layer in model.model.layers:
        router = BiasBalancedRouter(cfg)
        router.weight.data.copy_(layer.mlp.gate.weight.data)  # el reemplazo hereda la inicialización
        layer.mlp.gate = router
    return model


def moe(model, layer: int):
    """(expertos, router) de la capa: `mlp.gate` en Qwen2-MoE, `block_sparse_moe.router` en Granite."""
    blk = model.model.layers[layer]
    m = blk.mlp if hasattr(blk, "mlp") else blk.block_sparse_moe
    return m.experts, (m.gate if hasattr(m, "gate") else m.router)


def routers(model) -> list:
    return [moe(model, i)[1] for i in range(len(model.model.layers))]
