"""Validación de medidores (mf-moe-5tq.20): EMO contra un MoE estándar del mismo tamaño, en CPU.

EMO (arXiv 2605.06663) rutea cada documento dentro de un grupo de expertos: es modular por construcción. Su par
StdMoE tiene la misma arquitectura y los mismos datos; solo cambia el objetivo. Si los medidores de B1 y B2 no
separan a los dos, sus veredictos sobre Granite pierden peso.

Los modelos (1 B activos, 14 B totales, fp32: 54 GB) no entran en 15 GB de RAM. El forward va capa por capa: los
estados de todos los documentos quedan en memoria y cada experto se lee del safetensors (mmap) solo para sus
tokens. No ejecuta el código remoto del repo (modeling_emo.py, leído): pre-norm, RoPE, sin qk-norm; el último de
los 128 expertos es compartido (siempre activo, peso 1) y el top-7 sale de la softmax de los 127 ruteados.
Medidas, sobre los mismos textos para los dos modelos:
- control del forward: pérdida de LM en nats por token (un forward mal armado da una pérdida alta);
- huella por dominio: JSD entre mitades del mismo dominio y de dominios distintos (`vigia/colab/fingerprint.py`),
  y expertos efectivos (exp de la entropía del uso) por documento y por dominio: EMO debería concentrar cada
  documento en menos expertos (control positivo de que la modularidad se ve en la inferencia);
- B1 (pesos): `composition`, `shared_atoms` y `tree_likeness` sobre 128 neuronas por experto ruteado (16.256; con
  las 32.512 de 256 por experto, `shared_atoms` pasa los 5 GB y el proceso muere por memoria);
- B1b (función): las mismas medidas con `activation_signatures` sobre los tokens de 2 documentos por dominio;
- B2 (router): `hierarchical_routing` sobre las filas de los 127 ruteados, k = 7;
- B11 (mf-moe-5tq.22): `tree_likeness` de los estados ocultos que entran al MoE (8192 tokens por capa).
Uso: python colab/emo_analysis.py SALIDA.json MODELO [documentos por dominio] [capas de B1/B2]
"""

import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import GPT2TokenizerFast

from colab.expert_tree_analysis import texts
from mf_moe.blocks import (
    activation_signatures,
    composition,
    hierarchical_routing,
    neuron_signatures,
    shared_atoms,
    tree_likeness,
)

DOMAINS, LEN, B1_NEURONS, B2_TOKENS, B1B_DOCS = ("fines", "turco", "espanol", "web", "codigo"), 256, 128, 8192, 2


class Weights:
    """Tensores de un snapshot de HF por nombre, leídos con mmap."""

    def __init__(self, model_id: str):
        snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*"))[0]
        self.dir, self.cfg = snap, json.load(open(f"{snap}/config.json"))
        index = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
        files = {f: safe_open(f"{snap}/{f}", "pt") for f in set(index.values())}
        self.where = {k: files[f] for k, f in index.items()}

    def __call__(self, name: str) -> torch.Tensor:
        return self.where[name].get_tensor(name).float()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), -1) * sin


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def forward(W: Weights, ids: torch.Tensor, keep: dict[int, torch.Tensor]) -> dict:
    """ids [docs, LEN]. Devuelve uso por documento y capa (docs, capas, ruteados), entradas al MoE de las capas de
    `keep` (índices de tokens a guardar) y pérdida de LM por documento."""
    c = W.cfg
    D, T = ids.shape
    nh, hd, E, eps = c["num_attention_heads"], c["hidden_size"] // c["num_attention_heads"], c["num_experts"], c["rms_norm_eps"]
    routed, k = E - c["num_shared_experts"], c["num_experts_per_tok"] - c["num_shared_experts"]
    inv = 1.0 / c["rope_theta"] ** (torch.arange(0, hd, 2).float() / hd)
    freqs = torch.outer(torch.arange(T).float(), inv)
    cos, sin = torch.cat((freqs, freqs), -1).cos(), torch.cat((freqs, freqs), -1).sin()
    h = W("model.embed_tokens.weight")[ids]  # [D, T, d]
    usage = np.zeros((D, c["num_hidden_layers"], routed), dtype=np.int64)
    kept = {}
    for L in range(c["num_hidden_layers"]):
        t0, p = time.time(), f"model.layers.{L}."
        x = rms(h, W(p + "pre_attention_layernorm.weight"), eps)
        Wq, Wk, Wv = (W(p + f"self_attn.{n}_proj.weight") for n in "qkv")
        a = []
        for i in range(0, D, 16):  # de a 16 documentos: acota la memoria de la atención
            q, kk, v = (F.linear(x[i : i + 16], w).view(-1, T, nh, hd).transpose(1, 2) for w in (Wq, Wk, Wv))
            a.append(F.scaled_dot_product_attention(rotate(q, cos, sin), rotate(kk, cos, sin), v, is_causal=True))
        h = h + F.linear(torch.cat(a).transpose(1, 2).reshape(D, T, -1), W(p + "self_attn.o_proj.weight"))
        x = rms(h, W(p + "pre_feedforward_layernorm.weight"), eps).reshape(D * T, -1)
        if L in keep:
            kept[L] = x[keep[L]].clone()
        logits = F.linear(x, W(p + "mlp.gate.weight"))
        wts, idx = torch.softmax(logits[:, :routed], -1).topk(k, -1)
        for d in range(D):
            usage[d, L] = np.bincount(idx[d * T : (d + 1) * T].flatten().numpy(), minlength=routed)
        out = torch.zeros_like(x)
        for e in range(E):
            if e < routed:
                tok, slot = torch.where(idx == e)
                gain = wts[tok, slot, None]
            else:  # compartido: todos los tokens, peso = softmax de un solo logit = 1
                tok, gain = torch.arange(len(x)), 1.0
            if len(tok):
                xe, pe = x[tok], f"{p}mlp.experts.{e}."
                y = F.silu(F.linear(xe, W(pe + "gate_proj.weight"))) * F.linear(xe, W(pe + "up_proj.weight"))
                out.index_add_(0, tok, F.linear(y, W(pe + "down_proj.weight")) * gain)
        h = h + out.view(D, T, -1)
        print(json.dumps({"capa": L, "segundos": round(time.time() - t0)}), flush=True)
    h = rms(h, W("model.norm.weight"), eps)
    head = W("lm_head.weight")
    loss = [F.cross_entropy(F.linear(h[d, :-1], head), ids[d, 1:]).item() for d in range(D)]
    return {"uso": usage, "entradas": kept, "perdida": loss}


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD en bits entre dos distribuciones: cae en [0, 1]."""
    m = (p + q) / 2
    kl = lambda a: float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))
    return (kl(p) + kl(q)) / 2


def fingerprint(counts: dict[str, np.ndarray], reps: int = 20, seed: int = 0) -> dict:
    """Huella por dominio con mitades de igual tamaño (como `vigia/colab/fingerprint.py`). counts[dominio] es
    (documentos, capas, expertos); la JSD se promedia sobre capas."""
    rng, names = np.random.default_rng(seed), list(counts)
    dist = lambda c: c.sum(0) / c.sum(0).sum(1, keepdims=True)
    mean_jsd = lambda a, b: float(np.mean([jsd(x, y) for x, y in zip(dist(a), dist(b))]))
    within, between = {n: [] for n in names}, {(a, b): [] for i, a in enumerate(names) for b in names[i + 1 :]}
    for _ in range(reps):
        halves = {}
        for n in names:
            i, h = rng.permutation(len(counts[n])), len(counts[n]) // 2
            halves[n] = (counts[n][i[:h]], counts[n][i[h : 2 * h]])
            within[n].append(mean_jsd(*halves[n]))
        for a, b in between:
            between[(a, b)].append(mean_jsd(halves[a][0], halves[b][0]))
    eff = lambda c: float(np.mean([np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in dist(c)]))
    w = {n: round(float(np.mean(v)), 5) for n, v in within.items()}
    bt = {f"{a} | {b}": round(float(np.mean(v)), 5) for (a, b), v in between.items()}
    return {"dentro": w, "entre": bt, "dentro_medio": round(float(np.mean(list(w.values()))), 5),
            "entre_medio": round(float(np.mean(list(bt.values()))), 5),
            "expertos_efectivos_documento": round(float(np.mean([eff(c[i : i + 1]) for c in counts.values() for i in range(len(c))])), 2),
            "expertos_efectivos_dominio": round(float(np.mean([eff(c) for c in counts.values()])), 2)}  # fmt: skip


def main(path: str, model_id: str, docs: int = 32, layers: str = "0,4,8,12,15") -> None:
    t0, Ls, rng = time.time(), [int(x) for x in str(layers).split(",")], np.random.default_rng(0)
    W = Weights(model_id)
    tok = GPT2TokenizerFast.from_pretrained(W.dir)  # la clase de su tokenizer_config; Auto* pide el código remoto
    cache = os.path.join(os.path.dirname(os.path.abspath(path)), f"textos_{2 * docs}.json")
    if not os.path.exists(cache):  # los dos modelos ven los mismos textos
        json.dump({d: texts(d, n_docs=2 * docs) for d in DOMAINS}, open(cache, "w"), ensure_ascii=False)
    corpus = json.load(open(cache))
    # el doble de textos: un tercio de los de turco, español y web no llega a LEN tokens
    full = [[x[:LEN] for x in (tok(t)["input_ids"] for t in corpus[d]) if len(x) >= LEN][:docs] for d in DOMAINS]
    dom = np.repeat(np.arange(len(DOMAINS)), [len(f) for f in full])
    ids = torch.tensor([x for f in full for x in f])
    D = len(ids)
    b2 = np.sort(rng.choice(D * LEN, min(B2_TOKENS, D * LEN), replace=False))
    first = np.concatenate([np.flatnonzero(dom == i)[:B1B_DOCS] for i in range(len(DOMAINS))])
    b1b = np.concatenate([np.arange(d * LEN + 1, (d + 1) * LEN) for d in first])  # sin la posición 0
    keep = {L: torch.from_numpy(np.concatenate([b2, b1b])) for L in Ls}
    fw = forward(W, ids, keep)
    loss = np.array(fw["perdida"])
    res = {"modelo": model_id, "dominios": DOMAINS, "documentos": np.bincount(dom).tolist(), "largo": LEN,
           "perdida_lm": round(float(loss.mean()), 4),
           "perdida_lm_por_dominio": {d: round(float(loss[dom == i].mean()), 4) for i, d in enumerate(DOMAINS)},
           "huella": fingerprint({d: fw["uso"][dom == i] for i, d in enumerate(DOMAINS)}),
           "segundos_forward": round(time.time() - t0), "capas": {}}  # fmt: skip
    np.save(path.replace(".json", "_uso.npy"), fw["uso"])
    print(json.dumps({k: res[k] for k in ("perdida_lm", "huella")}), flush=True)
    c = W.cfg
    routed, k = c["num_experts"] - c["num_shared_experts"], c["num_experts_per_tok"] - c["num_shared_experts"]
    for L in Ls:
        t, p = time.time(), f"model.layers.{L}.mlp."
        H = fw["entradas"][L].numpy()
        sel = np.sort(rng.choice(c["intermediate_size"], min(B1_NEURONS, c["intermediate_size"]), replace=False))
        w_in = np.concatenate([np.stack([W(f"{p}experts.{e}.{n}_proj.weight").numpy()[sel] for e in range(routed)])
                               for n in ("gate", "up")], 1)  # fmt: skip
        w_out = np.stack([W(f"{p}experts.{e}.down_proj.weight").numpy()[:, sel] for e in range(routed)])
        Xw, lab = neuron_signatures(w_in, w_out)
        Hf = H[len(b2) :]
        n = np.linalg.norm(Hf, axis=1)
        Xf, _ = activation_signatures(Hf[n < np.percentile(n, 99)], w_in)
        res["capas"][L] = {"b1": {"composicion": composition(Xw, lab), "atomos_compartidos": shared_atoms(Xw, lab)},
                           "b1b": {"tokens": len(Xf[0]), "composicion": composition(Xf, lab),
                                   "atomos_compartidos": shared_atoms(Xf, lab)},
                           "b2": hierarchical_routing(H[: len(b2)], W(p + "gate.weight")[:routed].numpy(), k),
                           "arbol_pesos": tree_likeness(Xw), "arbol_estados": tree_likeness(H[: len(b2)]),
                           "segundos": round(time.time() - t)}  # fmt: skip
        del fw["entradas"][L], Xw, Xf, w_in, w_out
        print(json.dumps({"capa_medida": L, "segundos": res["capas"][L]["segundos"]}), flush=True)
        json.dump(res, open(path, "w"), ensure_ascii=False)
    res["segundos"] = round(time.time() - t0)
    json.dump(res, open(path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *[int(x) if x.isdigit() else x for x in sys.argv[3:]])
