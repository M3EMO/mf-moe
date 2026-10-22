"""Análisis del árbol natural de expertos (ticket 13): solo inferencia, en una L4.

Para cada modelo: 32 secuencias de 512 tokens por dominio (12 dominios en 4 familias, con el tokenizer del
modelo), ruteo capturado por documento con `mf_moe.expert_tree.Recorder` y medidas de `analyze`. Ticket 14:
- `fractal` por niveles familia / dominio / documento (F2 de la spec);
- la figura del territorio de cada experto en tres capas (`territory_shape`: dimensiones intrínseca, de
  correlación y lineal, perfil de escalas y nulo gaussiano), y la misma figura para los grupos de expertos de cada
  nivel del dendrograma (1, 2, 4, 8 grupos y los expertos solos): si el perfil se repite en cada nivel, los
  territorios son autosimilares (F7).
Un modelo que no carga (sin soporte nativo en transformers; nunca se corre código remoto) o un dominio sin datos
se saltean y se reportan. Uso: python expert_tree_analysis.py OUT [MODELOS separados por coma]
"""

import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from mf_moe.expert_tree import DOMAINS, FAMILIES, Recorder, analyze, fractal, territory_shape

MODELS = [  # (id, cuantizar a 4 bits)
    ("ibm-granite/granite-3.1-1b-a400m-base", False),
    ("allenai/OLMoE-1B-7B-0924", False),
    ("deepseek-ai/DeepSeek-V2-Lite", True),
]
FW2 = {"fines": "fin_Latn", "estonio": "ekk_Latn", "hungaro": "hun_Latn", "turco": "tur_Latn", "azeri": "azj_Latn",
       "uzbeko": "uzn_Latn", "espanol": "spa_Latn", "portugues": "por_Latn", "italiano": "ita_Latn"}  # fmt: skip
OTHER = {"web": ("HuggingFaceFW/fineweb-edu", "sample-10BT"), "codigo": ("common-pile/stackv2", None),
         "matematica": ("open-web-math/open-web-math", None)}  # fmt: skip
SEQS, LEN, PER_DOC = 32, 512, 2  # a lo sumo 2 secuencias por documento: cada dominio cubre >= 16 documentos
CAP, MIN_PTS = 700, 100  # puntos por territorio (pdist es cuadrático) y mínimo para medirlo


def _sample(X: np.ndarray, rng) -> np.ndarray:
    return X[rng.choice(len(X), CAP, replace=False)] if len(X) > CAP else X


def territories(rec: Recorder, usage: np.ndarray) -> dict:
    """Figura de cada experto y de los grupos de cada nivel del dendrograma de expertos, en las capas guardadas."""
    rng, out = np.random.default_rng(0), {}
    for layer, parts in rec.kept.items():
        hs = torch.cat([h for h, _, _ in parts]).numpy()
        idx = torch.cat([i for _, i, _ in parts]).numpy()
        U = usage[layer]
        E = (U / U.sum(0, keepdims=True).clip(min=1)).T
        live = np.flatnonzero(E.sum(1) > 0)
        Z = linkage(pdist(E[live], "cosine"), "average")
        levels = {}
        for k in (1, 2, 4, 8, len(live)):
            g = np.zeros(rec.n, dtype=int) - 1
            g[live] = fcluster(Z, k, "maxclust") if k > 1 else 1
            shapes = []
            for grp in np.unique(g[g >= 0]):
                mine = np.isin(idx, np.flatnonzero(g == grp)).any(1)
                if mine.sum() >= MIN_PTS:
                    shapes.append(territory_shape(_sample(hs[mine], rng), n_null=1))
            levels[str(k)] = {"grupos_medidos": len(shapes),
                              "perfil_medio": np.round(np.mean([s["perfil"] for s in shapes], 0), 3).tolist() if shapes else None,
                              "correlacion_media": round(float(np.mean([s["correlacion"] for s in shapes])), 3) if shapes else None,
                              "correlacion_nula_media": round(float(np.mean([s["correlacion_nula"] for s in shapes])), 3) if shapes else None,
                              "figuras": shapes if k == len(live) else None}  # fmt: skip
        out[str(layer)] = levels
    return out


def texts(domain: str, n_docs: int = 120, chars: int = 6000) -> list[str]:
    name, cfg = ("HuggingFaceFW/fineweb-2", FW2[domain]) if domain in FW2 else OTHER[domain]
    out = []
    for row in load_dataset(name, cfg, split="train", streaming=True):
        t = row.get("text") or row.get("content") or ""
        if len(t) > 200:
            out.append(t[:chars])
        if len(out) >= n_docs:
            break
    return out


def seqs(tok, docs: list[str]) -> torch.Tensor:
    out = []
    for d in docs:
        ids = tok(d)["input_ids"]
        out += [ids[i : i + LEN] for i in range(0, len(ids) - LEN + 1, LEN)][:PER_DOC]
        if len(out) >= SEQS:
            break
    return torch.tensor(out[:SEQS])


def main(out: str, only: str = "") -> None:
    Path(out).mkdir(parents=True, exist_ok=True)
    corpus, res = {}, {}
    for d in DOMAINS:
        try:
            corpus[d] = texts(d)
        except Exception as e:  # noqa: BLE001 - un dominio sin datos se reporta y se saltea
            print(f"== dominio salteado {d}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    doms = [d for d in DOMAINS if corpus.get(d)]
    print(f"== dominios {doms}", flush=True)
    for mid, q4 in [m for m in MODELS if not only or any(o in m[0] for o in only.split(","))]:
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=False)
            kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)} \
                if q4 else {"dtype": torch.bfloat16}  # fmt: skip
            model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=False, device_map="cuda", **kw).eval()
        except Exception as e:  # noqa: BLE001
            res[mid] = {"salteado": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"== modelo salteado {mid}: {res[mid]['salteado']}", flush=True)
            continue
        n_layers = model.config.num_hidden_layers
        docs_per = SEQS // PER_DOC
        rec = Recorder(model, len(doms) * docs_per, keep_layers=(n_layers // 4, n_layers // 2, 3 * n_layers // 4))
        n_seq = {}
        with torch.no_grad():
            for di, d in enumerate(doms):
                x = seqs(tok, corpus[d])
                n_seq[d] = len(x)
                for j in range(0, len(x), PER_DOC):  # un documento por vez: `usage` queda por documento
                    rec.domain = di * docs_per + j // PER_DOC
                    model(input_ids=x[j : j + PER_DOC].to("cuda"))
        rec.close()
        doc_dom = np.repeat(np.arange(len(doms)), docs_per)
        by_dom = np.stack([rec.usage[:, doc_dom == di].sum(1) for di in range(len(doms))], 1)
        layers = analyze(by_dom, rec.coact, doms)
        fam = np.array([list(FAMILIES).index(next(f for f, ds in FAMILIES.items() if d in ds)) for d in doms])
        frac = fractal(rec.usage, [fam[doc_dom], doc_dom, np.arange(len(doc_dom))], n_null=10)
        res[mid] = {"expertos": rec.n, "top_k": rec.k, "routers": rec.mods, "secuencias": n_seq, "capas": layers,
                    "uso": by_dom.tolist(), "uso_por_documento": rec.usage.tolist(), "fractal": frac,
                    "territorios": territories(rec, by_dom), "segundos": round(time.time() - t0)}  # fmt: skip
        mid_layers = layers[len(layers) // 4 : 3 * len(layers) // 4]
        summary = {k: round(sum(c[k] for c in mid_layers) / len(mid_layers), 4)
                   for k in ("similitud_misma_familia", "similitud_otra_familia", "rand_ajustado_familias",
                             "info_mutua_norm", "cofenetica_expertos")}  # fmt: skip
        res[mid]["capas_del_medio"] = summary
        print(f"== {mid}: {json.dumps(summary)}", flush=True)
        (Path(out) / "arbol.json").write_text(json.dumps(res, ensure_ascii=False))
        del model
        torch.cuda.empty_cache()
    print("== fin del análisis", flush=True)  # el marcador lo escribe el job después del tar


if __name__ == "__main__":
    main(*sys.argv[1:3])
