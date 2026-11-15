"""B27: ¿pesar la pérdida de cada ejemplo por su duda real crea el nivel 2 (certeza contra plasticidad) en Granite?
Prerregistro en el ticket 09 (mf-moe-generalidad). La puerta oráculo usa la pérdida base de la ida: es el techo de cualquier
señal real. Banco, conjuntos, semillas de lotes y dosis son los de B25; solo el brazo ida.
- oraculo: w = 0,25 + 1,5 · rango(duda) / (n − 1), media 1 sobre el banco.
- azar: los mismos pesos, permutados entre hechos.
- uniforme: sin pesos, solo el conjunto 0, para chequear la réplica contra B25.
B27b (main_b): la señal sin etiqueta (probabilidad máxima del primer token, solo con la pregunta) y el oráculo fuerte (pesos 0,02 a 1,98).
B27c (main_c): la sonda ridge sobre el estado del último token de la pregunta (capas 8, 12, 16 y 20) predice la NLL del valor real;
cada conjunto usa una sonda ajustada sin sus hechos.
Uso: python -m colab.b27_puerta SALIDA.json BANCO.json
     python -m colab.b27_puerta b SALIDA.json BANCO.json
     python -m colab.b27_puerta c SALIDA.json BANCO.json CANDIDATOS.json
     python -m colab.b27_puerta l SALIDA.json BANCO.json CANDIDATOS.json   (B28e, ticket 10: la sonda relee el modelo tras cada paso)
"""

import json
import os
import random
import sys

import numpy as np
import torch

from colab import b23_uniedit as b23
from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab.b16_conceptos import DEVICE
from mf_moe.phase2 import amp

LOW, SPAN = 0.25, 1.5
LAYERS, ALPHAS = (8, 12, 16, 20), np.logspace(0, 5, 11)
DOSES_L = (10, 160)  # B28e: sin la dosis 40, para entrar en el tope de unidades


def gate(doubt: np.ndarray, low: float = LOW, span: float = SPAN) -> np.ndarray:
    rank = np.argsort(np.argsort(doubt))
    return low + span * rank / (len(doubt) - 1)


@torch.no_grad()
def own_confidence(model, tok, facts: list[dict]) -> np.ndarray:
    """Probabilidad máxima del siguiente token al final de la pregunta de la ida: no conoce ningún valor."""
    model.eval()
    out = []
    for i in range(0, len(facts), b17.EVAL_BATCH):
        rows = [tok(b24.fwd(x)[0])["input_ids"] for x in facts[i : i + b17.EVAL_BATCH]]
        L = max(map(len, rows))
        ids = torch.tensor([r + [0] * (L - len(r)) for r in rows], device=DEVICE)
        mask = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows], device=DEVICE)
        with amp(DEVICE):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        last = logits[torch.arange(len(rows)), torch.tensor([len(r) - 1 for r in rows])]
        out += torch.softmax(last, -1).max(-1).values.tolist()
    model.train()
    return np.array(out)


@torch.no_grad()
def prompt_states(model, tok, facts: list[dict], layers: tuple = LAYERS) -> np.ndarray:
    """Estado del último token de la pregunta de la ida en cada capa de `layers`, concatenado: no ve el valor."""
    model.eval()
    out = []
    for i in range(0, len(facts), b17.EVAL_BATCH):
        rows = [tok(b24.fwd(x)[0])["input_ids"] for x in facts[i : i + b17.EVAL_BATCH]]
        L = max(map(len, rows))
        ids = torch.tensor([r + [0] * (L - len(r)) for r in rows], device=DEVICE)
        mask = torch.tensor([[1] * len(r) + [0] * (L - len(r)) for r in rows], device=DEVICE)
        with amp(DEVICE):
            hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        last = torch.tensor([len(r) - 1 for r in rows], device=DEVICE)
        out.append(torch.cat([hs[l][torch.arange(len(rows)), last].float() for l in layers], -1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def ridge_fit(X: np.ndarray, y: np.ndarray, alphas: np.ndarray = ALPHAS):
    """Ridge estandarizada con intercepto; alpha por error dejando uno afuera, como RidgeCV de sklearn (forma dual: d >> n).
    Devuelve la predicción como función de estados nuevos, y el alpha elegido."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    A, ym = (X - mu) / sd, y.mean()
    S, U = np.linalg.eigh(A @ A.T)
    S, Uy = np.clip(S, 0, None), U.T @ (y - ym)
    loo = lambda a: np.mean(((y - ym - U @ (S / (S + a) * Uy)) / (1 - (U**2) @ (S / (S + a)) - 1 / len(y))) ** 2)  # 1/n: el intercepto
    a = min(alphas, key=loo)
    beta = A.T @ (U @ (Uy / (S + a)))
    return (lambda X_new: ym + ((X_new - mu) / sd) @ beta), float(a)


def ridge(X: np.ndarray, y: np.ndarray, X_new: np.ndarray, alphas: np.ndarray = ALPHAS, return_alpha: bool = False):
    f, a = ridge_fit(X, y, alphas)
    return (f(X_new), a) if return_alpha else f(X_new)


def cross_fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Predicción de cada grupo >= 0 con una sonda ajustada sin ese grupo; el grupo −1 solo entrena."""
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups[groups >= 0]):
        pred[groups == g] = ridge(X[groups != g], y[groups != g], X[groups == g])
    return pred


def main(path: str, bank_path: str, model=None, tok=None) -> None:
    facts = json.load(open(bank_path, encoding="utf-8"))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    w = gate(base_i[:, 0])
    rand = np.array(random.Random("b27:azar").sample(list(w), len(w)))
    res["config"] = {"n_hechos": len(facts), "conjuntos": b24.SETS, "dosis": list(b24.DOSES), "peso_min": LOW, "peso_rango": SPAN,
                     "lr": b24.b17.LR, "batch": b24.b17.BATCH}  # fmt: skip
    res["banco"], res["pesos"] = facts, {"oraculo": w.round(4).tolist(), "azar": rand.round(4).tolist()}
    res["base"] = {"gana": {"ida": float(b23.wins(base_i).mean()), "vuelta": float(b23.wins(base_v).mean())},
                   "perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}  # fmt: skip
    print(json.dumps({"base": res["base"]["gana"]}), flush=True)
    arms = {"uniforme": ("ida", None, {0}), "oraculo": ("ida", w, None), "azar": ("ida", rand, None)}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms=arms)


def main_b(path: str, bank_path: str, model=None, tok=None) -> None:
    facts = json.load(open(bank_path, encoding="utf-8"))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    conf = own_confidence(model, tok, facts)
    w_sig, w_strong = gate(-conf), gate(base_i[:, 0], 0.02, 1.96)
    rho = float(np.corrcoef(np.argsort(np.argsort(conf)), np.argsort(np.argsort(-base_i[:, 0])))[0, 1])
    res["config"] = {"n_hechos": len(facts), "conjuntos": b24.SETS, "dosis": list(b24.DOSES), "senal": [LOW, SPAN], "fuerte": [0.02, 1.96],
                     "lr": b24.b17.LR, "batch": b24.b17.BATCH}  # fmt: skip
    res["senal"] = {"confianza": conf.round(5).tolist(), "spearman_con_certeza": round(rho, 4)}
    res["pesos"] = {"senal": w_sig.round(4).tolist(), "oraculo_fuerte": w_strong.round(4).tolist()}
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    print(json.dumps({"senal_spearman_con_certeza": res["senal"]["spearman_con_certeza"]}), flush=True)
    arms = {"senal": ("ida", w_sig, None), "oraculo_fuerte": ("ida", w_strong, None)}
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms=arms)


def probe(model, tok, facts: list[dict], cands: list[dict], tag: str = "b25"):
    """La sonda de B27c: predicción cruzada para el banco, la lista de entrenamiento y la sonda ajustada sin cada conjunto de stage2."""
    n, key = len(facts), lambda x: (x["cabeza"], x["propiedad"])
    order = random.Random(f"{tag}:conjuntos").sample(range(n), n)  # los conjuntos de stage2 con el mismo tag
    group = {key(facts[i]): min(j // (n // b24.SETS), b24.SETS) for j, i in enumerate(order)}  # el sobrante es el grupo SETS
    pool = cands + [x for x in facts if key(x) not in {key(c) for c in cands}]
    y = b17.def_losses(model, tok, [b24.fwd(x) for x in pool])
    X, g = prompt_states(model, tok, pool, LAYERS), np.array([group.get(key(x), -1) for x in pool])
    fits = {s: ridge_fit(X[g != s], y[g != s])[0] for s in np.unique(g[g >= 0])}
    where = {key(x): i for i, x in enumerate(pool)}
    p_bank = np.array([fits[group[key(x)]](X[where[key(x)]][None])[0] for x in facts])
    return p_bank, pool, (fits, [group[key(x)] for x in facts])


def main_c(path: str, bank_path: str, cand_path: str, model=None, tok=None) -> None:
    facts, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, cand_path))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    n = len(facts)
    p_bank, pool, _ = probe(model, tok, facts, cands)
    w = gate(p_bank)
    rho = float(np.corrcoef(np.argsort(np.argsort(p_bank)), np.argsort(np.argsort(base_i[:, 0])))[0, 1])
    res["config"] = {"n_hechos": n, "n_sonda": len(pool), "capas": list(LAYERS), "alphas": ALPHAS.tolist(), "conjuntos": b24.SETS,
                     "dosis": list(b24.DOSES), "sonda": [LOW, SPAN], "lr": b24.b17.LR, "batch": b24.b17.BATCH}  # fmt: skip
    res["sonda"] = {"prediccion": p_bank.round(4).tolist(), "spearman_con_perdida_base": round(rho, 4)}
    res["pesos"] = {"sonda": w.round(4).tolist()}
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    print(json.dumps({"sonda_spearman_con_perdida_base": res["sonda"]["spearman_con_perdida_base"]}), flush=True)
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", arms={"sonda": ("ida", w, None)})


def main_l(path: str, bank_path: str, cand_path: str, model=None, tok=None) -> None:
    """B28e, puerta en línea: tras cada paso, la sonda de B27c relee el estado actual de los hechos que se sobrescriben.
    El peso es el rango de esa lectura entre el banco (los demás hechos con su lectura base): en el paso 0 es el peso de B27c."""
    facts, cands = (json.load(open(p, encoding="utf-8")) for p in (bank_path, cand_path))
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    model, tok = b23.load(model, tok, True)
    base_i, base_v = b24.losses(model, tok, facts)
    p_bank, pool, (fits, group) = probe(model, tok, facts, cands)

    def online(over: list[int]) -> np.ndarray:
        p = p_bank.copy()
        p[over] = fits[group[over[0]]](prompt_states(model, tok, [facts[i] for i in over], LAYERS))
        return gate(p)[over]

    rho = float(np.corrcoef(np.argsort(np.argsort(p_bank)), np.argsort(np.argsort(base_i[:, 0])))[0, 1])
    res["config"] = {"n_hechos": len(facts), "n_sonda": len(pool), "capas": list(LAYERS), "conjuntos": b24.SETS, "dosis": list(DOSES_L),
                     "sonda": [LOW, SPAN], "lr": b24.b17.LR, "batch": b24.b17.BATCH, "relectura": "cada paso"}  # fmt: skip
    res["sonda"] = {"prediccion": p_bank.round(4).tolist(), "spearman_con_perdida_base": round(rho, 4)}
    res["pesos"] = {"sonda_linea_paso0": gate(p_bank).round(4).tolist()}
    res["base"] = {"perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    print(json.dumps({"sonda_spearman_con_perdida_base": res["sonda"]["spearman_con_perdida_base"]}), flush=True)
    b24.stage2(model, tok, facts, base_i, base_v, res, path, dose=True, tag="b25", doses=DOSES_L, arms={"sonda_linea": ("ida", online, None)})


if __name__ == "__main__":
    {"b": lambda: main_b(*sys.argv[2:4]), "c": lambda: main_c(*sys.argv[2:5]), "l": lambda: main_l(*sys.argv[2:5])}.get(sys.argv[1], lambda: main(*sys.argv[1:3]))()

