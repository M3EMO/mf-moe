"""Análisis de B20 (ticket 15): ¿un árbol aprendido de la matriz concepto × entrenamiento predice mejor que un mapa plano?

Cada mapa es una covarianza entre conceptos y todos usan el mismo estimador: se dejan afuera todos los conjuntos de un
tema, se observa la mitad de los conceptos de un conjunto nuevo (protocolo de Jin y Ren, NeurIPS 2025) y el kriging predice
la otra mitad. Si los mapas difieren, es por su estructura. El efecto aditivo se quita antes, porque ya es una base conocida.
Uso: .venv/bin/python -m colab.b20_analisis RESULTADOS.json   |   .venv/bin/python -m colab.b20_analisis --sintetico REPS
"""

import json
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

N_SEEDS, DRAWS, INNER = 300, 10, 3  # la validación sintética muestra que 30 semillas no distinguen árbol de plano
RANKS, TREE_RANKS, NUGGETS, NEIGHBORS = (1, 2, 3, 5, 8, 15), (3, 8, 15), (0.1, 1.0, 10.0), (3, 5, 10)
MIN_EFFECT = 0.02  # fijado con la validación sintética, antes de ver datos
LEARNED = ("arbol", "plano", "knn")  # se reajustan con los conjuntos de entrenamiento
FIXED = ("wordnet", "bloques")  # no dependen de los conjuntos


def residuals(Z: np.ndarray) -> np.ndarray:
    R = Z - Z.mean(1, keepdims=True)
    return R - R.mean(0, keepdims=True)


def psd(K: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.clip(w, 0, None)) @ V.T


def flat_cov(R: np.ndarray, k: int) -> np.ndarray:
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    return (U[:, :k] * s[:k] ** 2) @ U[:, :k].T / R.shape[1]


def tree_cov(R: np.ndarray, k: int | None = None) -> np.ndarray:
    """Promedia la covarianza empírica por nodo del dendrograma: mismos datos que el plano, con forma de árbol.
    Con `k`, agrupa sobre los primeros k factores: los perfiles crudos tienen demasiado ruido para encontrar los grupos."""
    n, S = len(R), R @ R.T / R.shape[1]
    F = R
    if k is not None:
        U, s, _ = np.linalg.svd(R, full_matrices=False)
        F = U[:, :k] * s[:k]
    L = linkage(np.nan_to_num(pdist(F, "correlation"), nan=1.0), "average")
    groups, K = {i: [i] for i in range(n)}, np.zeros((n, n))
    for t, (a, b, *_) in enumerate(L):
        A, B = groups.pop(int(a)), groups.pop(int(b))
        K[np.ix_(A, B)] = K[np.ix_(B, A)] = S[np.ix_(A, B)].mean()
        groups[n + t] = A + B
    np.fill_diagonal(K, np.diag(S))
    return psd(K)


def block_cov(labels: np.ndarray, parents: np.ndarray) -> np.ndarray:
    return psd((labels[:, None] == labels[None]) * 0.5 + (parents[:, None] == parents[None]) * 0.5)


def krige(K: np.ndarray, seeds: np.ndarray, rest: np.ndarray, y: np.ndarray, nugget: float) -> np.ndarray:
    A = K[np.ix_(seeds, seeds)]
    lam = nugget * max(float(np.trace(A)) / len(seeds), 1e-9)
    return K[np.ix_(rest, seeds)] @ np.linalg.solve(A + lam * np.eye(len(seeds)), y[seeds] - y[seeds].mean())


def knn(R: np.ndarray, seeds: np.ndarray, rest: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Promedio ponderado de los k conjuntos más parecidos en las semillas: la variante KNN de Jin y Ren."""
    a = R[seeds] - R[seeds].mean(0)
    b = y[seeds] - y[seeds].mean()
    sim = a.T @ b / (np.linalg.norm(a, axis=0) * np.linalg.norm(b) + 1e-9)
    top = np.argsort(-sim)[:k]
    return R[np.ix_(rest, top)] @ np.clip(sim[top], 0, None)


def rho(x: np.ndarray, y: np.ndarray) -> float:
    return 0.0 if np.ptp(x) == 0 or np.ptp(y) == 0 else float(spearmanr(x, y)[0])


def _predictors(R: np.ndarray, fixed: dict) -> dict:
    """Candidatos de cada mapa: nombre → lista de funciones (semillas, resto, y) → predicción."""
    out = {"arbol": [lambda s, r, y, K=tree_cov(R, min(k, min(R.shape))), g=g: krige(K, s, r, y, g) for k in TREE_RANKS for g in NUGGETS]}
    out["plano"] = [lambda s, r, y, K=flat_cov(R, min(k, min(R.shape))), g=g: krige(K, s, r, y, g) for k in RANKS for g in NUGGETS]
    out["knn"] = [lambda s, r, y, k=k: knn(R, s, r, y, k) for k in NEIGHBORS]
    for name, K in fixed.items():
        out[name] = [lambda s, r, y, K=K, g=g: krige(K, s, r, y, g) for g in NUGGETS]
    return out


def _tune(Z: np.ndarray, fixed: dict, rng) -> dict:
    """Elige hiperparámetros con validación interna, separando conjuntos dentro del entrenamiento."""
    scores: dict = {}
    n = len(Z)
    for fold in np.array_split(rng.permutation(Z.shape[1]), INNER):
        tr = np.setdiff1d(np.arange(Z.shape[1]), fold)
        beta = Z[:, tr].mean(1)
        for name, cands in _predictors(residuals(Z[:, tr]), fixed).items():
            cur = scores.setdefault(name, np.zeros(len(cands)))
            for c in fold:
                seeds = rng.choice(n, N_SEEDS, replace=False)
                rest = np.setdiff1d(np.arange(n), seeds)
                y = Z[:, c] - beta
                for i, f in enumerate(cands):
                    cur[i] += rho(f(seeds, rest, y), y[rest])
    return {k: int(np.argmax(v)) for k, v in scores.items()}


def within(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    vals = [rho(pred[groups == g], y[groups == g]) for g in np.unique(groups) if (groups == g).sum() >= 5]
    return float(np.mean(vals)) if vals else 0.0


def evaluate(Z: np.ndarray, topics: np.ndarray, labels: np.ndarray, parents: np.ndarray, W: np.ndarray, seed: int = 0, draws: int = DRAWS) -> dict:
    """Z: conceptos × conjuntos. `topics`: tema de cada conjunto. `labels`/`parents`: tema y supertema de cada concepto."""
    rng, n = np.random.default_rng(seed), len(Z)
    fixed = {"wordnet": psd(W - W.mean()), "bloques": block_cov(labels, parents)}
    names = (*LEARNED, *FIXED)
    glob = {m: [] for m in names}
    inside = {m: [] for m in names}
    additive, total = [], []
    for t in np.unique(topics):
        tr, te = np.where(topics != t)[0], np.where(topics == t)[0]
        best = _tune(Z[:, tr], fixed, rng)
        cands = _predictors(residuals(Z[:, tr]), fixed)
        beta = Z[:, tr].mean(1)
        g, w, a, o = {m: [] for m in names}, {m: [] for m in names}, [], []
        for c in te:
            y = Z[:, c] - beta
            for _ in range(draws):
                seeds = rng.choice(n, N_SEEDS, replace=False)
                rest = np.setdiff1d(np.arange(n), seeds)
                for m in names:
                    p = cands[m][best[m]](seeds, rest, y)
                    g[m].append(rho(p, y[rest]))
                    w[m].append(within(p, y[rest], labels[rest]))
                    if m == "arbol":
                        o.append(rho(beta[rest] + p, Z[rest, c]))
                a.append(rho(beta[rest], Z[rest, c]))
        for m in names:
            glob[m].append(float(np.mean(g[m])))
            inside[m].append(float(np.mean(w[m])))
        additive.append(float(np.mean(a)))
        total.append(float(np.mean(o)))
    return {"global": {m: np.array(v) for m, v in glob.items()}, "dentro": {m: np.array(v) for m, v in inside.items()},
            "aditivo_total": np.array(additive), "arbol_total": np.array(total)}  # fmt: skip


def ci(x: np.ndarray, n: int = 10000) -> tuple[float, float, float]:
    b = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    lo, hi = np.percentile(b, [5, 95])
    return float(x.mean()), float(lo), float(hi)


def holds(x: np.ndarray, minimum: float) -> bool:
    m, lo, _ = ci(x)
    return bool(lo > 0 and m >= minimum)


def verdict(res: dict) -> dict:
    """Reglas preregistradas: IC90 sobre temas, sin 0, y efecto mínimo."""
    g = res["global"]
    D = {m: g["arbol"] - g[m] for m in ("plano", "knn", "wordnet", "bloques")}
    best_flat = max(("plano", "knn"), key=lambda m: g[m].mean())
    return {"H_estructura": holds(g[best_flat], 0.05), "H_arbol": all(holds(d, MIN_EFFECT) for d in D.values()),
            "D": {m: ci(d) for m, d in D.items()}, "mejor_plano": best_flat}  # fmt: skip


def synthetic(world: str, rng, n_topics: int = 10, per_topic: int = 60, sets: int = 3, signal: float = 0.14) -> tuple:
    """Mundo con las proporciones de B19: nivel 0,57; conjunto 0,04; interacción repetible `signal`; ruido 0,25.
    `oculto` es un árbol verdadero que no coincide con las etiquetas: WordNet y bloques no pueden ayudar."""
    n, m = n_topics * per_topic, n_topics * sets
    labels = np.repeat(np.arange(n_topics), per_topic)
    parents = labels // 2
    idx = np.arange(n)
    levels = [parents, labels, idx // (per_topic // 5), idx // (per_topic // 15)]
    if world in ("arbol", "oculto"):
        S = sum(rng.normal(size=(lv.max() + 1, m))[lv] for lv in levels)
        S = rng.permutation(S) if world == "oculto" else S
    elif world == "bloques":
        S = sum(rng.normal(size=(lv.max() + 1, m))[lv] for lv in levels[:2])
    elif world == "plano":
        S = rng.normal(size=(n, 5)) @ rng.normal(size=(5, m))
    else:
        S = rng.normal(size=(n, m))
    S = S / S.std() * np.sqrt(signal)
    Z = rng.normal(size=(n, 1)) * np.sqrt(0.57) + rng.normal(size=(1, m)) * np.sqrt(0.04) + S + rng.normal(size=(n, m)) * np.sqrt(0.25)
    W = 0.25 * (labels[:, None] == labels[None]) + 0.25 * (parents[:, None] == parents[None]) + 0.5 * np.eye(n)
    return Z, np.repeat(np.arange(n_topics), sets), labels, parents, W


def reliability(R: dict) -> np.ndarray:
    """Por repetición: correlación de los desvíos del conjunto contra su repetición, con otro orden de lotes."""
    keys = [k for k in R if "|r" not in k]
    Z = np.array([R[k]["subida"] for k in keys]).T
    out = []
    for k in (x for x in R if "|r" in x):
        # la base incluye al conjunto: si lo excluye, comparte su error con la repetición e infla la fiabilidad
        base = Z.mean(1)
        a, b = np.array(R[k.split("|r")[0]]["subida"]) - base, np.array(R[k]["subida"]) - base
        out.append(float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1]))
    return np.array(out)


def signal(R: dict) -> float:
    """Fracción de la varianza total que es interacción repetible. B19: ~0,10 por área, ~0,12 sin el sesgo de 5 columnas. Con 0,12 hay potencia; con 0,10 no."""
    Z = np.array([R[k]["subida"] for k in R if "|r" not in k]).T
    I = Z - Z.mean(1, keepdims=True) - Z.mean(0, keepdims=True) + Z.mean()
    return float(I.var() / Z.var() * max(reliability(R).mean(), 0.0))


def from_json(path: str) -> dict:
    """Matriz y etiquetas desde la salida de colab.b20_matriz. Las repeticiones quedan para C2."""
    from nltk.corpus import wordnet as wn

    data = json.load(open(path))
    R, tops = data["areas"], data["temas"]
    keys = [k for k in R if "|r" not in k]
    names = R[keys[0]]["evaluados"]
    area_of = {x: k for k, v in data["evaluacion"].items() for x in v}
    labels = np.array([list(tops).index(area_of[x]) for x in names])
    parents = np.unique([tops[area_of[x]] for x in names], return_inverse=True)[1]
    syn = [wn.synset(x) for x in names]
    W = np.array([[a.wup_similarity(b) or 0.0 for b in syn] for a in syn])
    Z = np.array([R[k]["subida"] for k in keys]).T
    topics = np.array([list(tops).index(k.split("|")[0]) for k in keys])
    return {"Z": Z, "topics": topics, "labels": labels, "parents": parents, "W": W, "C2": reliability(R), "tiempos": [R[k]["segundos"] for k in R]}


def report(res: dict) -> None:
    for scope in ("global", "dentro"):
        for m, v in res[scope].items():
            print(f"{scope:7s} {m:8s} %+.3f [%+.3f; %+.3f]" % ci(v))
    print("aditivo_total %+.3f [%+.3f; %+.3f]" % ci(res["aditivo_total"]))
    print("arbol_total   %+.3f [%+.3f; %+.3f]" % ci(res["arbol_total"]))
    v = verdict(res)
    for m, x in v["D"].items():
        print(f"D arbol−{m:8s} %+.3f [%+.3f; %+.3f]" % x)
    print({k: v[k] for k in ("H_estructura", "H_arbol", "mejor_plano")})


def main(argv: list[str]) -> None:
    if argv[0] == "--sintetico":
        reps = int(argv[1])
        for world in ("oculto", "arbol", "bloques", "plano", "nulo"):
            H, A, Ds = [], [], []
            for rep in range(reps):
                Z, topics, labels, parents, W = synthetic(world, np.random.default_rng(1000 + rep), sets=6)
                v = verdict(evaluate(Z, topics, labels, parents, W, seed=rep, draws=3))
                H.append(v["H_estructura"])
                A.append(v["H_arbol"])
                Ds.append([v["D"][m][0] for m in ("plano", "knn", "wordnet", "bloques")])
                print(world, rep, v["H_estructura"], v["H_arbol"], np.round(Ds[-1], 3).tolist(), flush=True)
            print(f"RESUMEN {world}: H_estructura {np.mean(H):.2f}, H_arbol {np.mean(A):.2f}, D medio {np.mean(Ds, 0).round(3).tolist()}", flush=True)
        return
    d = from_json(argv[0])
    print("C2 %+.3f [%+.3f; %+.3f]" % ci(d["C2"]), "segundos por corrida", int(np.median(d["tiempos"])))
    report(evaluate(d["Z"], d["topics"], d["labels"], d["parents"], d["W"]))


if __name__ == "__main__":
    main(sys.argv[1:])
