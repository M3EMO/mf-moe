"""B13 (mf-moe-5tq.27): lenguas con un árbol conocido y medidas de si el ruteo lo recupera.

Datos: Random Hierarchy Model multilingüe (MRHM; Doshi et al., arXiv 2610.07168) generalizado de 2 a 8 lenguas.
Las lenguas son las hojas de un árbol binario de profundidad 3. La tabla de reglas que expande el nivel l de la
gramática pertenece al ancestro de profundidad k(l) = max(0, l - L + 4) del árbol de lenguas: dos lenguas a
distancia d (niveles hasta su ancestro común) comparten todas las tablas salvo las d de abajo.
Dentro de una lengua y un nivel, las M·V s-tuplas son distintas: la gramática no es ambigua.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.stats import spearmanr

L, S, M, V = 4, 2, 4, 16  # niveles, ramas, reglas por símbolo, símbolos por nivel (Doshi: 4, 2, 4, 16)
DEPTH = 3
N_LANG = 2**DEPTH
SEQ = S**L
BOS, SHARED0, OWN0 = 2, 32, 128  # ids de byte: vocabulario compartido 32..47; propio por lengua 128..255


def owner_depth(level: int) -> int:
    return max(0, level - L + DEPTH + 1)


def clade(lang: np.ndarray, depth: int) -> np.ndarray:
    """Ancestro de cada lengua a esa profundidad del árbol de lenguas (0 = raíz, DEPTH = la lengua)."""
    return np.asarray(lang) >> (DEPTH - depth)


def tree_distance() -> np.ndarray:
    """[lenguas, lenguas]: niveles hasta el ancestro común."""
    return np.array([[(a ^ b).bit_length() for b in range(N_LANG)] for a in range(N_LANG)])


def rules(seed: int) -> list[np.ndarray]:
    """Por nivel l: [nodos de profundidad owner_depth(l), V, M, S] con las s-tuplas de cada símbolo."""
    out = []
    for level in range(L):
        tabs = []
        for node in range(2 ** owner_depth(level)):
            rng = random.Random(f"{seed}:{level}:{node}")
            tup = rng.sample(range(V**S), V * M)  # distintas: cada tupla tiene un solo padre
            tabs.append(np.array([[(t // V ** (S - 1 - j)) % V for j in range(S)] for t in tup]).reshape(V, M, S))
        out.append(np.stack(tabs))
    return out


def sample(tabs: list[np.ndarray], lang: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """Oraciones [n, SEQ] de las lenguas `lang`, y los símbolos latentes de cada nivel, [n, S**l] para l < L."""
    syms, latent = rng.integers(V, size=(len(lang), 1)), []
    for level in range(L):
        latent.append(syms)
        node = clade(lang, owner_depth(level))[:, None]
        syms = tabs[level][node, syms, rng.integers(M, size=syms.shape)].reshape(len(lang), -1)
    return syms, latent


def sample_family(tabs: list[np.ndarray], n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """Traducciones (B13b): n oraciones latentes escritas en las N_LANG lenguas, [n, N_LANG, SEQ]. Cada nodo del árbol
    de lenguas elige sus reglas una vez y lo heredan sus hojas: dos lenguas a distancia d comparten los símbolos
    latentes hasta el nivel L - d (como los pares de traducción de Doshi). Latentes: [n, N_LANG, S**l] para l < L."""
    syms, latent = np.repeat(rng.integers(V, size=(n, 1, 1)), N_LANG, 1), []
    for level in range(L):
        latent.append(syms)
        k = owner_depth(level)
        node = clade(np.arange(N_LANG), k)
        r = rng.integers(M, size=(n, 2**k, syms.shape[2]))[:, node]
        syms = tabs[level][node[None, :, None], syms, r].reshape(n, N_LANG, -1)
    return syms, latent


def ids(syms: np.ndarray, lang: np.ndarray, own_vocab: bool) -> np.ndarray:
    """[n, 1 + SEQ] ids de byte con BOS al principio."""
    tok = OWN0 + V * np.asarray(lang)[:, None] + syms if own_vocab else SHARED0 + syms
    return np.concatenate([np.full((len(syms), 1), BOS), tok], 1)


def usage(idx: np.ndarray, groups: np.ndarray, n_groups: int, n_exp: int) -> np.ndarray:
    """[grupos, expertos]: uso de cada experto por grupo; `idx` es [tokens, k]."""
    c = np.zeros((n_groups, n_exp))
    np.add.at(c, (np.repeat(groups, idx.shape[1]), idx.reshape(-1)), 1)
    return c / c.sum(1, keepdims=True)


def jsd_matrix(P: np.ndarray) -> np.ndarray:
    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    n = len(P)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        m = (P[i] + P[j]) / 2
        D[i, j] = D[j, i] = (kl(P[i], m) + kl(P[j], m)) / 2
    return D


def _upper(D: np.ndarray) -> np.ndarray:
    return D[np.triu_indices(len(D), 1)]


def _triplets(D: np.ndarray, T: np.ndarray) -> float:
    """Fracción de tripletas de hojas cuyo par más cercano en D es el par hermano del árbol T (azar = 1/3)."""
    hits = []
    for a, b, c in itertools.combinations(range(len(D)), 3):
        pairs = [(a, b), (a, c), (b, c)]
        true = min(pairs, key=lambda p: T[p])
        hits.append(min(pairs, key=lambda p: D[p]) == true)
    return float(np.mean(hits))


def tree_recovery(D: np.ndarray, n_perm: int = 1000, seed: int = 0) -> dict:
    """Mantel (Spearman) y tripletas de D contra el árbol de lenguas, con un nulo que permuta las hojas."""
    T, rng = tree_distance(), np.random.default_rng(seed)
    perms = [rng.permutation(N_LANG) for _ in range(n_perm)]
    # D constante (todas las lenguas rutean igual): Spearman no está definido; 0 = sin estructura
    r = lambda t: 0.0 if np.ptp(_upper(D)) == 0 else float(spearmanr(_upper(D), _upper(t)).statistic)
    null_r = [r(T[p][:, p]) for p in perms]
    null_t = [_triplets(D, T[p][:, p]) for p in perms[:200]]
    return {"mantel": round(r(T), 4), "mantel_nulo_p95": round(float(np.percentile(null_r, 95)), 4),
            "tripletas": round(_triplets(D, T), 4), "tripletas_nulo_p95": round(float(np.percentile(null_t, 95)), 4)}  # fmt: skip


def kmeans_topk(H: np.ndarray, k: int, top: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """Control de la crítica «Myth» (arXiv 2604.09780): k-means sobre los estados que entran al router; cada estado
    elige sus `top` centroides más cercanos, como el top-k del router. [tokens, top]."""
    rng = np.random.default_rng(seed)
    C = H[rng.choice(len(H), k, replace=False)]
    d2 = lambda C: (C**2).sum(1)[None] - 2 * H @ C.T
    for _ in range(iters):
        a = d2(C).argmin(1)
        C = np.stack([H[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
    return np.argsort(d2(C), 1)[:, :top]


def agreement(A: np.ndarray, B: np.ndarray) -> float:
    """Fracción media de expertos compartidos entre los top-k de tokens alineados, [tokens, k] cada uno."""
    return float((A[:, :, None] == B[:, None, :]).any(-1).mean())


def mknn(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Vecinos mutuos (Doshi et al.): fracción de los k vecinos coseno de cada oración en X que también lo son de su
    traducción en Y. X, Y: [oraciones, d], alineadas por fila."""
    def nn(Z):
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        C = Z @ Z.T
        np.fill_diagonal(C, -np.inf)
        return np.argsort(-C, 1)[:, :k]

    a, b = nn(X), nn(Y)
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))]))


def by_distance(M: np.ndarray) -> list[float]:
    """Media de una matriz [lenguas, lenguas] por distancia en el árbol, d = 1..DEPTH."""
    T = tree_distance()
    return [round(float(M[T == d].mean()), 4) for d in range(1, DEPTH + 1)]
