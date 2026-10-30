"""¿Los expertos están hechos de bloques elementales que se repiten? (B1, mapa de bloques).

Un átomo es una neurona de un experto (Goodfire, "Neural networks are already clean"): su fila de la compuerta
(qué detecta) y su columna de salida (qué escribe). `composition` agrupa las neuronas de todos los expertos de
una capa en un árbol jerárquico (Ward) y lo corta en K = 2, 4, ... grupos. En cada nivel mide:
- radio: distancia media de una neurona al centro de su grupo; si el árbol es autosimilar, log(radio) cae en
  línea recta con log(K) (pendiente constante);
- forma: razón de participación de cada grupo (cuántas direcciones usa); autosimilar = igual en cada nivel;
- mezcla: entropía de los expertos dentro de cada grupo (1 = la familia cruza expertos, 0 = vive en uno solo).
Nulos: una gaussiana con el mismo espectro (radio y forma) y las etiquetas de experto barajadas (mezcla).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import pearsonr, spearmanr

from mf_moe.expert_tree import participation_ratio


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def neuron_signatures(w_in: np.ndarray, w_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w_in [E, 2I, H] (compuerta y up, en ese orden), w_out [E, H, I] → (firmas [E·I, 2H], experto de cada una).

    Cambiar el signo de up y de la columna de salida juntos no cambia la función: el signo se fija con la
    componente más grande de la salida."""
    E, two_i, H = w_in.shape
    gate, down = w_in[:, : two_i // 2].reshape(-1, H), np.transpose(w_out, (0, 2, 1)).reshape(-1, H)
    down = down * np.sign(down[np.arange(len(down)), np.abs(down).argmax(1)])[:, None]
    return np.concatenate([_unit(gate), _unit(down)], 1), np.repeat(np.arange(E), two_i // 2)


def activation_signatures(H: np.ndarray, w_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B1b: firma funcional. H [T, d] son tokens fijos que entran a la capa; w_in [E, 2I, d] (compuerta y up).
    Firma de una neurona = su activación silu(compuerta)·up sobre los T tokens, centrada y unitaria (el coseno
    entre firmas es la correlación). Se evalúa en todos los tokens, no solo en los ruteados: con soportes
    disjuntos, dos expertos nunca se parecerían. El signo se fija como en `neuron_signatures`."""
    E, two_i, d = w_in.shape
    gate, up = (H @ w_in[:, s].reshape(-1, d).T for s in (slice(None, two_i // 2), slice(two_i // 2, None)))
    A = (gate / (1 + np.exp(-gate)) * up).T
    A = A - A.mean(1, keepdims=True)
    A = A * np.sign(A[np.arange(len(A)), np.abs(A).argmax(1)])[:, None]
    return _unit(A), np.repeat(np.arange(E), two_i // 2)


def weight_function_agreement(Xw: np.ndarray, Xf: np.ndarray, n_pairs: int = 20000, seed: int = 0) -> dict:
    """Correlación entre el coseno por pesos y el coseno por función, sobre pares de neuronas al azar."""
    i, j = np.random.default_rng(seed).integers(0, len(Xw), (2, n_pairs))
    i, j = i[i != j], j[i != j]
    cw, cf = [(_unit(X[i]) * _unit(X[j])).sum(1) for X in (Xw, Xf)]
    return {"pares": len(i), "pearson": float(pearsonr(cw, cf)[0]), "spearman": float(spearmanr(cw, cf)[0])}


def _levels(X: np.ndarray, labels: np.ndarray, ks: list[int], n_experts: int) -> dict:
    Z = linkage(X, "ward")
    radius, shape, mix = [], [], []
    for k in ks:
        c = fcluster(Z, k, "maxclust")
        sq, prs, ent = 0.0, [], []
        for g in np.unique(c):
            m = c == g
            sq += ((X[m] - X[m].mean(0)) ** 2).sum()
            if m.sum() >= 8:
                prs.append((m.sum(), participation_ratio(X[m])))
            if m.sum() >= 2 and n_experts > 1:
                p = np.bincount(labels[m], minlength=n_experts) / m.sum()
                p = p[p > 0]
                ent.append((m.sum(), -(p * np.log(p)).sum() / np.log(min(n_experts, m.sum()))))
        radius.append(float(np.sqrt(sq / len(X))))
        shape.append(float(np.average([v for _, v in prs], weights=[w for w, _ in prs])) if prs else float("nan"))
        mix.append(float(np.average([v for _, v in ent], weights=[w for w, _ in ent])) if ent else float("nan"))
    slopes = np.diff(np.log(radius)) / np.diff(np.log(ks))
    return {"radio": radius, "pendientes": slopes.round(4).tolist(),
            "cv_pendientes": float(slopes.std() / max(abs(slopes.mean()), 1e-12)), "forma": shape, "mezcla": mix}  # fmt: skip


def composition(X: np.ndarray, labels: np.ndarray, ks: list[int] | None = None, n_sample: int = 8192,
                dims: int = 128, seed: int = 0) -> dict:  # fmt: skip
    """Árbol de composición de las neuronas y sus nulos (ver el docstring del módulo)."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    X, labels = X[pick] - X[pick].mean(0), labels[pick]
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    P = U[:, :dims] * S[:dims]  # coordenadas PCA: varianzas S²/n en ejes independientes
    ks = ks or [2**i for i in range(1, int(np.log2(len(P) // 4)) + 1)]
    n_experts = int(labels.max()) + 1
    gauss = rng.standard_normal(P.shape) * P.std(0)
    return {"neuronas": len(P), "niveles": ks, "real": _levels(P, labels, ks, n_experts),
            "nulo_gaussiano": _levels(gauss, labels, ks, n_experts),
            "mezcla_barajada": _levels(P, rng.permutation(labels), ks, n_experts)["mezcla"]}  # fmt: skip


def shared_atoms(X: np.ndarray, labels: np.ndarray, n_sample: int = 4096, seed: int = 0) -> dict:
    """¿Hay neuronas casi repetidas entre expertos? Coseno con la vecina más parecida de otro experto, contra el
    mismo cálculo sobre una gaussiana con el mismo espectro."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(X), min(n_sample, len(X)), replace=False)

    def stats(A: np.ndarray) -> dict:
        A = _unit(A)
        C = A[pick] @ A.T
        C[np.arange(len(pick)), pick] = -1.0
        other = np.where(labels[None, :] != labels[pick][:, None], C, -1.0).max(1)
        same = np.where(labels[None, :] == labels[pick][:, None], C, -1.0).max(1)
        return {"coseno_otro_experto_mediana": float(np.median(other)), "coseno_mismo_experto_mediana": float(np.median(same)),
                "frac_vecina_en_otro_experto": float((other > same).mean()),
                "frac_coseno_otro_mayor_0.7": float((other > 0.7).mean())}  # fmt: skip

    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    gauss = (rng.standard_normal((len(X), len(S))) * (S / np.sqrt(len(X)))) @ Vt + X.mean(0)
    return {"real": stats(X), "nulo_gaussiano": stats(gauss)}


def hierarchical_routing(H: np.ndarray, W: np.ndarray, k: int, groups: tuple[int, ...] = (4, 8), n_null: int = 20,
                         seed: int = 0) -> list[dict]:  # fmt: skip
    """B2: ¿el router admite direcciones jerárquicas? H [tokens, d] entra al router W [E, d].

    Ruteo en dos niveles: los expertos se agrupan por sus filas del router (Ward sobre filas unitarias); cada
    grupo se representa con el centro de sus filas; el token elige los m mejores grupos y, dentro de ellos, sus
    k mejores expertos. `recall` = parte del top-k plano que se recupera; `costo` = productos (grupos +
    candidatos) sobre E. El nulo agrupa al azar con los mismos tamaños."""
    rng = np.random.default_rng(seed)
    logits = H @ W.T
    flat = np.argsort(-logits, 1)[:, :k]
    flat_mask = np.zeros_like(logits, dtype=bool)
    np.put_along_axis(flat_mask, flat, True, 1)

    def recall(assign: np.ndarray, G: int, m: int) -> tuple[float, float]:
        C = np.stack([W[assign == g].mean(0) for g in range(G)])
        top_g = np.argsort(-(H @ C.T), 1)[:, :m]
        cand = (assign[None, :, None] == top_g[:, None, :]).any(2)
        masked = np.where(cand, logits, -np.inf)
        pick = np.argsort(-masked, 1)[:, :k]
        ok = np.take_along_axis(flat_mask, pick, 1) & np.isfinite(np.take_along_axis(masked, pick, 1))
        return float(ok.sum(1).mean() / k), float((G + cand.sum(1).mean()) / len(W))

    out = []
    for G in groups:
        assign = fcluster(linkage(_unit(W), "ward"), G, "maxclust") - 1
        G = int(assign.max()) + 1
        for m in range(1, G + 1):
            r, cost = recall(assign, G, m)
            nulls = [recall(rng.permutation(assign), G, m)[0] for _ in range(n_null)]
            out.append({"grupos": G, "m": m, "recall": round(r, 4), "recall_nulo": round(float(np.mean(nulls)), 4),
                        "recall_nulo_p95": round(float(np.percentile(nulls, 95)), 4), "costo": round(cost, 3)})  # fmt: skip
    return out


def tree_likeness(X: np.ndarray, n_quads: int = 20000, seed: int = 0) -> dict:
    """¿El mapa es un árbol o una nube? Condición de 4 puntos de Gromov sobre cuartetos al azar, relativa a cada
    cuarteto: (S_mayor - S_segunda) / S_mayor, con S = sumas de distancias de los 3 apareos. 0 = árbol exacto.
    Relativa al cuarteto y no al diámetro: así las ramas finas pesan igual que las gruesas. Árbol ≠ autosimilar:
    un árbol puede tener ramas de formas distintas en cada nivel. Nulo: gaussiana con el mismo espectro."""
    rng = np.random.default_rng(seed)

    def delta(A: np.ndarray) -> dict:
        q = A[rng.integers(0, len(A), (n_quads, 4))]
        d = lambda a, b: np.linalg.norm(q[:, a] - q[:, b], axis=1)  # noqa: E731
        s = np.sort(np.stack([d(0, 1) + d(2, 3), d(0, 2) + d(1, 3), d(0, 3) + d(1, 2)], 1), 1)
        s = s[s[:, 2] > 0]
        r = (s[:, 2] - s[:, 1]) / s[:, 2]
        return {"delta_rel_medio": float(r.mean()), "delta_rel_mediana": float(np.median(r))}

    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    gauss = (rng.standard_normal((len(X), len(S))) * (S / np.sqrt(len(X)))) @ Vt
    return {"real": delta(Xc), "nulo_gaussiano": delta(gauss)}
