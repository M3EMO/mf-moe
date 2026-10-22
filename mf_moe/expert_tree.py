"""¿Existe un árbol natural de expertos? (ticket 13): ruteo de un MoE preentrenado sobre dominios con familias.

`Recorder` engancha los routers de cualquier MoE de transformers y cuenta, por capa, cuántas veces cada dominio
elige cada experto y cuántas veces dos expertos se eligen juntos en un token. `analyze` responde con números:
- similitud entre dominios: coseno de sus perfiles de uso; ¿los idiomas de una familia se parecen más entre sí?
- agrupamiento jerárquico de los dominios: ¿recupera las familias? (índice de Rand ajustado);
- información mutua normalizada experto-dominio (especialización) y correlación cofenética del agrupamiento
  de los expertos (¿la estructura es jerárquica?).
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

FAMILIES = {
    "uralica": ("fines", "estonio", "hungaro"),
    "turquica": ("turco", "azeri", "uzbeko"),
    "romance": ("espanol", "portugues", "italiano"),
    "ingles": ("web", "codigo", "matematica"),
}
DOMAINS = [d for fam in FAMILIES.values() for d in fam]


def moe_shape(config) -> tuple[int, int]:
    """(expertos ruteados, top-k) con los nombres de cada familia de modelos."""
    n = next(getattr(config, k) for k in ("num_experts", "num_local_experts", "n_routed_experts") if getattr(config, k, None))
    return n, config.num_experts_per_tok


class Recorder:
    """Engancha los módulos que devuelven logits o índices del router y acumula uso y coactivación por capa."""

    def __init__(self, model, n_domains: int):
        self.n, self.k = moe_shape(model.config)
        self.domain, self.hooks, self.mods = 0, [], []
        for name, mod in model.named_modules():
            if name.endswith(("mlp.gate", "mlp.router", "block_sparse_moe.router", "block_sparse_moe.gate")):
                self.mods.append(name)
                self.hooks.append(mod.register_forward_hook(self._hook(len(self.mods) - 1)))
        L = len(self.mods)
        self.usage = np.zeros((L, n_domains, self.n), dtype=np.int64)
        self.coact = np.zeros((L, self.n, self.n), dtype=np.float64)

    def _hook(self, layer: int):
        def hook(_, __, out):
            outs = out if isinstance(out, (tuple, list)) else (out,)
            ts = [t for t in outs if torch.is_tensor(t)]
            idx = next((t for t in ts if not t.is_floating_point() and t.shape[-1] == self.k), None)
            if idx is None:  # solo logits: el top-k sale de ellos
                logits = next(t for t in ts if t.is_floating_point() and t.shape[-1] == self.n)
                idx = logits.float().topk(self.k, dim=-1).indices
            idx = idx.reshape(-1, self.k).cpu()
            self.usage[layer, self.domain] += np.bincount(idx.flatten().numpy(), minlength=self.n)
            onehot = torch.zeros(len(idx), self.n).scatter_(1, idx, 1.0)
            self.coact[layer] += (onehot.T @ onehot).numpy()

        return hook

    def close(self) -> None:
        for h in self.hooks:
            h.remove()


def _ari(a: list[int], b: list[int]) -> float:
    """Índice de Rand ajustado entre dos particiones (Hubert y Arabie, 1985)."""
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    ua, ub = sorted(set(a)), sorted(set(b))
    m = np.array([[sum(1 for x, y in zip(a, b) if x == i and y == j) for j in ub] for i in ua])
    s_ij, s_a, s_b = comb(m).sum(), comb(m.sum(1)).sum(), comb(m.sum(0)).sum()
    exp, mx = s_a * s_b / comb(len(a)), (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 0.0


def analyze(usage: np.ndarray, coact: np.ndarray, domains: list[str] = DOMAINS) -> list[dict]:
    """Medidas por capa a partir de `usage` (capas, dominios, expertos) y `coact` (capas, expertos, expertos)."""
    fam_of = {d: f for f, ds in FAMILIES.items() for d in ds}
    labels = [list(FAMILIES).index(fam_of[d]) for d in domains]
    out = []
    for U, C in zip(usage, coact):
        P = U / U.sum(1, keepdims=True).clip(min=1)  # perfil de uso de cada dominio
        S = 1 - squareform(pdist(P, "cosine"))
        pairs = list(itertools.combinations(range(len(domains)), 2))
        within = [S[i, j] for i, j in pairs if labels[i] == labels[j]]
        across = [S[i, j] for i, j in pairs if labels[i] != labels[j]]
        dom_tree = linkage(pdist(P, "cosine"), "average")
        cut = fcluster(dom_tree, len(FAMILIES), "maxclust")
        joint = U / U.sum()
        pd_, pe = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        mi = float(np.nansum(joint * np.log(joint / (pd_ * pe) + 1e-30)))
        E = (U / U.sum(0, keepdims=True).clip(min=1)).T  # perfil de dominios de cada experto
        live = E.sum(1) > 0
        coph = float(cophenet(linkage(pdist(E[live], "cosine"), "average"), pdist(E[live], "cosine"))[0]) \
            if live.sum() > 2 else float("nan")  # fmt: skip
        diag = np.diag(C).clip(min=1)
        jac = C / (diag[:, None] + diag[None, :] - C).clip(min=1)  # coactivación de Jaccard
        out.append({
            "similitud_misma_familia": round(float(np.mean(within)), 4),
            "similitud_otra_familia": round(float(np.mean(across)), 4),
            "rand_ajustado_familias": round(_ari(labels, list(cut)), 4),
            "info_mutua_norm": round(mi / np.log(len(domains)), 4),
            "cofenetica_expertos": round(coph, 4),
            "coactivacion_media": round(float(jac[~np.eye(len(jac), dtype=bool)].mean()), 4),
            "similitud_dominios": np.round(S, 3).tolist(),
        })  # fmt: skip
    return out




def _cond_info(U: np.ndarray, parent: np.ndarray) -> float:
    """I(experto; etiqueta | padre) / H(etiqueta | padre), con U (etiquetas, expertos) en conteos de tokens."""
    N = U.sum()
    mi = h = 0.0
    for g in np.unique(parent):
        V = U[parent == g]
        n = V.sum()
        J = V / n
        pr, pe = J.sum(1), J.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):  # expertos sin uso en el grupo: 0 · log 0 = 0
            mi += n / N * float(np.nansum(J * np.log(J / (pr[:, None] * pe[None]) + 1e-30)))
        h -= n / N * float(np.sum(pr * np.log(pr + 1e-30)))
    return mi / h if h > 0 else 0.0


def fractal(usage: np.ndarray, levels: list[np.ndarray], n_null: int = 20, seed: int = 0) -> list[dict]:
    """¿Los expertos resuelven cada nivel de la jerarquía de los datos tanto como el anterior? (ticket 14).
    `usage` (capas, etiquetas finas, expertos); `levels` mapea cada etiqueta fina a su etiqueta en cada nivel,
    de grueso a fino (p. ej. familia, idioma, documento). Por nivel: I(experto; nivel | nivel de arriba) normalizada,
    menos un nulo multinomial (cada hijo sortea sus tokens con el perfil de su padre: el sesgo de muestra finita).
    `autosimilitud` = exceso mínimo / exceso máximo: 1 = cada nivel se separa tanto como el anterior; 0 = la
    estructura vive en un solo nivel. Probado y descartado: el conteo de cajas sobre el dendrograma de expertos
    (log N contra log 1/h da r2 ~0,9 también en el nulo)."""
    rng = np.random.default_rng(seed)
    out = []
    for U in usage:
        exc, real = [], []
        for j, lab in enumerate(levels):
            par = levels[j - 1] if j else np.zeros_like(lab)
            M = np.zeros((lab.max() + 1, U.shape[1]))
            np.add.at(M, lab, U)
            pm = np.zeros(lab.max() + 1, dtype=int)
            pm[lab] = par
            keep = M.sum(1) > 0
            M, pm = M[keep], pm[keep]
            r = _cond_info(M, pm)
            P = {g: M[pm == g].sum(0) / M[pm == g].sum() for g in np.unique(pm)}
            null = np.mean([_cond_info(np.array([rng.multinomial(int(m.sum()), P[g]) for m, g in zip(M, pm)]), pm)
                            for _ in range(n_null)])  # fmt: skip
            real.append(r)
            exc.append(r - null)
        exc = np.array(exc)
        out.append({"informacion": np.round(real, 4).tolist(), "exceso": np.round(exc, 4).tolist(),
                    "autosimilitud": round(float(exc.min() / exc.max()) if exc.max() > 0 else 0.0, 4)})  # fmt: skip
    return out
