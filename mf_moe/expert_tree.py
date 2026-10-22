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

    def __init__(self, model, n_domains: int, keep_layers: tuple[int, ...] = (), keep_frac: float = 0.02, seed: int = 0):
        """`keep_layers`: capas donde además guarda una muestra (`keep_frac` de los tokens) de los estados que entran
        al router, con el experto que eligió cada uno: el territorio de cada experto (ver `territory_shape`)."""
        self.n, self.k = moe_shape(model.config)
        self.domain, self.hooks, self.mods = 0, [], []
        self.keep_layers, self.keep_frac, self.rng = set(keep_layers), keep_frac, torch.Generator().manual_seed(seed)
        self.kept: dict[int, list[tuple[torch.Tensor, torch.Tensor, int]]] = {}
        for name, mod in model.named_modules():
            if name.endswith(("mlp.gate", "mlp.router", "block_sparse_moe.router", "block_sparse_moe.gate")):
                self.mods.append(name)
                self.hooks.append(mod.register_forward_hook(self._hook(len(self.mods) - 1)))
        L = len(self.mods)
        self.usage = np.zeros((L, n_domains, self.n), dtype=np.int64)
        self.coact = np.zeros((L, self.n, self.n), dtype=np.float64)

    def _hook(self, layer: int):
        def hook(_, inp, out):
            outs = out if isinstance(out, (tuple, list)) else (out,)
            ts = [t for t in outs if torch.is_tensor(t)]
            idx = next((t for t in ts if not t.is_floating_point() and t.shape[-1] == self.k), None)
            if idx is None:  # solo logits: el top-k sale de ellos
                logits = next(t for t in ts if t.is_floating_point() and t.shape[-1] == self.n)
                idx = logits.float().topk(self.k, dim=-1).indices
            idx = idx.reshape(-1, self.k).cpu()
            if layer in self.keep_layers:
                hs = inp[0].reshape(len(idx), -1)
                pick = torch.rand(len(idx), generator=self.rng) < self.keep_frac
                self.kept.setdefault(layer, []).append((hs[pick.to(hs.device)].float().cpu(), idx[pick], self.domain))
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


def two_nn(X: np.ndarray) -> float:
    """Dimensión intrínseca por los dos vecinos más cercanos (Facco y otros, 2017): estimador de máxima verosimilitud
    sobre el cociente entre la distancia al segundo y al primer vecino."""
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    r = np.sort(D, 1)[:, :2]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = r[:, 1] / r[:, 0]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(len(mu) / np.log(mu).sum()) if len(mu) else float("nan")


def correlation_dim(X: np.ndarray, lo: float = 0.02, hi: float = 0.3) -> float:
    """Dimensión de correlación (Grassberger y Procaccia, 1983): pendiente de log C(r) contra log r, con C(r) la
    fracción de pares a distancia menor que r, entre los cuantiles `lo` y `hi` de las distancias. Es el conteo de
    cajas para nubes de puntos: no necesita una grilla, que en 1.024 dimensiones sería imposible."""
    d = pdist(X)
    rs = np.quantile(d[d > 0], np.linspace(lo, hi, 10))
    C = np.array([(d < r).mean() for r in rs])
    return float(np.polyfit(np.log(rs), np.log(C), 1)[0])


def participation_ratio(X: np.ndarray) -> float:
    """Dimensión lineal efectiva: (Σλ)² / Σλ² de los autovalores de la covarianza."""
    ev = np.clip(np.linalg.eigvalsh(np.cov(X, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (ev**2).sum())


SCALES = ((0.002, 0.01), (0.01, 0.05), (0.05, 0.25), (0.25, 0.75))  # cuantiles de distancia, de cerca a lejos


def territory_shape(X: np.ndarray, n_null: int = 3, seed: int = 0) -> dict:
    """La "figura" que arma un experto: la nube de estados que el router le manda. Tres dimensiones (intrínseca,
    de correlación y lineal) y las mismas medidas en un nulo gaussiano con la misma media y covarianza. Una nube
    con estructura no lineal (una superficie curva, racimos, un fractal) da dimensiones intrínseca y de correlación
    menores que su nulo; una nube gaussiana da lo mismo que el nulo."""
    X = np.unique(X, axis=0)  # tokens repetidos dan estados idénticos en las capas bajas: distancia 0 no mide nada
    rng = np.random.default_rng(seed)
    nulls = [rng.multivariate_normal(X.mean(0), np.cov(X, rowvar=False), len(X), method="eigh") for _ in range(n_null)]
    out = {"puntos": len(X), "intrinseca": two_nn(X), "correlacion": correlation_dim(X), "lineal": participation_ratio(X),
           # perfil de escalas: la dimensión de correlación de cerca a lejos. Un fractal da un perfil plano en un valor
           # no entero; racimos dan un perfil que sube en las escalas chicas; una curva con ruido, alto solo de muy cerca
           "perfil": [round(correlation_dim(X, a, b), 3) for a, b in SCALES],
           "intrinseca_nula": float(np.mean([two_nn(N) for N in nulls])),
           "correlacion_nula": float(np.mean([correlation_dim(N) for N in nulls]))}  # fmt: skip
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}
