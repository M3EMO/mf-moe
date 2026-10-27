import numpy as np

from mf_moe.blocks import (
    activation_signatures,
    composition,
    neuron_signatures,
    shared_atoms,
    weight_function_agreement,
)


def _hierarchical(n_levels: int = 6, dim: int = 16, ratio: float = 0.35, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Nube autosimilar: cada punto se parte en dos hijos a distancia `ratio` veces la del nivel anterior.
    Devuelve también la rama de nivel 2 (una de 4) de cada punto."""
    rng = np.random.default_rng(seed)
    pts, step, branch = np.zeros((1, dim)), 1.0, np.zeros(1, dtype=int)
    for level in range(n_levels):
        d = rng.standard_normal((len(pts), dim))
        d = d / np.linalg.norm(d, axis=1, keepdims=True) * step
        pts, step = np.concatenate([pts + d, pts - d]), step * ratio
        branch = np.concatenate([2 * branch, 2 * branch + 1]) if level < 2 else np.concatenate([branch, branch])
    return np.repeat(pts, 16, 0) + 1e-3 * rng.standard_normal((len(pts) * 16, dim)), np.repeat(branch, 16)


def test_signatures_sign_invariant():
    rng = np.random.default_rng(0)
    w_in, w_out = rng.standard_normal((2, 6, 8)), rng.standard_normal((2, 8, 3))
    X, lab = neuron_signatures(w_in, w_out)
    w_in2, w_out2 = w_in.copy(), w_out.copy()
    w_in2[:, 3:] *= -1  # up y salida con el signo cambiado: misma función
    w_out2 *= -1
    assert np.allclose(X, neuron_signatures(w_in2, w_out2)[0]) and lab.tolist() == [0, 0, 0, 1, 1, 1]


def test_self_similar_cloud_has_steady_slope_unlike_gaussian():
    X, _ = _hierarchical()
    r = composition(X, np.zeros(len(X), dtype=int), ks=[2, 4, 8, 16, 32], dims=16)
    assert r["real"]["cv_pendientes"] < 0.5 * r["nulo_gaussiano"]["cv_pendientes"]
    assert np.mean(r["real"]["pendientes"]) < np.mean(r["nulo_gaussiano"]["pendientes"])  # cae más rápido


def test_mixing_detects_families_inside_one_expert():
    X, lab = _hierarchical()  # cada rama de nivel 2 es un experto
    r = composition(X, lab, ks=[2, 4, 8], dims=16)
    assert r["real"]["mezcla"][-1] < 0.1 and r["mezcla_barajada"][-1] > 0.8


def test_shared_atoms_finds_copies_across_experts():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((200, 32))
    X = np.concatenate([base, base + 0.05 * rng.standard_normal(base.shape)])
    r = shared_atoms(X, np.repeat([0, 1], 200), n_sample=100)
    assert r["real"]["frac_coseno_otro_mayor_0.7"] > 0.9 and r["nulo_gaussiano"]["frac_coseno_otro_mayor_0.7"] < 0.1


def test_hierarchical_routing_beats_random_groups_when_router_is_clustered():
    from mf_moe.blocks import hierarchical_routing

    rng = np.random.default_rng(0)
    centers = rng.standard_normal((4, 32)) * 3
    W = np.repeat(centers, 8, 0) + 0.3 * rng.standard_normal((32, 32))  # 4 familias de 8 expertos
    H = centers[rng.integers(0, 4, 2000)] + rng.standard_normal((2000, 32))
    r = {x["m"]: x for x in hierarchical_routing(H, W, k=4, groups=(4,), n_null=5) if x["grupos"] == 4}
    assert r[1]["recall"] > 0.9 > r[1]["recall_nulo_p95"] and r[4]["recall"] == 1.0


def test_activation_signatures_see_same_function_behind_different_weights():
    rng = np.random.default_rng(0)
    H = np.concatenate([rng.standard_normal((500, 8)), np.zeros((500, 24))], 1)  # tokens en las primeras 8 dims
    w0 = rng.standard_normal((1, 64, 32))
    w1 = w0.copy()
    w1[..., 8:] += 3 * rng.standard_normal((1, 64, 24))  # pesos distintos fuera del espacio de los tokens
    w1[:, 32:] *= -1  # up con el signo cambiado: misma neurona
    w_in = np.concatenate([w0, w1])
    Xf, lab = activation_signatures(H, w_in)
    Xw, _ = neuron_signatures(w_in, rng.standard_normal((2, 32, 32)))
    assert np.allclose(Xf[:32], Xf[32:]) and lab.tolist() == [0] * 32 + [1] * 32
    assert shared_atoms(Xf, lab, n_sample=64)["real"]["frac_coseno_otro_mayor_0.7"] == 1.0
    assert shared_atoms(Xw, lab, n_sample=64)["real"]["frac_coseno_otro_mayor_0.7"] < 0.1


def test_weight_function_agreement():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16))
    assert weight_function_agreement(X, X)["pearson"] > 0.99
    assert abs(weight_function_agreement(X, rng.standard_normal((300, 16)))["pearson"]) < 0.05
