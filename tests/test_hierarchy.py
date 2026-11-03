import numpy as np

from mf_moe import hierarchy as h


def test_dos_lenguas_a_distancia_d_difieren_en_las_d_tablas_de_abajo():
    T = h.tree_distance()
    assert T[0, 1] == 1 and T[0, 2] == 2 and T[0, 7] == 3 and T[5, 5] == 0
    for a in range(h.N_LANG):
        for b in range(h.N_LANG):
            shared = [h.clade(a, h.owner_depth(lv)) == h.clade(b, h.owner_depth(lv)) for lv in range(h.L)]
            assert shared == [lv < h.L - T[a, b] for lv in range(h.L)]


def test_reglas_sin_ambiguedad_y_oraciones_validas():
    tabs = h.rules(0)
    for t in tabs:
        for node in t:
            assert len({tuple(x) for x in node.reshape(-1, h.S)}) == h.V * h.M
    rng = np.random.default_rng(0)
    lang = rng.integers(h.N_LANG, size=50)
    syms, latent = h.sample(tabs, lang, rng)
    assert syms.shape == (50, h.SEQ) and [x.shape[1] for x in latent] == [1, 2, 4, 8]
    for i in range(50):  # cada par de la superficie sale de su padre con una regla de su lengua
        tab = tabs[h.L - 1][h.clade(lang[i], h.owner_depth(h.L - 1))]
        for j, parent in enumerate(latent[-1][i]):
            assert any((tab[parent, r] == syms[i, 2 * j : 2 * j + 2]).all() for r in range(h.M))


def test_traducciones_comparten_los_latentes_hasta_el_nivel_l_menos_d():
    tabs, T = h.rules(0), h.tree_distance()
    syms, latent = h.sample_family(tabs, 40, np.random.default_rng(0))
    assert syms.shape == (40, h.N_LANG, h.SEQ)
    for a in range(h.N_LANG):
        for b in range(h.N_LANG):
            for lv in range(h.L):
                same = (latent[lv][:, a] == latent[lv][:, b]).all()
                assert same if lv <= h.L - T[a, b] else not same
            assert (syms[:, a] == syms[:, b]).all() == (a == b)
    for a in range(h.N_LANG):  # cada lengua sigue su propia gramática
        tab = tabs[h.L - 1][h.clade(a, h.owner_depth(h.L - 1))]
        for i in range(40):
            for j, parent in enumerate(latent[-1][i, a]):
                assert any((tab[parent, r] == syms[i, a, 2 * j : 2 * j + 2]).all() for r in range(h.M))


def test_ids_vocabulario_compartido_o_propio():
    syms, lang = np.array([[0, 15]]), np.array([7])
    assert h.ids(syms, lang, False).tolist() == [[h.BOS, 32, 47]]
    assert h.ids(syms, lang, True).tolist() == [[h.BOS, 128 + 112, 128 + 127]]


def test_tree_recovery_detecta_el_arbol_y_no_el_azar():
    r = h.tree_recovery(h.tree_distance().astype(float))
    assert r["mantel"] > r["mantel_nulo_p95"] and r["tripletas"] == 1.0
    D = np.random.default_rng(1).random((h.N_LANG, h.N_LANG))
    r = h.tree_recovery((D + D.T) / 2)
    assert r["tripletas"] <= r["tripletas_nulo_p95"] + 0.05


def test_uso_y_jsd():
    idx = np.array([[0, 1], [0, 1], [2, 3]])
    U = h.usage(idx, np.array([0, 0, 1]), 2, 4)
    assert U.tolist() == [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]]
    assert abs(h.jsd_matrix(U)[0, 1] - 1.0) < 1e-9


def test_kmeans_separa_dos_nubes():
    rng = np.random.default_rng(0)
    H = np.concatenate([rng.normal(0, 0.1, (50, 4)), rng.normal(5, 0.1, (50, 4))])
    a = h.kmeans_topk(H, 2, 1)[:, 0]
    assert len(set(a[:50])) == 1 and len(set(a[50:])) == 1 and a[0] != a[50]


def test_acuerdo_mknn_y_distancia():
    A = np.array([[0, 1], [2, 3]])
    assert h.agreement(A, A) == 1.0 and h.agreement(A, np.array([[1, 5], [4, 6]])) == 0.25
    X = np.random.default_rng(0).normal(size=(30, 5))
    assert h.mknn(X, X) == 1.0 and h.mknn(X, np.random.default_rng(1).normal(size=(30, 5))) < 0.6
    assert h.by_distance(h.tree_distance().astype(float)) == [1.0, 2.0, 3.0]
