import numpy as np
import torch

from mf_moe import expert_tree as T
from mf_moe.granite import tiny
from mf_moe.model import build


def _usage(rng, family_blocks: bool, experts=24):
    U = np.zeros((1, len(T.DOMAINS), experts), dtype=np.int64)
    for d, name in enumerate(T.DOMAINS):
        fam = next(i for i, ds in enumerate(T.FAMILIES.values()) if name in ds)
        lo = fam * 6 if family_blocks else 0
        hi = lo + 6 if family_blocks else experts
        U[0, d] = np.bincount(rng.integers(lo, hi, 2000), minlength=experts)
    return U, np.zeros((1, experts, experts))


def test_family_blocks_are_recovered_and_noise_is_not():
    rng = np.random.default_rng(0)
    tree = T.analyze(*_usage(rng, True))[0]
    flat = T.analyze(*_usage(rng, False))[0]
    assert tree["rand_ajustado_familias"] == 1.0 and tree["similitud_misma_familia"] > 0.9 > tree["similitud_otra_familia"]
    assert flat["rand_ajustado_familias"] < 0.5 and abs(flat["similitud_misma_familia"] - flat["similitud_otra_familia"]) < 0.05
    assert tree["info_mutua_norm"] > flat["info_mutua_norm"]


def test_recorder_counts_every_routing_slot_in_granite_and_qwen2_moe():
    ids = torch.randint(3, 60, (2, 8))
    for model in (tiny(n_experts=6), build(6, d=64, layers=2, ctx=32)):
        rec = T.Recorder(model, n_domains=2)
        rec.domain = 1
        with torch.no_grad():
            model(input_ids=ids)
        rec.close()
        n, k = T.moe_shape(model.config)
        assert len(rec.mods) == model.config.num_hidden_layers
        assert (rec.usage[:, 1].sum(1) == ids.numel() * k).all() and rec.usage[:, 0].sum() == 0
        assert np.allclose(np.diagonal(rec.coact, axis1=1, axis2=2).sum(1), ids.numel() * k)



def test_fractal_scores_nested_specialization_above_a_single_level():
    rng = np.random.default_rng(0)
    fam = np.array([i for i, ds in enumerate(T.FAMILIES.values()) for _ in ds])
    levels = [fam, np.arange(len(T.DOMAINS))]

    def usage(nested: bool):  # 24 expertos, 6 por familia; anidado = además cada dominio usa su par en la familia
        U = np.zeros((1, len(T.DOMAINS), 24), dtype=np.int64)
        for d in range(len(T.DOMAINS)):
            lo = fam[d] * 6 + (2 * (d % 3) if nested else 0)
            U[0, d] = np.bincount(rng.integers(lo, lo + (2 if nested else 6), 3000), minlength=24)
        return U

    nested, flat = T.fractal(usage(True), levels)[0], T.fractal(usage(False), levels)[0]
    assert nested["autosimilitud"] > 0.5 > flat["autosimilitud"]
    assert flat["exceso"][0] > 0.5 and abs(flat["exceso"][1]) < 0.02  # una familia sola: el idioma no suma


def test_territory_shape_tells_a_flat_sheet_a_fractal_and_a_cloud_apart():
    rng = np.random.default_rng(0)
    rot = np.linalg.qr(rng.normal(size=(40, 40)))[0]
    sheet = np.c_[rng.uniform(size=(700, 2)), np.zeros((700, 38))] @ rot  # plano de 2 dimensiones dentro de 40
    v, p, pts = np.array([[0, 0], [1, 0], [0.5, 0.866]]), np.array([0.3, 0.3]), []
    for i in range(2700):  # triángulo de Sierpinski por el juego del caos
        p = (p + v[rng.integers(3)]) / 2
        if i >= 2000:
            pts.append(p)
    sier = np.c_[np.array(pts), np.zeros((700, 38))] @ rot
    cloud = rng.normal(size=(700, 8)) @ rng.normal(size=(8, 40))  # gaussiana de 8 dimensiones dentro de 40
    s, f, c = T.territory_shape(sheet), T.territory_shape(sier), T.territory_shape(cloud)
    assert 1.6 < s["intrinseca"] < 2.5 and 1.7 < s["correlacion"] < 2.3
    assert 1.3 < f["correlacion"] < 1.8 < f["correlacion_nula"]  # Sierpinski: log 3 / log 2 ≈ 1,585
    assert c["intrinseca"] > 5 and abs(c["correlacion"] - c["correlacion_nula"]) < 0.6  # sin estructura: igual al nulo


def test_recorder_keeps_a_sample_of_each_experts_territory():
    model = tiny(n_experts=6)
    rec = T.Recorder(model, n_domains=1, keep_layers=(1,), keep_frac=0.5)
    with torch.no_grad():
        model(input_ids=torch.randint(3, 60, (4, 32)))
    rec.close()
    hs, idx, dom = rec.kept[1][0]
    assert hs.shape[1] == model.config.hidden_size and len(hs) == len(idx) and 30 < len(hs) < 100 and dom == 0
