import copy
import functools

import numpy as np
import torch

from colab import b14_congelado as b14
from colab.b9_bits import PAIRS, SINGLES, eval_sets, fresh, train
from mf_moe.model import moe


def test_choose_uso_tfidf_y_azar():
    u = np.array([[5, 1, 9, 0], [0, 7, 7, 2]])
    bg = np.array([[1, 1, 100, 1], [1, 1, 1, 1]])
    assert b14.choose({"k": 2}, u, bg, 0) == [[0, 2], [1, 2]]
    # el experto 2 de la capa 0 es el más usado, pero el texto general también lo usa: tfidf lo descarta
    assert b14.choose({"k": 2, "tfidf": True}, u, bg, 0)[0] == [0, 1]
    azar = b14.choose({"k": 2, "azar": True}, u, bg, 0)
    assert all(len(set(r)) == 2 for r in azar) and azar == b14.choose({"k": 2, "azar": True}, u, bg, 0)


def test_rama_congela_filas_y_graduada_con_omega_cero_es_el_control(monkeypatch):
    monkeypatch.setattr(b14, "STAGE_B", 4)
    monkeypatch.setattr(b14, "train", functools.partial(train, every=2))
    ev = eval_sets(SINGLES + PAIRS)
    prims, pairs = {t: ev[t][:20] for t in SINGLES}, {t: ev[t][:20] for t in PAIRS}
    general = {"general": ev[SINGLES[0]][:20]}
    model, opt = fresh(0)
    omega = {L: torch.zeros(16) for L in range(4)}
    before = moe(model, 1)[0].gate_up_proj.detach().clone()

    def run(arm):
        m, o = copy.deepcopy((model, opt))
        out = b14.branch(m, o, arm, omega, prims, general, pairs, 0)
        return out, moe(m, 1)[0].gate_up_proj.detach()

    out, w = run({"k": 3})
    assert set(out["perdida_primitivas"]) == set(SINGLES) and all(v > 0 for v in out["perdida_primitivas"].values())
    rows = out["congelados"][1]
    assert torch.equal(w[rows], before[rows])
    assert not torch.equal(w[sorted(set(range(16)) - set(rows))], before[sorted(set(range(16)) - set(rows))])
    assert torch.equal(run({"c": 10.0})[1], run({})[1])  # Ω = 0: la escala graduada es 1
