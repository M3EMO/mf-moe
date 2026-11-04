import json

import numpy as np

from colab import b15_repeticion as b
from mf_moe import hierarchy as h


def test_arm_weights_follow_tree():
    dist = h.tree_distance()[0]  # [0, 1, 2, 2, 3, 3, 3, 3]
    assert b.arm_weights("sin_repeticion", dist) is None
    u = b.arm_weights("uniforme", dist)
    assert u[0] == 0 and np.allclose(u[1:], 1 / 7)
    c = b.arm_weights("cerca", dist)
    assert c[0] == 0 and c[1] > c[2] > c[4] and np.isclose(c[1], 4 / 7)
    far = b.arm_weights("lejos", dist)
    assert np.allclose(far, [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25])
    rise = np.array([9, 0.5, 0, 0, 0, 0, 0, 1.0])
    p = b.arm_weights("perdida", dist, rise)
    assert p[0] == 0 and p[1] > p[2] and np.isclose(p.sum(), 1)


def test_smoke_runs_and_resumes(tmp_path):
    out = str(tmp_path / "b15.json")
    b.main(out, 1, 20, 10)
    r = json.load(open(out))
    assert set(r["brazos"]) == set(b.ARMS) and len(r["cierre_a"]) == h.N_LANG
    assert len(r["brazos"]["cerca"]["olvido_d123"]) == 3
    b.main(out, 1, 20, 10)  # retoma desde el checkpoint de A sin repetir brazos
    assert json.load(open(out))["brazos"] == r["brazos"]
