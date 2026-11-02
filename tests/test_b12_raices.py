import numpy as np

from colab.b9_bits import SINGLES
from colab.b12_raices import parents


def test_parent_is_the_expert_a_primitive_uses_more_than_the_others():
    # Todas usan mucho el experto 0; cada primitiva i usa además el experto i + 1.
    rt = {t: {"idx": np.array([[[0, i + 1]]] * 10)} for i, t in enumerate(SINGLES)}
    top, lift = parents(rt, 8)
    assert [int(top[t][0]) for t in SINGLES] == list(range(1, len(SINGLES) + 1))
    assert lift[SINGLES[0]][0] == 0.5
