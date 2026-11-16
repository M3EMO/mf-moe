import json

import numpy as np

from colab import b24_inversion as b24
from colab import b38_capacidad as b38
from colab import b40_fisher as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_share_is_one_when_every_entry_is_kept_with_unit_fisher():
    model = granite.tiny()
    sparse = b38.fisher(model, chars, [("The p of X is", " Y.")], top=1.0)
    ones = {n: (idx, f * 0 + 1, t) for n, (idx, f, t) in sparse.items()}
    r, g2 = b.share(model, chars, ("The p of H is", " N."), ones)
    assert abs(r - 1) < 1e-6 and g2 > 0


def test_main_writes_one_share_per_fact_and_leaves_the_model_untouched(tmp_path):
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    model = granite.tiny()
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    paths = [str(tmp_path / f) for f in ("r.json", "banco.json", "cands.json")]
    json.dump(facts, open(paths[1], "w"))
    json.dump([{"tipo": "t", "propiedad": "p", "cabeza": f"X{i}", "valor": f"Y{i}"} for i in range(3)], open(paths[2], "w"))
    b.main(*paths, model=model, tok=chars)
    h = json.load(open(paths[0]))["hechos"]
    assert len(h["r_nuevo"]) == len(h["r_viejo"]) == 12 and all(r >= 0 for r in h["r_nuevo"])
    assert all(np.allclose(p.detach(), start[n]) for n, p in model.named_parameters())
