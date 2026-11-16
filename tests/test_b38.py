import json

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b38_capacidad as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def test_kenyon_with_zero_output_changes_nothing_and_keeps_k_units():
    k = b.Kenyon(64, sparse=True)
    h = torch.randn(3, 5, 64)
    assert torch.equal(k(h), h) and (k.w_in.sum(1) == b.CLAWS).all()
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    a = torch.relu(x @ k.w_in.T)
    assert int((a * (a >= a.topk(k.k, -1).values[..., -1:]) > 0).sum(-1).max()) <= k.k


def test_ewc_hook_pulls_toward_the_start():
    lin = torch.nn.Linear(3, 1, bias=False)
    theta = lin.weight.detach().flatten().clone()
    sparse = {"weight": (torch.tensor([0, 2]), torch.tensor([1.0, 2.0]), theta[[0, 2]])}
    hooks = b.ewc_hooks(lin, sparse, lam=10.0)
    with torch.no_grad():
        lin.weight += 0.5
    lin(torch.zeros(1, 3)).sum().backward()  # la pérdida no depende de los pesos: queda solo la penalidad
    assert torch.allclose(lin.weight.grad.flatten(), torch.tensor([5.0, 0.0, 10.0]))
    [h.remove() for h in hooks]


def test_choose_prefers_low_damage_among_values_that_learn():
    run = lambda rise, new: {"subida_ida": rise, "sobrescritos": [0], "perdida_nuevo": [new]}
    res = {"corridas": {"s0|a=1|d40": run([9, 0.5], 0.1), "s0|a=2|d40": run([9, 0.2], 0.9), "s0|a=3|d40": run([9, 0.3], 0.4)}}
    assert b.choose(res, "a", (1, 2, 3)) == (3, True)
    res["corridas"]["s0|a=3|d40"]["perdida_nuevo"] = [0.8]
    res["corridas"]["s0|a=1|d40"]["perdida_nuevo"] = [0.7]
    assert b.choose(res, "a", (1, 2, 3)) == (1, False)


def test_b38_ewc_at_zero_repeats_b25_and_the_fly_leaves_the_base_untouched(tmp_path, monkeypatch):
    for k, v in {"K": 2, "SETS": 2}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    for k, v in {"DOSES": (1, 3), "LAYER": 1, "LAMBDAS": (0.0,), "LRS": (1e-3,), "EXPAND": 4}.items():
        monkeypatch.setattr(b, k, v)
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    model = granite.tiny()
    base_i, base_v = b24.losses(model, chars, facts)
    ref = {"corridas": {}}
    b24.stage2(model, chars, facts, base_i, base_v, ref, str(tmp_path / "s.json"), dose=True, tag="b25", doses=(1, 3),
               arms={"ida": ("ida", None, None)})  # fmt: skip
    start = {n: p.detach().clone() for n, p in model.named_parameters()}
    b25 = {"corridas": {k.replace("|ida|d3", "|ida|d40"): v for k, v in ref["corridas"].items()}}
    paths = [str(tmp_path / f) for f in ("r.json", "banco.json", "b25.json", "cands.json")]
    json.dump(facts, open(paths[1], "w"))
    json.dump(b25, open(paths[2], "w"))
    json.dump([{"tipo": "t", "propiedad": "p", "cabeza": f"X{i}", "valor": f"Y{i}"} for i in range(4)] + facts[:2], open(paths[3], "w"))
    b.main(*paths, model=model, tok=chars)
    R = json.load(open(paths[0]))
    runs = R["corridas"]
    assert R["config"]["fisher_hechos"] == 4 and R["eleccion"]["ewc"] == 0.0
    assert np.allclose(runs["s1|ewc|d3"]["subida_ida"], ref["corridas"]["s1|ida|d3"]["subida_ida"], atol=2e-3)
    assert np.abs(runs["s1|mosca|d3"]["subida_ida"]).max() > 0 and "s1|mosca_densa|d3" in runs
    assert not hasattr(model, "kenyon") and all(p.requires_grad for p in model.parameters())
    assert all(torch.allclose(p, start[n]) for n, p in model.named_parameters())
