import json

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b24_inversion as b24
from colab import b28_cabeza as b
from mf_moe import granite


def chars(text: str) -> dict:
    return {"input_ids": [ord(c) % 256 for c in text]}


def small(monkeypatch):
    for k, v in {"K": 2, "SETS": 2, "DOSES": (1, 2)}.items():
        monkeypatch.setattr(b24, k, v)
    for k, v in {"CHECK_EVERY": 1, "BATCH": 4}.items():
        monkeypatch.setattr(b17, k, v)
    for k, v in {"LAYERS": (1, 2), "STEPS": 4, "EVAL_EVERY": 2, "BOOT": 20}.items():
        monkeypatch.setattr(b, k, v)
    facts = b24.add_donors([{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}", "valor_nuevo": f"N{i}", "cabeza_nueva": f"C{i}"}
                            for i in range(12)], "t")  # fmt: skip
    cands = [{"propiedad": "pq"[i % 2], "cabeza": f"H{i}", "valor": f"V{i}"} for i in range(22)]  # 10 fuera del banco
    return facts, cands


def test_parts_reads_the_last_prompt_token_and_the_value_loss(monkeypatch):
    monkeypatch.setattr(b, "LAYERS", (1, 2))
    model = granite.tiny()
    items = [("The p of Ka is", " Lo."), ("Vi is", " Dremor the q.")]
    h, nll, _ = b.parts(model, chars, items)
    ids, mask, _ = b17.batch(chars, items)
    hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
    assert torch.allclose(h[1, : hs[1].shape[-1]], hs[1][1, len("Vi is") - 1].float())
    assert np.allclose(nll.detach().numpy(), b17.def_losses(model, chars, items), atol=1e-4)


def test_frozen_head_never_moves_the_trunk_and_joint_does(monkeypatch):
    facts, cands = small(monkeypatch)
    model, train = granite.tiny(), cands[12:]
    before = {k: v.clone() for k, v in model.named_parameters()}
    X = b.b27.prompt_states(model, chars, train, b.LAYERS)
    head = b.Head(X.mean(0), X.std(0) + 1e-8)
    b.train_head(model, chars, head, train[2:], train[:2], facts, None, False, "t")
    assert all(torch.equal(v, before[k]) for k, v in model.named_parameters())
    refs = b.ref_logprobs(model, chars, [b24.fwd(x) for x in train[2:]])
    assert abs(float(b.kl(b.parts(model, chars, [b24.fwd(x) for x in train[2:4]])[2], refs[:2]))) < 1e-2  # mismo modelo: KL casi 0
    b.train_head(model, chars, head, train[2:], train[:2], facts, refs, True, "t")
    assert model.training  # stage2 sigue en modo train
    assert any(not torch.equal(v, before[k]) for k, v in model.named_parameters())


def test_phase_one_stops_without_inheritance_and_phase_two_runs_with_it(tmp_path, monkeypatch):
    facts, cands = small(monkeypatch)
    bank, cand, out = str(tmp_path / "banco.json"), str(tmp_path / "c.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    json.dump(cands, open(cand, "w"))
    monkeypatch.setattr(b, "decide", lambda d: "no_hereda")
    b.main(out, bank, cand, model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    assert res["fase1"]["n_entrena"] + res["fase1"]["n_val"] == 10 and res["corridas"] == {}
    assert len(res["prediccion"]["cabeza_conjunta"]) == len(facts)
    out2 = str(tmp_path / "r2.json")
    for k, v in {"decide": lambda d: "hereda", "C1_MIN": -1.0, "I1_RHO": -1.0, "I1_SHIFT": 1e9}.items():
        monkeypatch.setattr(b, k, v)
    b.main(out2, bank, cand, model=granite.tiny(), tok=chars)
    res = json.load(open(out2))
    assert sorted(res["corridas"]) == sorted(f"s{s}|{a}|d{d}" for s in range(2) for a in ("cabeza", "oraculo") for d in (1, 2))
    assert abs(np.mean(res["pesos"]["cabeza"]) - 1) < 1e-3


def test_episode_ridge_matches_the_closed_form_ridge_and_carries_gradient():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(16, 50)), rng.normal(size=16)
    Z = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    p = b.episode_ridge(Z[:12], torch.tensor(y[:12], dtype=torch.float32), Z[12:], 3.0)
    A = X[:12] - X[:12].mean(0)
    w = A.T @ np.linalg.solve(A @ A.T + 3.0 * np.eye(12), y[:12] - y[:12].mean())
    assert np.allclose(p.detach().numpy(), y[:12].mean() + (X[12:] - X[:12].mean(0)) @ w, atol=1e-4)
    p.sum().backward()
    assert Z.grad is not None and Z.grad.abs().sum() > 0


def test_b28d_measures_two_replicas_and_runs_phase_two_only_when_it_inherits(tmp_path, monkeypatch):
    facts, cands = small(monkeypatch)
    for k, v in {"EPISODE": (3, 1), "CHECK_D": 2, "C1_MIN_D": -1.0, "I1_RHO": -1.0, "I1_SHIFT": 1e9}.items():
        monkeypatch.setattr(b, k, v)
    bank, cand, out = str(tmp_path / "banco.json"), str(tmp_path / "c.json"), str(tmp_path / "r.json")
    json.dump(facts, open(bank, "w"))
    json.dump(cands, open(cand, "w"))
    b.main_d(out, bank, cand, model=granite.tiny(), tok=chars)
    res = json.load(open(out))
    reps = res["fase1"]["replicas"]
    assert len(reps) == 2 and all(r["I1"] == "pasa" and len(r["prediccion"]) == len(facts) for r in reps)
    assert res["corridas"] == {} or res["fase1"]["D1"] == "hereda"
    monkeypatch.setattr(b, "decide_replicas", lambda reps: "hereda")
    out2 = str(tmp_path / "r2.json")
    b.main_d(out2, bank, cand, model=granite.tiny(), tok=chars)
    assert sorted(json.load(open(out2))["corridas"]) == sorted(f"s{s}|cabeza|d{d}" for s in range(2) for d in (1, 2))


def test_decide_replicas_needs_both_replicas():
    ok, low = {"I1": "pasa", "delta": [0.1, 0.02, 0.2]}, {"I1": "pasa", "delta": [0.0, -0.05, 0.04]}
    assert b.decide_replicas([ok, ok]) == "hereda" and b.decide_replicas([low, low]) == "no_hereda"
    assert b.decide_replicas([ok, low]) == "indeterminado" and b.decide_replicas([ok, {"I1": "falla"}]) == "sin_medida"
