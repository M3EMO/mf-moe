import random

import mf_moe.sequential as S
import mf_moe.train as T
from mf_moe.model import build as T_build
from mf_moe.tasks import TASKS, Example


def test_task_order_is_a_fixed_permutation_per_seed():
    assert S.task_order(0) == S.task_order(0) != S.task_order(1) and sorted(S.task_order(2)) == sorted(TASKS)


def test_metrics_on_a_hand_matrix():
    order = ["a", "b", "c"]
    R = [{"a": 0.9, "b": 0.0, "c": 0.0}, {"a": 0.5, "b": 0.9, "c": 0.1}, {"a": 0.3, "b": 0.6, "c": 0.9}]
    m = S.metrics(R, order)
    assert abs(m["exactitud_media"] - 0.6) < 1e-9
    assert abs(m["bwt"] - ((0.3 - 0.9) + (0.6 - 0.9)) / 2) < 1e-9  # -0,45
    assert abs(m["olvido"] - ((0.9 - 0.3) + (0.9 - 0.6)) / 2) < 1e-9  # +0,45


def test_replay_takes_ten_percent_of_each_batch_from_old_tasks(monkeypatch):
    seen = []

    def fake_encode(exs, device):
        seen.append([e.task for e in exs])
        return real_encode(exs, device)

    real_encode = T.encode
    monkeypatch.setattr(T, "encode", fake_encode)
    model = S.build(4, d=64, layers=1)
    buffer = {"suma": [Example("suma", "1 2 =", "3\n")], "inv": [Example("inv", "12 =", "21\n")]}
    T.train_until(model, T.adamw(model), ["mult"], {"mult": S.eval_set("mult", True, n=5)}, named=True,
                  rng=random.Random(0), device="cpu", batch=40, eval_every=3, patience=1, replay=buffer,
                  log=lambda r: None)  # fmt: skip
    batch = seen[0]
    assert batch.count("mult") == 36 and batch.count("suma") + batch.count("inv") == 4


def test_run_fills_the_matrix_and_the_buffer_on_cpu(monkeypatch):
    monkeypatch.setattr(S, "build", lambda n, seed=0: T_build(4, seed=seed, d=64, layers=1))
    monkeypatch.setattr(S, "BUFFER", 3)
    out = S.run(S.FIJO_REPETICION, seed=0, named=True, device="cpu", eval_n=5, batch=8, eval_every=2, patience=1,
                log=lambda r: None)  # fmt: skip
    assert len(out["R"]) == 12 and set(out["R"][0]) == set(TASKS) and len(out["pasos"]) == 12
    assert out["orden"] == S.task_order(0) and out["condicion"]["replay"] and "bwt" in out

