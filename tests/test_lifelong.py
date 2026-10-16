import random

import torch

from mf_moe.lifelong import Lifelong
from mf_moe.model import build, routers
from mf_moe.tasks import eval_set
from mf_moe.train import adamw, encode, train_until


def _train(model, opt, ll, task="suma"):
    train_until(model, opt, [task], {task: eval_set(task, True, n=5)}, named=True, rng=random.Random(0),
                device="cpu", batch=8, eval_every=2, patience=1, aux_loss=ll.aux_loss, log=lambda r: None)  # fmt: skip


def test_schedule_grows_from_six_to_twelve_every_other_task():
    model = build(6, d=64, layers=2)
    opt, ll, sizes = adamw(model), Lifelong(), []
    for i in range(12):
        ll.start_task(model, opt, i)
        sizes.append(routers(model)[0].num_experts)
    assert sizes == [6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12]


def test_old_experts_and_router_rows_stay_frozen_while_the_new_one_learns():
    model = build(6, d=64, layers=1)
    opt, ll = adamw(model), Lifelong()
    _train(model, opt, ll)
    ll.start_task(model, opt, 1)
    mlp = model.model.layers[0].mlp
    before = [p.detach().clone() for p in (mlp.experts.down_proj, mlp.gate.weight)]
    _train(model, opt, ll, task="inv")
    after = (mlp.experts.down_proj, mlp.gate.weight)
    assert all(torch.equal(a[:6], b[:6]) for a, b in zip(after, before))
    assert not torch.equal(after[0][6], before[0][6])  # el experto nuevo entrena


def test_distillation_is_zero_at_the_start_of_a_task_and_grows_after_training():
    model = build(6, d=64, layers=1)
    opt, ll = adamw(model), Lifelong()
    ids, _ = encode(eval_set("mult", True, n=6), "cpu")
    assert ll.aux_loss(ids, model(input_ids=ids)) == 0.0  # primera tarea: sin maestro
    ll.start_task(model, opt, 2)  # tarea sin crecimiento: solo cambia el maestro
    with torch.no_grad():
        assert float(ll.aux_loss(ids, model(input_ids=ids))) < 1e-6
    _train(model, opt, ll, task="mult")
    with torch.no_grad():
        assert float(ll.aux_loss(ids, model(input_ids=ids))) > 0
