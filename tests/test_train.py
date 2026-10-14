import random

import torch

from mf_moe.model import build
from mf_moe.tasks import Example, eval_set
from mf_moe.train import accuracy, encode, train_until


def test_encode_masks_everything_but_the_answer():
    ids, labels = encode([Example("suma", "1 2 =", "3\n"), Example("suma", "10 20 =", "30\n")], "cpu")
    assert (
        labels[0].tolist().count(-100) == ids.shape[1] - 2 and bytes(labels[1][labels[1] != -100]) == b"30\n"
    )


def test_accuracy_is_exact_match_under_teacher_forcing():
    class Oracle(torch.nn.Module):
        """Predice el siguiente byte real: exactitud 1; con un byte cambiado, 0."""

        def __init__(self, flip):
            super().__init__()
            self.flip = flip

        def forward(self, input_ids):
            nxt = torch.roll(input_ids, -1, dims=1)
            if self.flip:
                nxt[:, -2] = (nxt[:, -2] + 1) % 256
            return type("O", (), {"logits": torch.nn.functional.one_hot(nxt, 256).float()})()

    exs = [Example("inv", "123 =", "321\n")] * 3
    assert accuracy(Oracle(False), exs, "cpu") == 1.0 and accuracy(Oracle(True), exs, "cpu") == 0.0


def test_train_until_stops_by_stall_alarm_on_a_tiny_model():
    model = build(4, d=64, layers=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    evals = {"mult": eval_set("mult", True, n=20)}
    log = []
    res = train_until(model, opt, ["mult"], evals, named=True, rng=random.Random(0), device="cpu",
                      batch=8, eval_every=2, patience=2, log=log.append)  # fmt: skip
    assert res.stalled and res.steps >= 6 and log and log[-1]["min_acc"] < 0.88


def test_stall_alarm_waits_while_any_task_improves(monkeypatch):
    # Regresión de la referencia del 2026-10-14 (ticket 05 de mf-moe-olvido): con «mult» trabada en 0 %,
    # la alarma miraba solo la peor tarea y cortaba el entrenamiento conjunto aunque «inv» avanzara.
    import mf_moe.train as T

    calls = {"inv": 0}

    def fake_accuracy(model, examples, device, batch=250):
        if examples[0].task == "mult":
            return 0.0
        calls["inv"] += 1
        return min(0.1 * calls["inv"], 0.5)  # mejora 5 evaluaciones y después se planta

    monkeypatch.setattr(T, "accuracy", fake_accuracy)
    model = build(4, d=64, layers=1)
    evals = {t: eval_set(t, True, n=5) for t in ("inv", "mult")}
    res = T.train_until(model, torch.optim.AdamW(model.parameters()), ["inv", "mult"], evals, named=True,
                        rng=random.Random(0), device="cpu", batch=4, eval_every=1, patience=2, log=lambda r: None)  # fmt: skip
    assert res.stalled and len(res.history) == 7  # 5 que mejoran + 2 de paciencia
