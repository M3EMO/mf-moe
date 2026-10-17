import sys
import types

import numpy as np

sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=None))
sys.path.insert(0, "colab")
import prepare_phase2 as pp  # noqa: E402


class _Tok:
    vocab_size, eos_token_id = 100, 0

    def __call__(self, text):
        return {"input_ids": [int(x) for x in text.split()]}


def test_eval_takes_whole_documents_from_the_whole_stream(tmp_path, monkeypatch):
    # cada documento repite su número: así se ve de qué documento viene cada token
    docs = [{"text": " ".join([str(i + 1)] * (30 if i % 3 else 500))} for i in range(4000)]
    monkeypatch.setattr(pp, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda _: _Tok()))
    monkeypatch.setattr(pp, "load_dataset", lambda *a, **k: iter(docs))
    monkeypatch.setattr(pp, "SOURCES", {"x": ("falso", None)})
    monkeypatch.setattr(pp, "SEQ", 64)
    monkeypatch.setattr(pp, "EVAL_SEQS", 50)
    monkeypatch.setattr(pp, "EVAL_PER_DOC", 2)
    pp.main(str(tmp_path), 64 * 400)
    ev, tr = np.load(tmp_path / "x_eval.npy"), np.load(tmp_path / "x_train.npy")
    assert ev.shape == (50, 64) and tr.shape == (400, 64)
    e, t = set(np.unique(ev)) - {0}, set(np.unique(tr)) - {0}
    assert not e & t and len(e) >= 25 and max(e) > 0.9 * max(t)
