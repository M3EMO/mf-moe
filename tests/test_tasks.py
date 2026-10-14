import random

from mf_moe.tasks import DEFAULT_DIGITS, EOS, REVERSED_ANSWER, RULES, TASKS, eval_set, make_example, signature


def test_rules_hand_values():
    assert RULES["suma"](["12", "30"]) == "42"
    assert RULES["mult"](["12", "11"]) == "132"
    assert RULES["inv"](["1203"]) == "3021"
    assert RULES["ord"](["3120"]) == "0123"
    assert RULES["bin"](["5"]) == "101"
    assert RULES["mod"](["17", "5"]) == "2"
    assert RULES["sumamult"](["1", "2", "10"]) == "30"
    assert RULES["invsuma"](["120", "1"]) == "22"  # inv("120") = "021" = 21
    assert RULES["binsuma"](["2", "3"]) == "101"
    assert RULES["modmult"](["4", "5", "6"]) == "2"
    assert RULES["digmult"](["12", "11"]) == "6"  # 132 -> 1+3+2
    assert RULES["ordsuma"](["19", "1"]) == "02"


def test_every_task_has_a_unique_signature():
    sigs = [signature(t) for t in TASKS]
    assert len(set(sigs)) == len(TASKS) == 12 and set(DEFAULT_DIGITS) == set(TASKS)


def test_named_and_unnamed_share_the_same_numbers():
    for t in TASKS:
        a = make_example(t, random.Random(7), named=True)
        b = make_example(t, random.Random(7), named=False)
        assert a.prompt == f"{t}: " + b.prompt and a.answer == b.answer and a.answer.endswith(EOS)
        assert [len(x) for x in b.prompt.removesuffix(" =").split()] == list(signature(t))


def test_eval_set_is_fixed_and_disjoint_from_training_seeds():
    assert [e.text for e in eval_set("suma", True, n=50)] == [e.text for e in eval_set("suma", True, n=50)]
    train = {make_example("suma", random.Random(s), True).text for s in range(200)}
    assert not train & {e.text for e in eval_set("suma", True, n=1000)}


def test_reversed_tasks_write_the_answer_from_the_units():
    for t in TASKS:
        ex = make_example(t, random.Random(3), named=False)
        ops = ex.prompt.removesuffix(" =").split()
        expected = RULES[t](ops)
        assert ex.answer == (expected[::-1] if t in REVERSED_ANSWER else expected) + EOS
    assert {"mult", "mod"} <= REVERSED_ANSWER and "suma" not in REVERSED_ANSWER
