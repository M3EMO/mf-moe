import random

from mf_moe.bits import INVERSE, PRIMS, RECURSIVE, apply, compositions, make_bits_example, primitives, recursive


def test_primitives_are_deterministic_length_preserving_and_invertible():
    rng = random.Random(0)
    for _ in range(50):
        b = "".join(rng.choice("01") for _ in range(16))
        for name, f in PRIMS.items():
            assert f(b) == f(b) and len(f(b)) == 16, name
        for name, inv in INVERSE.items():
            assert inv(PRIMS[name](b)) == b, name


def test_examples_carry_their_primitives_and_compose_in_order():
    ex = make_bits_example("rot+not", random.Random(1))
    bits = ex.prompt.split(" =")[0]
    assert primitives(ex.task) == ["rot", "not"]
    assert ex.answer.strip() == PRIMS["not"](PRIMS["rot"](bits))
    assert apply("rev+xor", bits) != apply("xor+rev", bits)  # el orden importa
    assert len(compositions(3)) == 6 + 36 + 216 and primitives("rec:rev") == ["rev"]


def test_recursive_variant_repeats_the_rule_at_every_scale():
    assert recursive("not", "0" * 16) == "0110100110010110"  # Thue-Morse
    rng = random.Random(2)
    for p in RECURSIVE:
        b = "".join(rng.choice("01") for _ in range(16))
        h = 8  # la misma regla a la mitad de escala, y la mitad derecha pasa por p
        assert apply(f"rec:{p}", b) == recursive(p, b[:h]) + PRIMS[p](recursive(p, b[h:])), p
