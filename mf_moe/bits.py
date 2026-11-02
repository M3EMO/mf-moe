"""Tareas de bits con bloques conocidos (B9a, mapa de bloques): verdad de base para medir si los expertos
aprenden primitivas y las reusan al componer.

Formato de mf_moe/tasks.py: «bits =salida». Una tarea es una composición de primitivas escrita con «+», en el
orden en que se aplican: «rot+not» rota y después niega. `primitives(tarea)` devuelve esa lista.
La variante recursiva «rec:<p>» aplica la misma primitiva en cada escala (bloques de 2, 4, 8...): en cada
bloque, la mitad derecha pasa por la primitiva. Con «not» sobre ceros da Thue-Morse: es el control positivo
de autosimilitud.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable

from mf_moe.tasks import EOS, Example

N_BITS = 16
MASK = "0011010111100010"  # fija, sin simetría (ni palíndromo ni periódica): si no, xor conmuta con rev o rot


def _xor(b: str) -> str:
    m = (MASK * (len(b) // len(MASK) + 1))[: len(b)]
    return "".join("1" if x != y else "0" for x, y in zip(b, m))


def _andor(b: str) -> str:
    h = len(b) // 2
    a, c = b[:h], b[h : 2 * h]
    return "".join(str(int(x) & int(y)) for x, y in zip(a, c)) + "".join(str(int(x) | int(y)) for x, y in zip(a, c)) + b[2 * h :]


# Todas conservan el largo: así cualquier composición es válida.
PRIMS: dict[str, Callable[[str], str]] = {
    "not": lambda b: b.translate(str.maketrans("01", "10")),
    "xor": _xor,
    "rot": lambda b: b[1:] + b[:1],
    "rev": lambda b: b[::-1],
    "par": lambda b: "".join(str(int(b[i]) ^ int(b[(i + 1) % len(b)])) for i in range(len(b))),  # ventana de 2
    "andor": _andor,
}
INVERSE: dict[str, Callable[[str], str]] = {
    "not": PRIMS["not"], "xor": _xor, "rot": lambda b: b[-1:] + b[:-1], "rev": PRIMS["rev"],
}  # fmt: skip
RECURSIVE = ("not", "xor", "rot", "rev", "par")  # andor necesita mitades de 2 bits o más


def primitives(task: str) -> list[str]:
    return [task[4:]] if task.startswith("rec:") else task.split("+")


def recursive(prim: str, b: str) -> str:
    """En cada escala s = 2, 4, ..., len(b): la mitad derecha de cada bloque de s bits pasa por `prim`."""
    s = 2
    while s <= len(b):
        b = "".join(b[i : i + s // 2] + PRIMS[prim](b[i + s // 2 : i + s]) for i in range(0, len(b), s))
        s *= 2
    return b


def apply(task: str, b: str) -> str:
    if task.startswith("rec:"):
        return recursive(task[4:], b)
    for p in primitives(task):
        b = PRIMS[p](b)
    return b


# shortcut: hay composiciones con la misma función (not conmuta con rot y rev; par absorbe a not, par∘rot = rot∘par);
# B9b debe agrupar tareas por función sobre ejemplos antes de medir información mutua experto × primitiva.
def compositions(max_len: int = 3) -> list[str]:
    return ["+".join(c) for k in range(1, max_len + 1) for c in itertools.product(PRIMS, repeat=k)]


def make_bits_example(task: str, rng: random.Random, named: bool = False, n_bits: int = N_BITS,
                      steps: bool = False) -> Example:  # fmt: skip
    """Con `steps`, la respuesta escribe cada resultado intermedio de una composición, separados por espacio."""
    b = "".join(rng.choice("01") for _ in range(n_bits))
    out = [apply(task, b)]
    if steps and not task.startswith("rec:"):
        out = list(itertools.accumulate(primitives(task), lambda x, p: PRIMS[p](x), initial=b))[1:]
    return Example(task, ("" if not named else f"{task}: ") + b + " =", " ".join(out) + EOS)
