"""Tareas sintéticas de dígitos para aprendizaje continuo (ticket 04 del mapa mf-moe-olvido).

Todas comparten el formato «n1 n2 ... =respuesta». En el escenario sin nombre, la tarea solo se
deduce de su firma: cuántos números trae y de qué largo. En el escenario con nombre, el ejemplo
empieza además con «<tarea>: ». Los dos escenarios usan exactamente los mismos números.
Los largos (`digits`) son la perilla de dificultad que calibra la prueba de capacidad.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

EOS = "\n"


def _rev(s: str) -> str:
    return s[::-1]


def _sorted(s: str) -> str:
    return "".join(sorted(s))


# Divisor fijo y ajeno a la base 10: con divisor variable no aprende (calibración 3, 2026-10-15).
MOD = 7

# Cada regla recibe los operandos como texto (sin ceros a la izquierda) y devuelve la respuesta.
RULES: dict[str, Callable[[list[str]], str]] = {
    "suma": lambda o: str(int(o[0]) + int(o[1])),
    "mult": lambda o: str(int(o[0]) * int(o[1])),
    "inv": lambda o: _rev(o[0]),
    "ord": lambda o: _sorted(o[0]),
    "bin": lambda o: bin(int(o[0]))[2:],
    "mod": lambda o: str(int(o[0]) % MOD),
    "sumamult": lambda o: str((int(o[0]) + int(o[1])) * int(o[2])),
    "invsuma": lambda o: str(int(_rev(o[0])) + int(o[1])),
    "binsuma": lambda o: bin(int(o[0]) + int(o[1]))[2:],
    "modmult": lambda o: str((int(o[0]) * int(o[1])) % MOD),
    "digmult": lambda o: str(sum(map(int, str(int(o[0]) * int(o[1]))))),
    "ordsuma": lambda o: _sorted(str(int(o[0]) + int(o[1]))),
}

# Largo en dígitos de cada operando: la firma (cantidad, largos) es distinta para cada tarea.
DEFAULT_DIGITS: dict[str, tuple[int, ...]] = {
    "suma": (7, 7),
    "mult": (5, 2),
    "inv": (12,),
    "ord": (10,),
    "bin": (5,),
    "mod": (6,),
    "sumamult": (4, 4, 2),
    "invsuma": (8, 6),
    "binsuma": (4, 3),
    "modmult": (4, 2),
    "digmult": (5, 1),
    "ordsuma": (6, 6),
}
TASKS = tuple(RULES)
# Familia de multiplicación y módulo: operandos y respuesta desde la unidad, como en los éxitos sin
# pasos intermedios de la literatura (nota del vault «mf-moe: por qué no aprenden mult y mod»).
LSB_FIRST = frozenset({"mult", "mod", "sumamult", "modmult", "digmult"})

# Composiciones: el resultado intermedio va escrito antes del final (decisión del usuario, 2026-10-15).
STEPS: dict[str, Callable[[list[str]], str]] = {
    "sumamult": lambda o: str(int(o[0]) + int(o[1])),
    "binsuma": lambda o: str(int(o[0]) + int(o[1])),
    "modmult": lambda o: str(int(o[0]) * int(o[1])),
    "digmult": lambda o: str(int(o[0]) * int(o[1])),
}


@dataclass(frozen=True)
class Example:
    task: str
    prompt: str
    answer: str  # incluye EOS: el modelo aprende dónde terminar

    @property
    def text(self) -> str:
        return self.prompt + self.answer


def _number(rng: random.Random, digits: int) -> str:
    return str(rng.randint(10 ** (digits - 1), 10**digits - 1))


def make_example(
    task: str, rng: random.Random, named: bool, digits: dict[str, tuple[int, ...]] | None = None
) -> Example:
    lengths = (digits or DEFAULT_DIGITS)[task]
    ops = [_number(rng, d) for d in lengths]
    show = _rev if task in LSB_FIRST else str
    prompt = ("" if not named else f"{task}: ") + " ".join(map(show, ops)) + " ="
    parts = [STEPS[task](ops)] if task in STEPS else []
    return Example(task, prompt, " ".join(map(show, parts + [RULES[task](ops)])) + EOS)


def signature(task: str, digits: dict[str, tuple[int, ...]] | None = None) -> tuple[int, ...]:
    """Lo único que distingue a la tarea sin nombre: cantidad y largo de los números."""
    return (digits or DEFAULT_DIGITS)[task]


def eval_set(task: str, named: bool, n: int = 1000, seed: int = 10_000, digits=None) -> list[Example]:
    """Ejemplos fijos de evaluación. La semilla está lejos de las de entrenamiento (0, 1, 2...)."""
    rng = random.Random(f"eval-{task}-{seed}")
    return [make_example(task, rng, named, digits) for _ in range(n)]
