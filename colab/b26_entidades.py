"""B26: celda 2×2 que separa origen y entidad (ticket 08). Hechos falsos sobre entidades reales, enseñados en las dos direcciones
con la etapa 1 de B24. La entidad es real, como en B25; el origen es el ajuste fino, como en B24.
Cabezas: organizaciones y obras de UniEdit. Valores: personas de UniEdit. Las propiedades, los nuevos y las reglas son los de B24.
Uso: python -m colab.b26_entidades SALIDA.json HECHOS.json
"""

import json
import random
import re
import sys

from colab import b24_inversion as b24

DOSES = (10, 40)  # las dosis donde B24 y B25 ya están medidos; en 40 la distancia entre los dos es mayor
NAME = re.compile(r"[A-Z][\w'.-]*( [A-Z][\w'.-]*){1,3}")


def bank(hechos_path: str, n: int | None = None, seed: str = "b26:banco") -> list[dict]:
    H = json.load(open(hechos_path, encoding="utf-8"))["hechos"]
    n, rng = n or b24.N_FACTS, random.Random(seed)
    heads = sorted({x["cabeza"] for t in ("organizacion", "obra_cultural") for x in H[t] if NAME.fullmatch(x["cabeza"])})
    values = sorted({x["cabeza"] for x in H["persona"] if NAME.fullmatch(x["cabeza"])} - set(heads))
    heads, values = rng.sample(heads, n), rng.sample(values, n)
    taken = {s for v in H.values() for x in v for s in (x["cabeza"], x["valor"])}
    new_v, new_h = b24.names(n, rng, taken), b24.names(n, rng, taken)
    facts = [{"propiedad": b24.PROPS[i % len(b24.PROPS)], "cabeza": h, "valor": v, "valor_nuevo": nv, "cabeza_nueva": nh}
             for i, (h, v, nv, nh) in enumerate(zip(heads, values, new_v, new_h))]  # fmt: skip
    return b24.add_donors(facts, seed)


def main(path: str, hechos_path: str, model=None, tok=None) -> None:
    b24.main(path, model, tok, dose=True, facts=bank(hechos_path), doses=DOSES)


if __name__ == "__main__":
    main(*sys.argv[1:3])
