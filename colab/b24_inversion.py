"""B24 (candidato 1): ¿el olvido es direccional? Si un hecho se sabe en las dos direcciones y el ajuste sobrescribe una sola,
¿la otra sobrevive? «The Illusion of Latent Generalization» (ICLR 2026, taller) halla que la ida y la vuelta viven en entradas
separadas; acá se prueba eso por su efecto, no por representaciones.

Banco ficticio, como Berglund (Reversal Curse): cabezas y valores inventados, todos distintos, así cada hecho es uno a uno y
Granite no trae nada de antes. Ida: «The {propiedad} of {cabeza} is» + « {valor}.». Vuelta: «{valor} is the {propiedad} of» + « {cabeza}.».
- Etapa 1: enseña los N_FACTS hechos en las dos direcciones hasta que la pérdida media baja de STOP1.
- Etapa 2: por conjunto (SETS conjuntos disjuntos de N_OVER hechos), dos brazos que parten de la etapa 1:
  «ida» enseña « {valor_nuevo}.» en la ida; «vuelta» enseña « {cabeza_nueva}.» en la vuelta. Sin anclas.
- Medida: subida de la pérdida del hecho viejo en cada dirección, para los sobrescritos y para los no tocados.
- Modo «dosis»: la etapa 2 no para por pérdida; corre hasta max(DOSES) pasos y mide en cada paso de DOSES.
Uso: python -m colab.b24_inversion SALIDA.json [dosis]
"""

import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b23_uniedit as b23
from colab.b16_conceptos import DEVICE
from mf_moe.phase2 import amp

N_FACTS, SETS, K = 256, 8, 4
N_OVER = N_FACTS // SETS  # los conjuntos cubren el banco: cada hecho se sobrescribe una vez por brazo
STOP1, MAX_STEPS1, CHECK1, GATE1 = 0.25, 2000, 50, 0.9
PROPS = ("founder", "director", "author", "inventor", "architect", "composer", "designer", "discoverer")
SYLLABLES = "ka lo vi dre mor tal sen qui bar nu fel gra osh pe zin ru tov ela mi dak sor ven hal it jo cre wes pul amb yri".split()
ARMS = ("ida", "vuelta")
DOSES = (10, 40, 160)  # 10 repite B24: con las mismas semillas, la etapa 2 de B24 paró siempre en el paso 10


def names(n: int, rng: random.Random, taken: set[str]) -> list[str]:
    """Nombres de dos palabras con sílabas al azar; distintos entre sí y de `taken`."""
    word = lambda: "".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))).capitalize()
    out = []
    while len(out) < n:
        x = f"{word()} {word()}"
        if x not in taken:
            taken.add(x)
            out.append(x)
    return out


def bank(n: int | None = None, seed: str = "b24:banco") -> list[dict]:
    n, rng, taken = n or N_FACTS, random.Random(seed), set()
    heads, values, new_v, new_h = (names(n, rng, taken) for _ in range(4))
    facts = [{"propiedad": PROPS[i % len(PROPS)], "cabeza": h, "valor": v, "valor_nuevo": nv, "cabeza_nueva": nh}
             for i, (h, v, nv, nh) in enumerate(zip(heads, values, new_v, new_h))]  # fmt: skip
    return add_donors(facts, seed)


def add_donors(facts: list[dict], seed: str) -> list[dict]:
    for i, x in enumerate(facts):  # donantes de igual propiedad, como en B23: compiten en cada dirección
        same = [y for j, y in enumerate(facts) if j != i and y["propiedad"] == x["propiedad"]]
        pick = random.Random(f"{seed}:donantes:{i}").sample(same, K)
        x["donantes"], x["donantes_inv"] = [y["valor"] for y in pick], [y["cabeza"] for y in pick]
    return facts


def fwd(x: dict, value: str | None = None, head: str | None = None) -> tuple[str, str]:
    return f"The {x['propiedad']} of {head or x['cabeza']} is", f" {value or x['valor']}."


def rev(x: dict, head: str | None = None, value: str | None = None) -> tuple[str, str]:
    return f"{value or x['valor']} is the {x['propiedad']} of", f" {head or x['cabeza']}."


def losses(model, tok, facts: list[dict], new: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Pérdida del objetivo real y de los K donantes en cada dirección: dos [hechos, K + 1].
    Con new=True mide la cara no entrenada del hecho nuevo: la vuelta desde « {valor_nuevo}» y la ida desde « {cabeza_nueva}»."""
    ida = [fwd(x, v, x["cabeza_nueva"] if new else None) for x in facts for v in [x["valor"], *x["donantes"]]]
    vuelta = [rev(x, h, x["valor_nuevo"] if new else None) for x in facts for h in [x["cabeza"], *x["donantes_inv"]]]
    L = b17.def_losses(model, tok, ida + vuelta).reshape(2, len(facts), -1)  # -1: B25 filtra con más donantes
    return L[0], L[1]


def train(model, tok, items: list[tuple[str, str]], stop: float, max_steps: int, check: int, key: str, hook=None) -> tuple[int, list[dict]]:
    """AdamW como B23, de a lotes de BATCH; para cuando la pérdida media de `items` baja de `stop`. `hook(step)` corre tras cada paso."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=b17.LR, weight_decay=0.0)
    rng, t0, hist, step = random.Random(key), time.time(), [], 0
    for step in range(1, max_steps + 1):
        ids, mask, labels = b17.batch(tok, [items[i] for i in rng.sample(range(len(items)), min(b17.BATCH, len(items)))])
        with amp(DEVICE):
            model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if hook:
            hook(step)
        if step % check == 0:
            cur = float(b17.def_losses(model, tok, items).mean())
            hist.append({"paso": step, "perdida": round(cur, 4), "segundos": round(time.time() - t0)})
            if cur < stop:
                break
    del opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return step, hist


def main(path: str, model=None, tok=None, dose: bool = False, facts: list[dict] | None = None, doses: tuple | None = None,
         stop1: float | None = None) -> None:
    facts, doses, stop1 = facts or bank(), doses or DOSES, stop1 or STOP1
    res = json.load(open(path)) if os.path.exists(path) else {"corridas": {}}
    res["config"] = {"n_hechos": len(facts), "conjuntos": SETS, "n_sobrescritos": N_OVER, "k": K, "stop1": stop1, "max_pasos1": MAX_STEPS1,
                     "puerta1": GATE1, "lr": b17.LR, "batch": b17.BATCH, "stop2": b17.STOP_LOSS, "max_pasos2": b17.MAX_STEPS, "dosis": list(doses) if dose else None}  # fmt: skip
    res["banco"] = facts
    model, tok = b23.load(model, tok, True)
    torch.manual_seed(0)
    # shortcut: la etapa 1 no se guarda entre VMs; una VM nueva la repite con la misma semilla y suma uno a `repeticiones1`
    steps, hist = train(model, tok, [f(x) for x in facts for f in (fwd, rev)], stop1, MAX_STEPS1, CHECK1, "b24:etapa1")
    base_i, base_v = losses(model, tok, facts)
    win = {"ida": float(b23.wins(base_i).mean()), "vuelta": float(b23.wins(base_v).mean())}
    res["repeticiones1"] = res.get("repeticiones1", -1) + 1
    res["etapa1"] = {"pasos": steps, "historia": hist, "gana": win, "perdida_ida": base_i[:, 0].round(4).tolist(), "perdida_vuelta": base_v[:, 0].round(4).tolist()}
    print(json.dumps({"etapa1": {"pasos": steps, "gana": win, "perdida": hist[-1]["perdida"] if hist else None}}), flush=True)
    if min(win.values()) < GATE1:
        res["corte"] = f"E1: la etapa 1 no enseña las dos direcciones (gana {win})"
        json.dump(res, open(path, "w"), ensure_ascii=False)
        print(json.dumps({"corte": res["corte"]}), flush=True)
        return
    stage2(model, tok, facts, base_i, base_v, res, path, dose, doses=doses)


def stage2(model, tok, facts: list[dict], base_i: np.ndarray, base_v: np.ndarray, res: dict, path: str, dose: bool, tag: str = "b24",
           doses: tuple | None = None) -> None:
    """Brazos ida y vuelta por conjunto, cada uno desde el estado actual del modelo; guarda en res["corridas"] tras cada brazo."""
    doses = doses or DOSES
    state = {k: v.detach().to("cpu", copy=True) for k, v in [*model.named_parameters(), *model.named_buffers()]
             if not isinstance(v, torch.nn.Parameter) or v.requires_grad}  # fmt: skip
    n = len(facts)
    n_over = n // SETS  # si n no es múltiplo de SETS, el sobrante nunca se sobrescribe y queda en el resto
    order = random.Random(f"{tag}:conjuntos").sample(range(n), n)
    for s in range(SETS):
        over = sorted(order[s * n_over : (s + 1) * n_over])
        for arm in ARMS:
            keys = [f"s{s}|{arm}|d{d}" for d in doses] if dose else [f"s{s}|{arm}"]
            if keys[-1] in res["corridas"]:
                continue
            new = [fwd(facts[i], facts[i]["valor_nuevo"]) if arm == "ida" else rev(facts[i], facts[i]["cabeza_nueva"]) for i in over]

            def measure(step: int, key: str) -> None:
                L_i, L_v = losses(model, tok, facts)
                N_i, N_v = losses(model, tok, [facts[i] for i in over], new=True)
                res["corridas"][key] = r = {
                    "conjunto": s, "brazo": arm, "sobrescritos": over, "pasos": step,
                    "nuevos_final": float(b17.def_losses(model, tok, new).mean()),
                    "subida_ida": (L_i[:, 0] - base_i[:, 0]).round(4).tolist(), "subida_vuelta": (L_v[:, 0] - base_v[:, 0]).round(4).tolist(),
                    "gana_ida": b23.wins(L_i).astype(int).tolist(), "gana_vuelta": b23.wins(L_v).astype(int).tolist(),
                    "nuevo_otra_cara_gana": b23.wins(N_v if arm == "ida" else N_i).astype(int).tolist()}  # fmt: skip
                m = np.isin(np.arange(n), over)
                print(json.dumps({"corrida": key, "pasos": step, "nuevos": round(r["nuevos_final"], 3),
                                  **{f"{d}_sobre": round(float(np.array(r[f"subida_{d}"])[m].mean()), 3) for d in ARMS},
                                  **{f"{d}_resto": round(float(np.array(r[f"subida_{d}"])[~m].mean()), 3) for d in ARMS}}), flush=True)  # fmt: skip

            torch.manual_seed(1000 + s)
            stop, top = (float("-inf"), max(doses)) if dose else (b17.STOP_LOSS, b17.MAX_STEPS)
            hook = (lambda step: step in doses and measure(step, f"s{s}|{arm}|d{step}")) if dose else None
            steps, hist = train(model, tok, new, stop, top, b17.CHECK_EVERY, f"{tag}:lotes:{s}:{arm}", hook)
            if not dose:
                measure(steps, keys[0])
            for k in keys:
                res["corridas"][k]["historia"] = hist
            model.load_state_dict(state, strict=False)  # cada brazo parte de la etapa 1
            json.dump(res, open(path, "w"), ensure_ascii=False)
    print(json.dumps({"FIN": len(res["corridas"])}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], dose=sys.argv[2:] == ["dosis"])
