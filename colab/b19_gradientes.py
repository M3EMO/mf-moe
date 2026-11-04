"""B19 (ticket 15): ¿el mapa de gradientes explica el daño reproducible que árbol, estado y router dejaron sin explicar?

B18: ~38 % de la variación por conjunto se repite y los predictores explican ~1,8 %. El olvido se predice con el
kernel cruzado entre tareas en modelos con la base congelada, y en LLMs es una pregunta abierta («Catastrophic
Forgetting is Low-Rank», CATS @ ICML 2026). Mismo banco y áreas de B18. Para cada miembro del área, tres medidas:
- ntk: −⟨∇L_c, ∇L_entrenamiento⟩ en los pesos iniciales: se mide antes de entrenar;
- primer_orden: ⟨∇L_c, Δθ⟩ con el cambio real de pesos, en total y por bloque (expertos, router, resto): control
  positivo de que el daño se lee en el espacio de pesos;
- uso_def: coseno entre el uso de expertos de la definición del concepto y el de los hechos nuevos, medio sobre las
  capas: la prueba justa del router, que B18 midió solo en el último token del sujeto.
Uso: .venv/bin/python -m colab.b19_gradientes SALIDA.json [areas separadas por coma]
"""

import sys

import numpy as np
import torch

from colab import b17_conflicto as b17
from colab import b18_mismo_concepto as b18
from colab.b16_conceptos import DEVICE
from mf_moe.model import routers
from mf_moe.phase2 import amp

N_SETS = 5  # más la repetición: B18 ya mostró que 6 conjuntos dan IC angostos


def block(name: str) -> str:
    return "expertos" if ".experts." in name else "router" if ".router." in name else "resto"


def usage(model, tok, items: list[tuple[str, str]]) -> np.ndarray:
    """Fracción de los tokens que predicen la definición que va a cada experto: [oraciones, capas, expertos]."""
    rs = routers(model)
    model.eval()
    for r in rs:
        r.record = True
    out = []
    with torch.no_grad():
        for prefix, rest in items:
            n, ids = len(tok(prefix)["input_ids"]), tok(prefix + rest)["input_ids"]
            with amp(DEVICE):
                model(input_ids=torch.tensor(ids)[None].to(DEVICE))
            out.append([torch.bincount(r.last[2][n - 1 : -1].reshape(-1), minlength=r.num_experts).float().cpu().numpy() for r in rs])
    for r in rs:
        r.record = False
    model.train()
    U = np.array(out)
    return U / U.sum(-1, keepdims=True)


class Probe:
    def __init__(self, model, tok, items: list[tuple[str, str]], names: list[str]):
        self.items, self.names = items, names
        self.U = usage(model, tok, items)

    def _backward(self, model, tok, items, weight: float) -> None:
        ids, mask, labels = b17.batch(tok, items)
        with amp(DEVICE):
            (model(input_ids=ids, attention_mask=mask, labels=labels).loss * weight).backward()

    def pre(self, model, tok, new, keep) -> None:
        """Gradiente de entrenamiento en los pesos iniciales (mitad hechos nuevos, mitad anclas, como cada paso)."""
        model.eval()
        model.zero_grad(set_to_none=True)
        sample = keep[: len(new)]
        chunks = [new[i : i + b17.BATCH] for i in range(0, len(new), b17.BATCH)]
        chunks2 = [sample[i : i + b17.BATCH] for i in range(0, len(sample), b17.BATCH)]
        for c in chunks:
            self._backward(model, tok, c, 0.5 / len(chunks))
        for c in chunks2:
            self._backward(model, tok, c, 0.5 / len(chunks2))
        self.g_train = {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        model.zero_grad(set_to_none=True)
        U_new = usage(model, tok, new).mean(0)  # [capas, expertos]
        num = (self.U * U_new[None]).sum(-1)
        den = np.linalg.norm(self.U, axis=-1) * np.linalg.norm(U_new, axis=-1)[None]
        self.uso = (num / den).mean(-1)
        model.train()

    def post(self, model, state: dict) -> None:
        """Cambio de pesos del entrenamiento, antes de restaurar."""
        self.delta = {n: p.detach() - state[n].to(p.device) for n, p in model.named_parameters() if p.requires_grad}

    def final(self, model, tok) -> dict:
        """Productos por concepto, con el modelo ya restaurado a los pesos iniciales."""
        model.eval()
        g_train = {n: g.to(DEVICE) for n, g in self.g_train.items()}
        first, ntk, blocks = [], [], {"expertos": [], "router": [], "resto": []}
        for item in self.items:
            model.zero_grad(set_to_none=True)
            self._backward(model, tok, [item], 1.0)
            per = dict.fromkeys(blocks, 0.0)
            k = 0.0
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    per[block(n)] += float((p.grad.float() * self.delta[n].float()).sum())
                    if n in g_train:
                        k -= float((p.grad.float() * g_train[n].float()).sum())
            first.append(sum(per.values()))
            ntk.append(k)
            for b, v in per.items():
                blocks[b].append(v)
        model.zero_grad(set_to_none=True)
        del g_train
        self.delta, self.g_train = None, None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        model.train()
        return {"evaluados_gradiente": self.names, "primer_orden": first, "ntk": ntk, "uso_def": self.uso.round(5).tolist(), "primer_orden_bloques": blocks}


def make_probe(model, tok, wn, area: str, ev) -> Probe:
    members = {x.name() for x in wn.synset(area).closure(lambda x: x.hyponyms())}
    inside = [x for x in ev if x.name() in members]
    return Probe(model, tok, [b18.fact(x, x.definition()) for x in inside], [x.name() for x in inside])


def main(path: str, areas: str = ",".join(b18.AREAS), model=None, tok=None) -> None:
    b18.N_SETS = N_SETS
    b18.main(path, areas, model=model, tok=tok, make_probe=make_probe)


if __name__ == "__main__":
    main(*sys.argv[1:3])
