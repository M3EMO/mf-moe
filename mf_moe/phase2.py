"""Fase 2 del mapa (ticket 12): ¿el MoE fractal olvida menos a escala, con un MoE preentrenado?

Granite-3.1-1B-A400M ajusta cuatro dominios de texto en fila. Se entrenan las capas superiores (desde
`trainable_from`: expertos, router y atención) en fp32 con AdamW; el resto queda congelado en bf16. Las capas
entrenables crecen de 32 a 48 expertos. Cada lote lleva 5 % de texto general (repetición).

Olvido: loss media por token sobre la evaluación de cada dominio. `aprendido[d]` es la loss del dominio d
justo al terminar de entrenarlo (antes de fusionar); `olvido[d]` = loss final − aprendido[d] (positivo =
olvidó). Es un oráculo: no hay juez.

Condiciones (todas terminan con hasta 48 expertos por capa entrenable y top-k fijo):
- control: crece a 48 antes de empezar, clonando los expertos más usados en texto general;
- division: crece por duda del router (`Divider`) durante el entrenamiento;
- division+fusion_alineada: además fusiona pares alineados por permutación al cerrar cada dominio;
- division+padre_congelado: además congela al padre después de cada división (protege lo viejo).

Retome (decisión del usuario): el checkpoint guarda solo los pesos entrenables en fp32 y el estado del bucle;
al retomar, AdamW arranca con momentos en cero y vuelve a calentar el learning rate.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from mf_moe.division import Divider, resize, split
from mf_moe.lifelong import freeze_rows
from mf_moe.merging import Merger
from mf_moe.model import moe, routers
from mf_moe.resume import rng_state, set_rng_state

DOMAINS = ("matematica", "codigo", "biomedico", "legal")
GENERAL = "general"


@dataclass(frozen=True)
class Cond2:
    name: str
    grow_at_start: bool = False
    divide: bool = False
    merge_align: bool = False
    freeze_parent: bool = False


CONDS2 = {c.name: c for c in [
    Cond2("control", grow_at_start=True),
    Cond2("division", divide=True),
    Cond2("division+fusion_alineada", divide=True, merge_align=True),
    Cond2("division+padre_congelado", divide=True, freeze_parent=True),
]}  # fmt: skip


@dataclass
class Cfg2:
    domains: tuple[str, ...] = DOMAINS
    batch: int = 8
    micro: int = 1  # secuencias por pasada; piloto 3, control a 48: 22,88 GB con 2 y 21,01 GB con 1 (15 % más lento)
    lr: float = 2e-5
    warmup: int = 100
    replay: float = 0.05
    trainable_from: int = 12
    n_final: int = 48
    check_every: int = 250
    z_split: float = 3.5  # umbral de la división (el de la fase 1)
    save_every: int = 2000
    calib_seqs: int = 8  # secuencias de entrenamiento por dominio para la fusión
    grow_row: str = "clon"  # fila del router de cada clon del control: "clon" (la del padre) o "cero"
    extra: dict = field(default_factory=dict)


def load_data(root: str) -> dict[str, dict[str, np.ndarray]]:
    """{dominio: {"train": (n, seq) uint16, "eval": (m, seq) uint16}} desde `<root>/<dominio>_<parte>.npy`."""
    return {d: {p: np.load(Path(root, f"{d}_{p}.npy"), mmap_mode="r") for p in ("train", "eval")}
            for d in (*DOMAINS, GENERAL)}  # fmt: skip


def set_trainable(model, trainable_from: int) -> list[int]:
    """Capas desde `trainable_from` en fp32 y entrenables; el resto (embeddings incluidos) congelado en bf16.
    Convierte solo lo congelado: pasar las capas entrenables por bf16 redondearía lo aprendido."""
    layers = list(range(trainable_from, len(model.model.layers)))
    model.requires_grad_(False)
    for name, mod in model.named_children():
        if name != "model":
            mod.to(torch.bfloat16)  # lm_head (atado a los embeddings)
    for name, mod in model.model.named_children():
        if name != "layers":
            mod.to(torch.bfloat16)  # embeddings, norma final, rotary
    for i, layer in enumerate(model.model.layers):
        if i in layers:
            layer.float().requires_grad_(True)
        else:
            layer.to(torch.bfloat16)
    return layers


def amp(device: str):
    """Autocast bf16 en toda pasada del modelo: mezcla capas congeladas en bf16 con entrenables en fp32."""
    return torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16)


def _adamw(model, lr: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    # fused: la versión foreach copia todos los momentos en temporales y no entró en la L4 (piloto 2, 2026-10-17)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0, fused=params[0].is_cuda)


@torch.no_grad()
def eval_loss(model, seqs: np.ndarray, device: str, batch: int = 8) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(seqs), batch):
        ids = torch.as_tensor(np.asarray(seqs[i : i + batch], dtype=np.int64), device=device)
        with amp(device):
            loss = model(input_ids=ids, labels=ids).loss
        total += float(loss) * len(ids)
        n += len(ids)
    model.train()
    return total / n


def _grow_control(model, opt, layers: list[int], n_final: int, calib: torch.Tensor, device: str,
                  row: str = "clon") -> None:  # fmt: skip
    """Control: clona hasta `n_final` los expertos más usados en texto general, antes de entrenar.
    Con `row="cero"` el clon arranca con logit 0: el router lo elige poco hasta que lo aprende."""
    rs = routers(model)
    for r in rs:
        r.record = True
    with torch.no_grad(), amp(device):
        model(input_ids=calib)
    for i in layers:
        r = rs[i]
        usage = torch.bincount(r.last[2].reshape(-1), minlength=r.num_experts)
        order = usage.argsort(descending=True).tolist()
        k = 0
        while r.num_experts < n_final:
            child = split(model, opt, i, order[k % len(order)], row_dir=None)
            if row == "cero":
                r.weight.data[child] = 0
            k += 1
    for r in rs:
        r.record = False


def run2(cond: Cond2, seed: int, model, data: dict, device: str, cfg: Cfg2 | None = None,
         ckpt: str | None = None, log=print) -> dict:  # fmt: skip
    cfg = cfg or Cfg2()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    layers = set_trainable(model, cfg.trainable_from)
    key = f"fase2/{cond.name}/semilla{seed}/{'-'.join(cfg.domains)}"
    div = Divider(signal="duda", z=cfg.z_split, max_experts=cfg.n_final, every=cfg.check_every,
                  layers=tuple(layers)) \
        if cond.divide else None  # fmt: skip
    mer = Merger(criterion="salidas", align=True, min_experts=routers(model)[layers[0]].num_experts,
                 layers=tuple(layers)) if cond.merge_align else None  # fmt: skip
    frozen: dict[int, set[int]] = {i: set() for i in layers}
    L0, L, learned, start, step_in, done_steps = None, [], [], 0, 0, []
    opt = _adamw(model, cfg.lr)
    if ckpt and Path(ckpt).exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        if ck["key"] != key:
            raise ValueError(f"el checkpoint {ckpt} es de la corrida {ck['key']!r}, no de {key!r}")
        st = ck["state"]
        resize(model, st["expertos_por_capa"])  # los expertos nuevos nacen en fp32 y entrenables
        missing = model.load_state_dict(ck["model"], strict=False).unexpected_keys
        assert not missing, missing
        opt = _adamw(model, cfg.lr)  # momentos en cero: el retome vuelve a calentar el learning rate
        L0, L, learned = st["L_inicial"], st["L"], st["aprendido"]
        start, step_in, done_steps = st["dominio"], st["paso"], st["pasos"]
        frozen = {int(k): set(v) for k, v in st["congelados"].items()}
        set_rng_state(rng, st["rng"])
        if div:
            div.events, div.z_max = st["division"], st.get("z_max", [])
        if mer:
            mer.events = st["fusion"]
    elif cond.grow_at_start:
        calib = torch.as_tensor(np.asarray(data[GENERAL]["train"][: cfg.batch], dtype=np.int64), device=device)
        _grow_control(model, opt, layers, cfg.n_final, calib, device, cfg.grow_row)
    if div:
        div.attach(model)

    def refreeze(only: set[int] | None = None) -> None:
        """Vuelve a congelar los padres de esas capas: una división reemplazó sus Parameters (y sus hooks)."""
        for i, rows in frozen.items():
            if rows and (only is None or i in only):
                experts, router = moe(model, i)
                for p in (experts.gate_up_proj, experts.down_proj, router.weight):
                    freeze_rows(opt, p, sorted(rows))

    def save(d: int, s: int) -> None:
        trainable = {k: v.detach() for k, v in model.state_dict().items()
                     if any(k.startswith(f"model.layers.{i}.") for i in layers)}  # fmt: skip
        state = {"L_inicial": L0, "L": L, "aprendido": learned, "dominio": d, "paso": s, "pasos": done_steps, "rng": rng_state(rng),
                 "expertos_por_capa": [r.num_experts for r in routers(model)],
                 "congelados": {str(k): sorted(v) for k, v in frozen.items()},
                 "division": div.events if div else [], "z_max": div.z_max if div else [], "fusion": mer.events if mer else []}  # fmt: skip
        tmp = Path(ckpt + ".tmp")
        torch.save({"key": key, "model": trainable, "state": state}, tmp)
        tmp.replace(ckpt)
        Path(ckpt + ".step").write_text(f"{d}:{s}\n")

    refreeze()
    evals = {d: data[d]["eval"] for d in (*cfg.domains, GENERAL)}
    if L0 is None:  # loss de partida, antes de entrenar
        L0 = {d: eval_loss(model, x, device) for d, x in evals.items()}
    for di in range(start, len(cfg.domains)):
        dom = cfg.domains[di]
        train = data[dom]["train"]
        steps = len(train) // cfg.batch
        order = list(range(len(train)))
        random.Random(f"{seed}-{dom}").shuffle(order)
        s0 = step_in if di == start else 0
        if div:
            div._acc = []
        n_events = len(div.events) if div else 0
        for s in range(s0, steps):
            n_old = max(1, round(cfg.replay * cfg.batch))
            rows = [train[j] for j in order[s * cfg.batch : (s + 1) * cfg.batch - n_old]]
            gen = data[GENERAL]["train"]
            rows += [gen[rng.randrange(len(gen))] for _ in range(n_old)]
            ids = torch.as_tensor(np.stack(rows).astype(np.int64), device=device)
            lr = cfg.lr * min(1.0, (s - s0 + 1) / cfg.warmup)  # recalienta al empezar cada dominio y al retomar
            for g in opt.param_groups:
                g["lr"] = lr
            loss = 0.0
            for c in ids.split(cfg.micro):  # acumula el gradiente: mismo paso que con el lote entero
                with amp(device):
                    out = model(input_ids=c, labels=c)
                (out.loss * len(c) / len(ids)).backward()
                loss += float(out.loss.detach()) * len(c) / len(ids)
                if div:
                    div.observe(model, c, out.logits.detach(), c)
                del out
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if div:
                if (s + 1) % div.every == 0:
                    div.check(model, opt, s + 1)
                if len(div.events) > n_events:  # una división reemplazó parámetros: AdamW los siguió en el lugar
                    new = div.events[n_events:]
                    if cond.freeze_parent:
                        for e in new:
                            frozen[e["capa"]].add(e["padre"])
                    refreeze({e["capa"] for e in new})
                    n_events = len(div.events)
            if (s + 1) % 50 == 0:
                log({"dominio": dom, "paso": s + 1, "de": steps, "loss": round(loss, 4),
                     "expertos": [routers(model)[i].num_experts for i in layers]})  # fmt: skip
            if ckpt and (s + 1) % cfg.save_every == 0:
                save(di, s + 1)
        learned.append(eval_loss(model, evals[dom], device))
        if mer:
            calib = np.concatenate([data[d]["train"][: cfg.calib_seqs] for d in cfg.domains[: di + 1]])
            with amp(device):
                mer(model, opt, torch.as_tensor(calib.astype(np.int64), device=device), task=dom)
        L.append({d: eval_loss(model, x, device) for d, x in evals.items()})
        done_steps.append(steps)
        log({"fin_dominio": dom, "aprendido": round(learned[-1], 4), "L": {k: round(v, 4) for k, v in L[-1].items()}})
        if ckpt:
            save(di + 1, 0)
    last = L[-1]
    forget = {d: last[d] - learned[i] for i, d in enumerate(cfg.domains)}
    old = cfg.domains[:-1]
    return {"condicion": asdict(cond), "semilla": seed, "cfg": asdict(cfg), "L_inicial": L0, "L": L,
            "aprendido": learned,
            "olvido": forget, "olvido_medio": sum(forget[d] for d in old) / max(1, len(old)),
            "general_inicio_fin": [L0[GENERAL], last[GENERAL]], "pasos": done_steps,
            "expertos_finales": [routers(model)[i].num_experts for i in layers],
            # parte del uso (media móvil del router) que va a expertos nacidos en la corrida: con fila en cero,
            # un control que nunca los elige tendría en la práctica los expertos del preentrenado
            "z_max_revisiones": div.z_max if div else None,
            "uso_expertos_nuevos": [round(float(routers(model)[i].load[model.config.num_local_experts :].sum()), 4)
                                    for i in layers],  # fmt: skip
            "eventos_division": div.events if div else [], "eventos_fusion": mer.events if mer else []}  # fmt: skip

