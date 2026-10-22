"""Fase 2 del mapa (ticket 12): ¿el MoE fractal olvida menos a escala, con un MoE preentrenado?

Granite-3.1-1B-A400M ajusta cuatro dominios de texto en fila. Se entrenan las capas superiores (desde
`trainable_from`: expertos, router y atención) en fp32 con AdamW; el resto queda congelado en bf16. Las capas
entrenables crecen de 32 a 48 expertos. Cada lote lleva 5 % de texto general (repetición).

Olvido: loss media por token sobre la evaluación de cada dominio. `aprendido[d]` es la loss del dominio d
justo al terminar de entrenarlo (antes de fusionar); `olvido[d]` = loss final − aprendido[d] (positivo =
olvidó). Es un oráculo: no hay juez.

Condiciones (todas terminan con hasta 48 expertos por capa entrenable y top-k fijo):
- control: crece a 48 antes de empezar, clonando los expertos más usados en texto general;
- division: crece durante el entrenamiento, con un calendario fijo; la duda del router (`Divider`) elige al padre;
- division+fusion_alineada: además fusiona pares alineados por permutación al cerrar cada dominio;
- division+padre_congelado: además congela al padre después de cada división (protege lo viejo);
- azar+padre_congelado: como la anterior, pero el padre se elige al azar (¿aporta la duda o alcanza congelar?);
- division+consolidacion_uso: crece igual, pero al cerrar cada dominio congela los 8 expertos por capa que ese
  dominio más usó (el diagnóstico mostró que el olvido vive en los expertos que usa lo viejo).
Todo experto nuevo nace silencioso (sesgo -1) y el balanceo de los nacidos lo recluta (pilotos 5 y 6).

Retome (decisión del usuario): el checkpoint guarda solo los pesos entrenables en fp32 y el estado del bucle;
al retomar, AdamW arranca con momentos en cero y vuelve a calentar el learning rate.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from mf_moe.division import Divider, resize, split
from mf_moe.granite import load
from mf_moe.lifelong import freeze_rows
from mf_moe.merging import Merger
from mf_moe.model import moe, routers
from mf_moe.resume import rng_state, set_rng_state

DOMAINS = ("fines", "turco", "indonesio", "suajili")
GENERAL = "general"


@dataclass(frozen=True)
class Cond2:
    name: str
    grow_at_start: bool = False
    divide: bool = False
    merge_align: bool = False
    freeze_parent: bool = False
    random_parent: bool = False  # divide expertos al azar con el mismo calendario: separa la duda del congelado
    consolidate: int = 0  # al cerrar cada dominio congela los k expertos que más usó (diagnóstico: el olvido vive ahí)
    consolidate_random: bool = False  # congela tantos expertos nuevos como el uso, pero al azar: ¿importa cuáles?


CONDS2 = {c.name: c for c in [
    Cond2("control", grow_at_start=True),
    Cond2("division", divide=True),
    Cond2("division+fusion_alineada", divide=True, merge_align=True),
    Cond2("division+padre_congelado", divide=True, freeze_parent=True),
    Cond2("azar+padre_congelado", divide=True, freeze_parent=True, random_parent=True),
    Cond2("division+consolidacion_uso", divide=True, consolidate=8),
    Cond2("control+consolidacion_uso", grow_at_start=True, consolidate=8),  # ¿hace falta dividir?
    Cond2("control+consolidacion_azar", grow_at_start=True, consolidate=8, consolidate_random=True),
]}  # fmt: skip


@dataclass
class Cfg2:
    domains: tuple[str, ...] = DOMAINS
    batch: int = 8
    observe_seqs: int = 8  # secuencias por paso que alimentan la señal de duda (piloto 5: con las 64, la división tardaba el doble)
    micro: int = 1  # secuencias por pasada; piloto 3, control a 48: 22,88 GB con 2 y 21,01 GB con 1 (15 % más lento)
    lr: float = 2e-5
    warmup: int = 100
    replay: float = 0.05
    trainable_from: int = 12
    n_final: int = 48
    check_every: int = 250
    z_split: float = 3.5  # umbral de la división (el de la fase 1)
    split_mix: float = 1.0  # fila del router del hijo: 1 = dirección de la duda (fase 1); ver `division.split`
    silent: bool = False  # hijos y clones nacen sin que el router los elija; ver `division.split`
    balance_u: float = 0.0  # paso del sesgo de balanceo de los nacidos, por pasada; 0 = ruteo de Granite
    load_decay: float = 0.999  # media móvil del uso por pasada; el balanceo se pasa de largo ~u/(1-decay)
    save_every: int = 2000
    calib_seqs: int = 8  # secuencias de entrenamiento por dominio para la fusión
    grow_row: str = "clon"  # clones del control: "clon" (fila del padre), "cero" (fila en cero) o "silencioso"
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


def write_parts(path: str, tag: str, size: int = 64 << 20, keep: int = 2) -> None:
    """Parte el checkpoint en `<path>.partes.<tag>/NNN` con un `info` ("n bytes") al final; el runner baja las
    partes. Bajar 3,6 GB de una vez carga el archivo en la RAM local (OOM, 2026-10-19), y partirlo con otro
    `exec` quedaría en cola detrás del job. Cada checkpoint tiene su carpeta y quedan las `keep` últimas: una
    bajada lenta (hasta 3 h) no mezcla partes de dos checkpoints del mismo tamaño."""
    base = Path(path).parent
    out = base / f"{Path(path).name}.partes.{tag}"
    out.mkdir(exist_ok=True)
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(size):
            (out / f"{n:03d}").write_bytes(chunk)
            n += 1
    (out / "info").write_text(f"{n} {Path(path).stat().st_size}\n")
    sets = sorted(base.glob(f"{Path(path).name}.partes.*"), key=lambda p: p.stat().st_mtime)
    for old in sets[:-keep]:
        for f in old.iterdir():
            f.unlink()
        old.rmdir()


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
            child = split(model, opt, i, order[k % len(order)], row_dir=None, silent=row == "silencioso")
            if row == "cero":
                r.weight.data[child] = 0
            k += 1
    for r in rs:
        r.record = False


@torch.no_grad()
def _most_used(model, layers: list[int], seqs: np.ndarray, device: str, k: int) -> dict[int, set[int]]:
    """Los k expertos de cada capa que el router más elige en `seqs` (los que guardan ese dominio)."""
    rs = routers(model)
    was = [r.record for r in rs]  # la división tiene el registro prendido y lo necesita después
    for r in rs:
        r.record = True
    model.eval()
    with amp(device):
        model.model(input_ids=torch.as_tensor(np.asarray(seqs, dtype=np.int64), device=device))
    model.train()
    out = {i: set(torch.bincount(rs[i].last[2].reshape(-1), minlength=rs[i].num_experts).topk(min(k, rs[i].num_experts)).indices.tolist())
           for i in layers}  # fmt: skip
    for r, w in zip(rs, was):
        r.record = w
    return out


def run2(cond: Cond2, seed: int, model, data: dict, device: str, cfg: Cfg2 | None = None,
         ckpt: str | None = None, log=print) -> dict:  # fmt: skip
    cfg = cfg or Cfg2()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    layers = set_trainable(model, cfg.trainable_from)
    for i in layers:
        routers(model)[i].u, routers(model)[i].decay = cfg.balance_u, cfg.load_decay
    key = f"fase2/{cond.name}/semilla{seed}/{'-'.join(cfg.domains)}"
    div = Divider(signal="duda", z=cfg.z_split, mix=cfg.split_mix, silent=cfg.silent,
                  random_parent=seed if cond.random_parent else None, max_experts=cfg.n_final, every=cfg.check_every,
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
        for r, n in zip(routers(model), st.get("nacidos_desde", [])):
            r.n_born = n
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
        calib = torch.as_tensor(np.asarray(data[GENERAL]["train"][: cfg.calib_seqs], dtype=np.int64), device=device)
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
                 "nacidos_desde": [r.n_born for r in routers(model)],
                 "congelados": {str(k): sorted(v) for k, v in frozen.items()},
                 "division": div.events if div else [], "z_max": div.z_max if div else [], "fusion": mer.events if mer else []}  # fmt: skip
        tmp = Path(ckpt + ".tmp")
        torch.save({"key": key, "model": trainable, "state": state}, tmp)
        tmp.replace(ckpt)
        write_parts(ckpt, f"{d}-{s}")
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
            for k, c in enumerate(ids.split(cfg.micro)):  # acumula el gradiente: mismo paso que con el lote entero
                with amp(device):
                    out = model(input_ids=c, labels=c)
                (out.loss * len(c) / len(ids)).backward()
                loss += float(out.loss.detach()) * len(c) / len(ids)
                if div and k * cfg.micro < cfg.observe_seqs:
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
        if cond.consolidate and di < len(cfg.domains) - 1:
            for i, top in _most_used(model, layers, data[dom]["train"][: 2 * cfg.calib_seqs], device, cond.consolidate).items():
                if cond.consolidate_random:  # misma cantidad de congelados nuevos que el uso, elegidos sin mirarlo
                    free = sorted(set(range(routers(model)[i].num_experts)) - frozen[i])
                    top = set(random.Random(f"{seed}/{di}/{i}").sample(free, len(top - frozen[i])))
                frozen[i] |= top
            refreeze()
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
            "z_max_revisiones": div.z_max if div else None,
            # parte del uso (media móvil del router) que va a expertos nacidos en la corrida: un control cuyos
            # clones nunca entran al ruteo tendría en la práctica los expertos del preentrenado
            "uso_expertos_nuevos": [round(float(routers(model)[i].load[model.config.num_local_experts :].sum()), 4)
                                    for i in layers],  # fmt: skip
            "congelados": {str(k): sorted(v) for k, v in frozen.items()},
            "eventos_division": div.events if div else [], "eventos_fusion": mer.events if mer else []}  # fmt: skip



# Las 15 corridas (ticket 12): 4 idiomas × ~10 M tokens, 152 pasos de 64 por dominio. Lote y lr del piloto 5; 16
# divisiones por capa repartidas en los 4 dominios (612 pasos / 16 = 38); el balanceo recluta a un nacido en
# ~1/4 de dominio (1 / (0,25 · 152 · 64) = 4e-4). Un checkpoint al cerrar cada dominio.
PRODUCTION = {"batch": 64, "lr": 6e-5, "warmup": 15, "check_every": 38, "z_split": -1e9, "split_mix": 0.1,
              "silent": True, "balance_u": 4e-4, "load_decay": 0.99, "grow_row": "silencioso", "save_every": 10**9}
EVAL_SEQS = 500


def main(data_dir: str, out: str, cond: str, seed: int, ckpt: str | None = None) -> None:
    data = {d: {"train": v["train"], "eval": v["eval"][:EVAL_SEQS]} for d, v in load_data(data_dir).items()}
    res = run2(CONDS2[cond], seed, load(attention_dropout=0.0), data, "cuda", Cfg2(**PRODUCTION), ckpt=ckpt,
               log=lambda r: print(json.dumps(r), flush=True))  # fmt: skip
    res["datos"] = json.loads(Path(data_dir, "datos.json").read_text())  # sha256 de cada dominio
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "result.json").write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), *sys.argv[5:6])
