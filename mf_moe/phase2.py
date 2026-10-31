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
- control+emo: el control con el ruteo de EMO (arXiv 2605.06663): cada documento rutea dentro de un grupo de
  expertos. Logra modularidad sin crecer; si olvida poco, el MoE fractal tiene que ganarle.
Todo experto nuevo nace silencioso (sesgo -1) y el balanceo de los nacidos lo recluta (pilotos 5 y 6).

Retome (decisión del usuario): el checkpoint guarda solo los pesos entrenables en fp32 y el estado del bucle;
al retomar, AdamW arranca con momentos en cero y vuelve a calentar el learning rate.
"""

from __future__ import annotations

import json
import os
import subprocess
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from mf_moe import fractal_tree as ft
from mf_moe.division import Divider, resize, split
from mf_moe.expert_tree import fractal
from mf_moe.granite import RecordingGraniteRouter, load
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
    consolidate_tfidf: bool = False  # B8: elige los k por uso_dominio / uso_general (Meta) en vez de uso crudo
    tree: bool = False  # árbol fractal (ticket 14): mitosis con hija de bajo rango; la hija vieja queda congelada
    lowrank_growth: bool = False  # V1: los clones del control son copias fijas + U·V, como las hojas del árbol
    max_depth: int = 0  # H10: profundidad máxima de las hojas que hacen mitosis (0 = sin límite; 1 = árbol de un nivel)
    emo: int = 0  # EMO: tamaño mínimo del grupo de expertos de cada documento, d ~ U{emo..n} (0 = apagado); ver `EmoRouter`


CONDS2 = {c.name: c for c in [
    Cond2("control", grow_at_start=True),
    Cond2("division", divide=True),
    Cond2("division+fusion_alineada", divide=True, merge_align=True),
    Cond2("division+padre_congelado", divide=True, freeze_parent=True),
    Cond2("azar+padre_congelado", divide=True, freeze_parent=True, random_parent=True),
    Cond2("division+consolidacion_uso", divide=True, consolidate=8),
    Cond2("control+consolidacion_uso", grow_at_start=True, consolidate=8),  # ¿hace falta dividir?
    Cond2("control+consolidacion_azar", grow_at_start=True, consolidate=8, consolidate_random=True),
    # B8, sparse memory finetuning (arXiv 2510.15103): congela lo propio del dominio, no lo que también usa el inglés
    Cond2("control+consolidacion_tfidf", grow_at_start=True, consolidate=8, consolidate_tfidf=True),
    # ticket 14, etapa C: el calendario dice cuándo (mismo presupuesto, 16 mitosis por capa) y la duda dice dónde,
    # en cualquier profundidad; un umbral absoluto nunca dispara (z > 3 en 0 de 192 revisiones de la fase 2)
    Cond2("arbol+consolidacion", divide=True, tree=True, consolidate=8),
    Cond2("arbol_azar+consolidacion", divide=True, tree=True, random_parent=True, consolidate=8),
    # V1 de la spec: mismos parámetros nuevos que el árbol (16 deltas de bajo rango por capa), ruteo plano
    Cond2("control_bajo_rango+consolidacion", grow_at_start=True, lowrank_growth=True, consolidate=8),
    # H10 de la spec: las hijas no vuelven a dividirse; mismo calendario de mitosis. Aísla la recursión (lo fractal)
    Cond2("arbol_un_nivel+consolidacion", divide=True, tree=True, consolidate=8, max_depth=1),
    # rival sin crecimiento: modularidad por documento con los mismos 48 expertos; d desde el top-k de Granite (8)
    Cond2("control+emo", grow_at_start=True, emo=8),
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
    lowrank_lr_mult: float = 1.0  # V8: el lr de los deltas U·V es este múltiplo del lr base
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


def drive_sync(ckpt: str, step: str) -> subprocess.Popen | None:
    """Con MFMOE_DRIVE (remoto de rclone, ver colab/phase2_vm.py) copia el checkpoint a Drive en segundo plano: un
    retiro de la VM ya no pierde el dominio mientras la PC baja 3,5 GB. El marcador va después del archivo y lleva
    el paso de este guardado, no el que haya en disco cuando termine la copia. Un lock serializa dos copias."""
    drive = os.environ.get("MFMOE_DRIVE")
    if not drive:
        return None
    cmd = f'timeout 3600 rclone copyto "{ckpt}" "{drive}/ckpt.pt" && printf "{step}\\n" | timeout 300 rclone rcat "{drive}/ckpt.pt.step"'
    try:
        return subprocess.Popen(["flock", ckpt + ".lock", "sh", "-c", cmd])
    except OSError as e:  # sin flock la corrida sigue: Drive es un respaldo, nunca un motivo para cortar
        print(json.dumps({"drive": f"no se pudo copiar: {e}"}), flush=True)
        return None


def _adamw(model, lr: float, lowrank_mult: float = 1.0) -> torch.optim.Optimizer:
    """Dos grupos: los deltas U·V (`lowrank`, V8) con su múltiplo del lr y el resto. El bucle fija `lr · mult`."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": [p for n, p in named if ".lowrank." not in n], "mult": 1.0},
              {"params": [p for n, p in named if ".lowrank." in n], "mult": lowrank_mult}]  # fmt: skip
    # fused: la versión foreach copia todos los momentos en temporales y no entró en la L4 (piloto 2, 2026-10-17)
    return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr, weight_decay=0.0, fused=named[0][1].is_cuda)


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
def _most_used(model, layers: list[int], seqs: np.ndarray, device: str, k: int,
               background: np.ndarray | None = None) -> dict[int, set[int]]:  # fmt: skip
    """Los k expertos de cada capa que el router más elige en `seqs` (los que guardan ese dominio).
    Con `background` (texto general), ordena por uso_dominio / uso_general: ver `_tfidf`."""
    rs = routers(model)
    was = [r.record for r in rs]  # la división tiene el registro prendido y lo necesita después
    for r in rs:
        r.record = True
    model.eval()

    def usage(x: np.ndarray) -> dict[int, torch.Tensor]:
        with amp(device):
            model.model(input_ids=torch.as_tensor(np.asarray(x, dtype=np.int64), device=device))
        return {i: torch.bincount(rs[i].last[2].reshape(-1), minlength=rs[i].num_experts) for i in layers}

    score = usage(seqs)
    if background is not None:
        score = {i: _tfidf(u, b) for (i, u), b in zip(score.items(), usage(background).values())}
    model.train()
    out = {i: set(u.topk(min(k, len(u))).indices.tolist()) for i, u in score.items()}
    for r, w in zip(rs, was):
        r.record = w
    return out


def _tfidf(usage: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """Puntaje de cada experto a la Meta (arXiv 2510.15103): cuánto más lo usa el dominio que el corpus de fondo.
    Compara fracciones de uso (los dos textos pueden rutear distinta cantidad de tokens); un experto que el
    inglés también usa mucho baja. El 1e-3 evita dividir por cero en los expertos que el fondo no elige."""
    return (usage.float() / usage.sum() + 1e-3) / (background.float() / background.sum() + 1e-3)


DRIFT_SEQS, REPORT_SEQS, PER_DOC = 16, 128, 8  # PER_DOC: secuencias de evaluación por documento (prepare_phase2)


@torch.no_grad()
def _route(model, layers: list[int], seqs: np.ndarray, device: str, cast: bool = True) -> torch.Tensor:
    """Hojas elegidas (capas, tokens, k) en modo eval, sin tocar el estado de los routers."""
    rs = routers(model)
    was, training = [r.record for r in rs], model.training
    for r in rs:
        r.record = True
    model.eval()
    ids = torch.as_tensor(np.asarray(seqs, dtype=np.int64), device=device)
    if cast:
        with amp(device):
            model.model(input_ids=ids)
    else:
        model.model(input_ids=ids)
    out = torch.stack([rs[i].last[2] for i in layers])
    for r, w in zip(rs, was):
        r.record = w
    model.train(training)
    return out


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Coincidencia media de los conjuntos top-k de cada token."""
    inter = (a[..., :, None] == b[..., None, :]).any(-1).sum(-1).float()
    return round(float((inter / (2 * a.shape[-1] - inter)).mean()), 4)


def _trainable_expert_params(model, layers: list[int], frozen: dict[int, set[int]]) -> int:
    """V1: parámetros de expertos que siguen entrenables al final (filas libres más deltas U·V libres)."""
    total = 0
    for i in layers:
        experts, _ = moe(model, i)
        lr = getattr(experts, "lowrank", None)
        fixed = frozen[i] | (set(lr.rows.tolist()) if lr is not None else set())
        per_row = sum(getattr(experts, n)[0].numel() for n in ft.NAMES)
        total += per_row * (experts.num_experts - len(fixed))
        if lr is not None:
            per_delta = sum(getattr(lr, f"{a}_{n}")[0].numel() for n in ft.NAMES for a in "uv") if len(lr.rows) else 0
            total += per_delta * sum(1 for r in lr.rows.tolist() if r not in frozen[i])
    return total


def _routing_report(model, layers: list[int], evals: dict, snaps: dict, device: str,
                    frozen: dict[int, set[int]] | None = None) -> dict:  # fmt: skip
    """Medidas del ticket 14 al final de la corrida. Nunca tira la corrida: un error queda en el resultado."""
    try:
        doms = list(evals)
        n = max(routers(model)[i].num_experts for i in layers)
        U = np.zeros((len(layers), len(doms) * (REPORT_SEQS // PER_DOC), n), dtype=np.int64)
        for d, dom in enumerate(doms):
            for j in range(0, min(REPORT_SEQS, len(evals[dom])), PER_DOC):
                idx = _route(model, layers, evals[dom][j : j + PER_DOC], device).reshape(len(layers), -1).cpu()
                for li in range(len(layers)):
                    U[li, d * (REPORT_SEQS // PER_DOC) + j // PER_DOC] += np.bincount(idx[li].numpy(), minlength=n)
        tot = U.sum(1)
        live = [routers(model)[i].num_experts for i in layers]
        ent, dead = [], []
        for li, m in enumerate(live):
            p = tot[li, :m] / tot[li, :m].sum()
            ent.append(round(float(-(p * np.log(p + 1e-30)).sum() / np.log(m)), 4))
            dead.append(round(float((p < 0.01 / m).mean()), 4))
        dom_of = np.repeat(np.arange(len(doms)), REPORT_SEQS // PER_DOC)
        frac = fractal(U, [dom_of, np.arange(len(dom_of))], n_null=10)
        x = evals[GENERAL][:8]
        rs = routers(model)
        keep = [{k: v.clone() for k, v in r.named_buffers()} for r in rs]
        model.train()
        with torch.no_grad(), amp(device):
            for r in rs:
                r.record = True
            model.model(input_ids=torch.as_tensor(np.asarray(x, dtype=np.int64), device=device))
            in_train = torch.stack([rs[i].last[2] for i in layers])
        for r, b in zip(rs, keep):  # el modo train actualiza el balanceo: se deshace
            for k, v in b.items():
                setattr(r, k, v)
        now = {d: _route(model, layers, evals[d][:DRIFT_SEQS], device).cpu() for d in snaps}
        drift = {d: _jaccard(now[d], s.long()) for d, s in snaps.items()}
        frozen = frozen or {i: set() for i in layers}
        to_frozen = lambda idx: round(float(np.mean([np.isin(idx[li].numpy(), sorted(frozen[i])).mean()  # noqa: E731
                                                     for li, i in enumerate(layers)])), 4)  # fmt: skip
        return {"entropia_uso": ent, "hojas_muertas": dead,
                "fractal": [{k: f[k] for k in ("exceso", "autosimilitud")} for f in frac],
                "train_vs_eval": _jaccard(in_train, _route(model, layers, x, device)),
                "bf16_vs_fp32": _jaccard(_route(model, layers, x, device), _route(model, layers, x, device, cast=False)),
                "deriva": drift,
                # R5: parte de los tokens de cada dominio terminado que va a expertos congelados, al cerrarlo y al final
                "uso_congelados": {d: [to_frozen(s.long()), to_frozen(now[d])] for d, s in snaps.items()},
                "forma": [ft.shape(model, i) for i in layers]}  # fmt: skip
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}


class EmoRouter(RecordingGraniteRouter):
    """EMO (arXiv 2605.06663; `v2/emo_router.py` de OLMo-core): suma las probabilidades de ruteo de los tokens de
    cada documento, toma sus top-d expertos como grupo y elige el top-k de cada token dentro del grupo. En
    entrenamiento d ~ U{d_min..n} por documento y por pasada; en evaluación d = (d_min + n) // 2, como la referencia.
    `d_min` se acota a [top-k, n]; con d = n el ruteo es el del control. Un documento termina en EOS y nunca cruza
    de secuencia (como los `segment_ids` de la referencia). El balanceo ya es global (media móvil del uso, nunca por
    lote ni por documento): es el mismo del control, sin la pérdida auxiliar de la referencia."""

    d_min: int
    seg: torch.Tensor  # (tokens,) documento de cada token del forward en curso; lo fija el pre-hook de `install_emo`

    def restrict(self, probs: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        n = probs.shape[-1]
        lo = max(self.top_k, min(self.d_min, n))
        if lo >= n:  # grupo = todos: ni siquiera sortea, así el ruteo y el RNG quedan como en el control
            return scores
        n_doc = int(self.seg.max()) + 1
        doc = probs.new_zeros(n_doc, n).index_add_(0, self.seg, probs.detach())
        if self.training:
            d = torch.randint(lo, n + 1, (n_doc, 1), device=probs.device)
        else:
            d = torch.full((n_doc, 1), (lo + n) // 2, device=probs.device)
        keep = doc.argsort(-1, descending=True).argsort(-1) < d  # puesto de cada experto en su documento
        return scores.masked_fill(~keep[self.seg], float("-inf"))


def install_emo(model, d_min: int) -> None:
    """Pasa los routers a `EmoRouter` sin tocar sus pesos ni buffers, y engancha el corte en documentos."""
    eos = model.config.eos_token_id

    def segments(_, args, kwargs):
        ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
        end = (ids == eos).long()
        seg = end.cumsum(1) - end + torch.arange(len(ids), device=ids.device)[:, None] * ids.shape[1]  # el EOS cierra su documento
        seg = torch.unique(seg.reshape(-1), return_inverse=True)[1]
        for r in routers(model):
            r.seg = seg

    model.model.register_forward_pre_hook(segments, with_kwargs=True)
    for r in routers(model):
        r.__class__, r.d_min = EmoRouter, d_min


def run2(cond: Cond2, seed: int, model, data: dict, device: str, cfg: Cfg2 | None = None,
         ckpt: str | None = None, log=print) -> dict:  # fmt: skip
    cfg = cfg or Cfg2()
    torch.manual_seed(seed)
    if cond.tree:
        ft.install(model)
    elif cond.lowrank_growth:
        ft.install_lowrank(model)
    if cond.emo:
        install_emo(model, cond.emo)
    rng = random.Random(seed)
    layers = set_trainable(model, cfg.trainable_from)
    for i in layers:
        routers(model)[i].u, routers(model)[i].decay = cfg.balance_u, cfg.load_decay
    key = f"fase2/{cond.name}/semilla{seed}/{'-'.join(cfg.domains)}"
    div = Divider(signal="duda", z=cfg.z_split, mix=cfg.split_mix, silent=cfg.silent,
                  random_parent=seed if cond.random_parent else None, max_experts=cfg.n_final, every=cfg.check_every,
                  splitter=ft.split if cond.tree else None, eligible=ft.below(cond.max_depth) if cond.max_depth else None,
                  mass=cond.tree,
                  layers=tuple(layers)) \
        if cond.divide else None  # fmt: skip
    mer = Merger(criterion="salidas", align=True, min_experts=routers(model)[layers[0]].num_experts,
                 layers=tuple(layers)) if cond.merge_align else None  # fmt: skip
    frozen: dict[int, set[int]] = {i: set() for i in layers}
    snaps: dict[str, torch.Tensor] = {}  # ruteo de cada dominio terminado al cerrarlo: mide la deriva
    L0, L, learned, start, step_in, done_steps = None, [], [], 0, 0, []
    curve, clock, resumes = [], [0.0, 0], 0  # M5: pérdida media cada 50 pasos; V2: segundos y pasos; V9
    opt = _adamw(model, cfg.lr, cfg.lowrank_lr_mult)
    if ckpt and Path(ckpt).exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        if ck["key"] != key:
            raise ValueError(f"el checkpoint {ckpt} es de la corrida {ck['key']!r}, no de {key!r}")
        st = ck["state"]
        (ft.resize if cond.tree else resize)(model, st["expertos_por_capa"])  # los expertos nuevos nacen en fp32 y entrenables
        if cond.lowrank_growth:
            scratch = torch.optim.SGD(model.parameters(), lr=0.0)
            for i in layers:
                ft.to_lowrank(model, scratch, i, range(model.config.num_local_experts, routers(model)[i].num_experts))
        for r, n in zip(routers(model), st.get("nacidos_desde", [])):
            r.n_born = n
        missing = model.load_state_dict(ck["model"], strict=False).unexpected_keys
        assert not missing, missing
        opt = _adamw(model, cfg.lr, cfg.lowrank_lr_mult)  # momentos en cero: el retome vuelve a calentar el lr
        L0, L, learned = st["L_inicial"], st["L"], st["aprendido"]
        start, step_in, done_steps = st["dominio"], st["paso"], st["pasos"]
        frozen = {int(k): set(v) for k, v in st["congelados"].items()}
        snaps = st.get("deriva", {})
        curve, clock, resumes = st.get("curva", []), st.get("reloj", [0.0, 0]), st.get("retomes", 0) + 1
        set_rng_state(rng, st["rng"])
        if div:
            div.events, div.z_max = st["division"], st.get("z_max", [])
        if mer:
            mer.events = st["fusion"]
    elif cond.grow_at_start:
        calib = torch.as_tensor(np.asarray(data[GENERAL]["train"][: cfg.calib_seqs], dtype=np.int64), device=device)
        _grow_control(model, opt, layers, cfg.n_final, calib, device, cfg.grow_row)
        if cond.lowrank_growth:
            for i in layers:
                ft.to_lowrank(model, opt, i, range(model.config.num_local_experts, routers(model)[i].num_experts))
    if div:
        div.attach(model)

    def refreeze(only: set[int] | None = None) -> None:
        """Vuelve a congelar los padres de esas capas: una división reemplazó sus Parameters (y sus hooks)."""
        for i, rows in frozen.items():
            if (cond.tree or cond.lowrank_growth) and (only is None or i in only):  # y las copias base de U·V
                ft.freeze(model, opt, i, rows)
            elif rows and (only is None or i in only):
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
                 "division": div.events if div else [], "z_max": div.z_max if div else [], "fusion": mer.events if mer else [],
                 "deriva": snaps, "curva": curve, "reloj": clock, "retomes": resumes}  # fmt: skip
        tmp = Path(ckpt + ".tmp")
        torch.save({"key": key, "model": trainable, "state": state}, tmp)
        tmp.replace(ckpt)
        write_parts(ckpt, f"{d}-{s}")
        Path(ckpt + ".step").write_text(f"{d}:{s}\n")
        drive_sync(ckpt, f"{d}:{s}")

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
        window = []
        for s in range(s0, steps):
            tic = time.perf_counter()
            n_old = max(1, round(cfg.replay * cfg.batch))
            rows = [train[j] for j in order[s * cfg.batch : (s + 1) * cfg.batch - n_old]]
            gen = data[GENERAL]["train"]
            rows += [gen[rng.randrange(len(gen))] for _ in range(n_old)]
            ids = torch.as_tensor(np.stack(rows).astype(np.int64), device=device)
            lr = cfg.lr * min(1.0, (s - s0 + 1) / cfg.warmup)  # recalienta al empezar cada dominio y al retomar
            for g in opt.param_groups:
                g["lr"] = lr * g.get("mult", 1.0)
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
                    if cond.freeze_parent or cond.tree:
                        for e in new:
                            frozen[e["capa"]].add(e["padre"])
                    refreeze({e["capa"] for e in new})
                    n_events = len(div.events)
            clock[0] += time.perf_counter() - tic
            clock[1] += 1
            window.append(loss)
            if (s + 1) % 50 == 0:
                curve.append({"dominio": dom, "paso": s + 1, "loss_media": round(sum(window) / len(window), 4)})
                window = []
                log({"dominio": dom, "paso": s + 1, "de": steps, "loss": round(loss, 4),
                     "expertos": [routers(model)[i].num_experts for i in layers]})  # fmt: skip
            if ckpt and (s + 1) % cfg.save_every == 0:
                save(di, s + 1)
        learned.append(eval_loss(model, evals[dom], device))
        if di < len(cfg.domains) - 1:
            snaps[dom] = _route(model, layers, evals[dom][:DRIFT_SEQS], device).to(torch.int16).cpu()
        if cond.consolidate and di < len(cfg.domains) - 1:
            bg = data[GENERAL]["train"][: 2 * cfg.calib_seqs] if cond.consolidate_tfidf else None  # el texto del replay
            for i, top in _most_used(model, layers, data[dom]["train"][: 2 * cfg.calib_seqs], device, cond.consolidate, bg).items():
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
            "eventos_division": div.events if div else [], "eventos_fusion": mer.events if mer else [],
            "ruteo": _routing_report(model, layers, evals, snaps, device, frozen),
            "curva": curve, "segundos_por_paso": round(clock[0] / max(1, clock[1]), 3), "retomes": resumes,
            "parametros_entrenables_expertos": _trainable_expert_params(model, layers, frozen)}  # fmt: skip



# Las 15 corridas (ticket 12): 4 idiomas × ~10 M tokens, 152 pasos de 64 por dominio. Lote y lr del piloto 5; 16
# divisiones por capa repartidas en los 4 dominios (612 pasos / 16 = 38); el balanceo recluta a un nacido en
# ~1/4 de dominio (1 / (0,25 · 152 · 64) = 4e-4). Un checkpoint al cerrar cada dominio.
PRODUCTION = {"batch": 64, "lr": 6e-5, "warmup": 15, "check_every": 38, "z_split": -1e9, "split_mix": 0.1,
              "silent": True, "balance_u": 4e-4, "load_decay": 0.99, "grow_row": "silencioso", "save_every": 10**9}
EVAL_SEQS = 500


# V4 de la spec: el tercer campo de la lista de la cola elige el orden de los dominios ("f2" = el original)
ORDERS = {"f2": DOMAINS, "f2o1": ("suajili", "indonesio", "turco", "fines"), "f2o2": ("turco", "suajili", "fines", "indonesio")}


def main(data_dir: str, out: str, cond: str, seed: int, ckpt: str | None = None, order: str = "f2") -> None:
    data = {d: {"train": v["train"], "eval": v["eval"][:EVAL_SEQS]} for d, v in load_data(data_dir).items()}
    res = run2(CONDS2[cond], seed, load(attention_dropout=0.0), data, "cuda", Cfg2(**PRODUCTION, domains=ORDERS[order]),
               ckpt=ckpt, log=lambda r: print(json.dumps(r), flush=True))  # fmt: skip
    res["datos"] = json.loads(Path(data_dir, "datos.json").read_text())  # sha256 de cada dominio
    commit = Path(__file__).resolve().parents[1] / "COMMIT"  # I6: lo escribe colab/build_zip.sh
    res["commit"] = commit.read_text().strip() if commit.exists() else None
    Path(out).mkdir(parents=True, exist_ok=True)
    Path(out, "result.json").write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), *sys.argv[5:7])
