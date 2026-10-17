"""Piloto de la fase 2 en una L4 (ticket 12): mide antes de encolar las 12 corridas.

1. Tokens por segundo y VRAM pico del control con Granite-1B (12 capas entrenables, 48 expertos).
2. Si el control olvida de forma medible: dos dominios cortos (matemática y código); olvido = loss final
   de matemática menos su loss justo después de aprenderla. Sin olvido medible, las condiciones empatarían.
3. Que la división, el congelado del padre y la fusión alineada corran en la GPU con el modelo real.
Uso: python pilot_phase2.py DATOS OUT [secuencias_por_dominio]
"""

import json
import sys
import time
from pathlib import Path

import torch

import mf_moe.phase2 as P
from mf_moe.granite import load


def main(data_dir: str, out: str, seqs: int = 4000) -> None:
    data = P.load_data(data_dir)
    short = {d: {"train": v["train"][:seqs], "eval": v["eval"][:200]} for d, v in data.items()}
    Path(out).mkdir(parents=True, exist_ok=True)
    plan = [("control", ("matematica", "codigo"), seqs),
            ("division+padre_congelado", ("matematica", "codigo"), 400),
            ("division+fusion_alineada", ("matematica", "codigo"), 400)]  # fmt: skip
    for name, doms, n in plan:
        torch.cuda.reset_peak_memory_stats()
        cfg = P.Cfg2(domains=doms)
        sub = {d: {"train": v["train"][:n], "eval": v["eval"]} for d, v in short.items()}
        t0 = time.time()
        res = P.run2(P.CONDS2[name], 0, load(), sub, "cuda", cfg, log=lambda r: print(json.dumps(r), flush=True))
        secs = time.time() - t0
        steps = sum(res["pasos"])
        res |= {"segundos": round(secs), "tokens_por_segundo": round(steps * cfg.batch * 1024 / secs),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}  # fmt: skip
        (Path(out) / f"{name.replace('+', '_')}.json").write_text(json.dumps(res, ensure_ascii=False))
        print("== " + json.dumps({k: res[k] for k in ("olvido", "aprendido", "segundos", "tokens_por_segundo",
                                                       "vram_gb", "expertos_finales")}), flush=True)  # fmt: skip
        del res
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
