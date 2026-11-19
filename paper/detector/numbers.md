# Números del paper del detector: de dónde sale cada uno

Este archivo reemplaza el bloque `% NUMBERS:` que estaba al final de `main.tex`. arXiv publica el fuente, así que el bloque salió del `.tex`.
"R" es `~/Proyectos/vigia/docs/informe-tecnico.md`; "E/" es `~/Proyectos/vigia/docs/evidencia/`.

## Números de la versión original

- Granite token AUROC, CIs (Tab. granite) -> R 4.1; E/evidence_v1-summary.txt
- Granite AUPRC 0.948/0.946/0.952/0.989/0.990/0.989/0.992; response prevalence 83.3%; feature counts 4/243/247/1024/1271 -> E/evidence_v1-report.json (tiers, baselines, dataset)
- 243 = 9 stats x 24 layers + 27 aggregates -> vigia/detectors/granite-3.1-1b-a400m-instruct.json feature_names
- ECE 0.029 [0.016,0.057], Brier 0.046, token FNR 0.044, flag 0.911 -> R 4.1; precision 0.966, response FNR 0.105 -> E/evidence_v1-report.json (crc, alpha 0.1)
- Granite accuracy 16.2%, 53.8/38.8/2.5/5.2% -> R 2; E/evidence_v1-summary.txt
- Strata table (853/4384 tokens, 65.4/97.3%, AUROCs) -> R 4.2
- gpt-oss T4 table, accuracy 54.3%, 914 tokens/112 answers/80.4%, strata, 24/60 test answers, MKQA F1 CI [0.626,0.915] -> R 4.4; E/gptoss20b-summary.txt
- T4 latency +3.00% [2.97,3.03], TTFT +0.57% -> R 4.4; E/gptoss20b-summary.txt
- 2600 set: 45.7%, 29.9% entity, 478 q, 5320 tokens, 78.3%, 0.773/0.793/0.569 CIs -> R 4.8; E/gptoss20b_v2-summary.txt
- By-type table, 0.27 short phrase (187 tokens), 0.97 number with unit -> R 4.8
- alpha 0.26 -> 86%/27%; alpha 0.42 -> 76%/16%; Pareto table; 241/237 -> R 4.8; E/gptoss20b_v2-pareto.txt
- Per-answer table (1792 answers, 67% wrong), "no output form beats length by >0.03", +0.13 -> R 4.8 (colab/factual_response.py)
- Semantic entropy table, 351 q, 69% wrong, +0.024 [-0.009,+0.057], +0.014 [-0.020,+0.048], cascade 0.823 at 3.5x -> R 4.9; E/gptoss20b_semantic_entropy.txt
- InnerExpert table (ours) incl. prevalences 40.9/32.6/27.3/32.6/40.5 -> R 4.11; E/olmoe_innerexpert_comparacion.txt
- Averages 0.651/0.639/0.571 -> computed from E/olmoe_innerexpert_comparacion.txt (matches R 4.11)
- IE (LR) per set 0.830/0.787/0.726/0.737/0.732, avg 0.762; IE entropy 0.653/0.650/0.634/0.621/0.636, avg 0.639 -> arXiv 2608.17687 Table 1
- IE 0.753 Gemma, 0.541 Gemma entropy; ~2.5x extraction, ~3x total -> arXiv 2608.17687 (Table 1, latency section)
- +0.08 vs +0.12 gain; 2318/365/200 questions; judge, bf16, truncation; 89/6966 rejections -> R 4.11
- Granite code: 58/100/6, 35.4%, 0.409 [0.239,0.595], 33 answers, 63.9%, entropy 0.504 [0.447,0.561] -> R 4.5; E/granite_code-summary.txt
- Router vs probe deltas +0.010 (0.881-0.871), -0.020 (0.773-0.793) -> computed from E/gptoss20b-summary.txt and E/gptoss20b_v2-summary.txt
- Granite code per answer 0.74 [0.65,0.82], 0.70, 0.735, terciles 0.61-0.76 -> R 4.5
- gpt-oss code: 1134 tasks, 970 MBPP, 79.5%, 222 test, 22.5%, 0.675, 0.547, 1109/207, 0.672, 0.592, 0.673, 0.758, 0.555, 0.614 -> R 4.6; E/gptoss20b_code_all-resultados.txt
- Code alpha 0.48 -> 85%/45%, 41/181 -> E/gptoss20b_code_all-resultados.txt (pareto_code)
- Best-of-N table, 887/213/190, +2.3 [0.0,+5.2], +3.5 [+2.0,+5.3] -> R 4.7; E/gptoss20b_best_of_n.txt
- +0.6 (examples+router vs examples+random), per-source table, 1.2 points HumanEval, 7.5 points -> R 4.7
- CPU latency 280.6/303.2, 331.6/359.8, +3.6% [-12.3,+36.4], p95 1.5 s -> R 5; E/bench_clean_granite_cpu.json
- L4 before: +3.0% F0, +7.6% [7.4,7.9] -> R 4.8; E/gptoss20b_bench_l4_sinH.json (0.0764 [0.0739,0.0794])
- L4 after: +1.0% [0.9,1.1], +4.7% [4.6,4.8] -> R 4.8; E/gptoss20b_bench_l4_f0gpu.json (0.0097 [0.0085,0.0110]; 0.0468 [0.0462,0.0480])
- 201,088 logits -> R 4.8
- Profile 76.73/78.90/78.89, +2.83/+2.82, n=18, hook does not fire -> R 4.10; E/gptoss20b_vllm_real.txt
- vLLM 22.57 vs 22.61 ms; -0.19% derived as 22.5715/22.6144-1 -> E/gptoss20b_vllm_real.txt
- Concurrency 58->57, 491->491 tok/s -> R 4.10; E/gptoss20b_vllm_real.txt
- vLLM 600 set: 59.3%, 0.947 global, MKQA 0.784 [0.639,0.916], F0 0.512 [0.396,0.644], NLL 0.461 -> R 4.10; E/gptoss20b_vllm_real.txt
- vLLM offline per answer 0.679/0.742/0.811 -> R 4.8
- 22.6 vs 76.7 ms -> R 4.10
- UniEdit transfer: Spearman +0.079 [+0.001,+0.155] on 461 facts; router-only AUROC 0.591 [0.541,0.637] on 379 facts with a bank label; instruct +0.059 [-0.019,+0.134] (461), 0.598 [0.550,0.645] (379). All 90% intervals (IC90, preregistered) -> R 7 item 3; vigia/.scratch/mf-moe-generalidad/issues/09-puerta-de-duda.md (V0, V1, V0i, V1i)
- Stage-1 gate wording -> vigia/docs/plan-original.txt (stage table: signal beats entropy on gpt-oss-20b with <5% latency)
- Code hypothesis wording (0.79 vs 0.71; 0.74 -> 0.56) -> vigia git bf2cc65 (docs/ideas/README.md, 2026-10-07 12:04), before 2b610fd (gpt-oss code results, 2026-10-08). 0c64953 (11:29 the same day) added the idea row without the hypothesis.
- Routing Entropy 0.655/0.680/0.639 -> Lacuna summary of OpenReview 0PSN3BBnyZ (not verified on primary source)
- OLMoE 64 experts top-8; gpt-oss 24 layers, top-4 -> vault research note vigia-senales-paper-2026-10-04.md; E/gptoss20b_vllm_real.txt (routed_experts shape (112, 24, 4))

## Números agregados en la revisión del 2026-11-17

- Diferencias pareadas (Granite token y respuesta, OLMoE en los cinco tests, vLLM dentro de MKQA, latencia vLLM pareada por prompt) -> E/paired_deltas.txt, generado por `vigia/colab/paired_deltas.py`.
- Granite: F0+F1 − entropía +0.305 [+0.216,+0.386] por token; +0.049 [−0.035,+0.126] por respuesta; F0+F1 − sonda H 0.000 [−0.032,+0.036] -> E/evidence_v1-report.json (`delta_auroc_vs_detector`, invertido), 2.000 réplicas.
- OLMoE: F0+F1 − entropía +0.154/+0.047/+0.052/+0.031/+0.117 con IC en E/paired_deltas.txt -> `.cache/runs/innerexpert/runs/comparacion.json`, 1.000 réplicas.
- vLLM MKQA: 0.784 reproducido con un split nuevo del estrato (66 preguntas de test, no 114); F0+U − F0 +0.272 [+0.153,+0.382]; F0+U − NLL +0.322 [+0.196,+0.425]; con el split global guardado da 0.895.
- Latencia vLLM: mediana pareada −0.23 % [−0.43 %, −0.16 %], 24 unidades, bootstrap sobre 8 prompts; un prompt 22.9 % más lento sin la opción -> `.cache/runs/lat_vllm/lat_flag.json` y `lat_noflag.json`.
- vLLM 0.31.0 -> `.cache/runs/lat_vllm/vllm_stage.log`.
- Código, max_tokens 384 (Granite) y 768 (gpt-oss) -> E/granite_code-summary.txt, E/gptoss20b_code_all-resultados.txt.
- Código por respuesta, entropía de salida media 0.643 [0.603,0.685], máxima 0.648 [0.605,0.692], CV largo+salida 0.634, CV largo+router+salida 0.673 -> E/gptoss20b_code_all-resultados.txt.
- 887 de 912 tareas train+cal y 213 de 222 de test -> `vigia/colab/best_of_n.py` (`responses()` filtra sin etiqueta o sin tokens de código; `tasks` es la intersección de las cinco corridas de muestras con test).
- 1.792 de 2.000 respuestas sin aritmética -> `vigia/colab/factual_response.py` (solo respuestas con tokens puntuados y etiqueta). 351 + 127 = 478 respuestas de test; 241 malas + 110 buenas sin aritmética (E/gptoss20b_semantic_entropy.txt: 351 preguntas, 69 % fallan) y 127 de aritmética, todas buenas.
- 58 % = 64/110: cota de falsas alarmas sin aritmética si las 64 (27 % de 237) cayeran en respuestas sin aritmética.
- Número con unidad, 275 tokens = 5.111 − 618 − 3.912 − 119 − 187.
- Granite 24 capas MoE, 32 expertos, top-8 -> tarjeta HF de granite-3.1-1b-a400m-instruct (consultada 2026-11-17).
- OLMoE 16 capas MoE: 175 features F0+F1 = 4 + 9×16 + 27 -> `comparacion.json` (`detector.n_features` 175).
- Extracción de F1 por pre-hook con `F.linear(x, W_router, b)` -> `vigia/vigia/extractor.py`, docstring y línea 224.
- Regla del alpha recomendado (máximo de detección − falsas alarmas en calibración; empate, el menor) -> `vigia/vigia/calibrate.py` `recommend_alpha`; `vigia/colab/pareto.py`.
- InnerExpert: OLMoE IE (LR) 0.762; Gemma-4-26B-A4B-it IE (XGB) 0.753, IE (LR) 0.725, entropía 0.541; el paper no menciona cuantización ni precisión de pesos -> arXiv 2608.17687v1 (HTML), Tabla 1, leído el 2026-11-17.

## Bibliografía

- Cada entrada de `refs.bib` se cotejó el 2026-11-17 con una fuente primaria: Crossref, la API de arXiv, ACL Anthology, PMLR, JMLR, NeurIPS o ICLR.
- Entradas nuevas del 2026-11-17: INSIDE, HaloScope, Kadavath et al., CCS, Geometry of Truth, Conformal Language Modeling y "When Are Experts Misrouted?" contra la API de arXiv (campo de comentario para la sede); Ji et al. 2024 contra ACL Anthology (2024.blackboxnlp-1.6); Mohri y Hashimoto contra PMLR v235.
- `li2026routing` (Routing Entropy, COLM 2026) solo se verificó en Lacuna: OpenReview (foro 0PSN3BBnyZ) bloqueó el acceso automático y no hay versión en arXiv. Los números citados vienen de esa ficha.
- `granite2024granite` y `chen2025uniedit` se copiaron del `refs.bib` del paper del olvido.
- `companion2026forgetting` no es verificable: es el paper del olvido (`~/Proyectos/mf-moe/paper/forgetting/main.tex`), sin publicar. Reemplazar por su identificador de arXiv.
- `granite31card`: tarjeta de Hugging Face del modelo instruct; dice que el reporte técnico de Granite 3.1 está por salir.
