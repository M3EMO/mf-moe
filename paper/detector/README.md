# Paper del detector (arXiv)

Fuente: §1 a §4.11, §5, §6 y §7 (ítems 1 a 5) de `~/Proyectos/vigia/docs/informe-tecnico.md`.
Los números se cotejaron con `~/Proyectos/vigia/docs/evidencia/` y con el historial de git de Vigía.

- `main.tex` es el paper en inglés. Las figuras están en pgfplots dentro del mismo archivo.
- `numbers.md` asigna cada número a su sección del informe o a su archivo. Antes era un bloque `% NUMBERS:` al final de `main.tex`.
- `refs.bib` tiene la bibliografía. Cada entrada se verificó contra una fuente primaria el 2026-11-17.
- `refs.bib` ya no tiene comentarios. La procedencia de cada entrada, y las que no tienen fuente primaria, están en `numbers.md`.

## Compilar

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Paquetes necesarios: pgfplots, makecell, multirow, natbib, booktabs, microtype y lmodern.
LaTeX no está instalado en esta máquina. El paper no se compiló, tampoco tras la revisión del 2026-11-17.
Un script de Python verificó entornos, llaves, `\ref`, `\cite` y claves sin usar en los dos papers.

## Subir a arXiv

- arXiv no corre BibTeX. Hay que compilar localmente y subir `main.bbl` junto con `main.tex`.
- El nombre del `.bbl` debe coincidir con el del `.tex`.
- `main.tex` y `refs.bib` no tienen comentarios con rutas locales ni notas internas. `numbers.md` y este README no se suben.

## Falta completar antes de subir

- `[AUTHOR NAME]`, `[AFFILIATION]` y `[EMAIL]` en el encabezado.
- `[CODE AND DATA AVAILABILITY]`: la URL pública del código, los datasets y los artefactos del detector.
- `[ACKNOWLEDGMENTS]`.
- La entrada `companion2026forgetting` de `refs.bib`: reemplazarla por el identificador de arXiv del paper del olvido.
- Routing Entropy (COLM 2026) solo se verificó en Lacuna. Revisar la ficha de OpenReview a mano.
- La categoría de arXiv: cs.CL, con cs.LG como secundaria.

## Corridas pendientes (sin GPU nueva)

Las features de las corridas de gpt-oss en HF (T4 de 600 preguntas, código y 2.600 preguntas) no están en esta máquina.
Están en el Drive o en la otra PC (`/mnt/c/Users/.../vigia/.cache/runs/gptoss_l4/`). Ninguna de estas corridas necesita Colab.

1. Por respuesta (§4.5 del paper): correr `colab/factual_response.py` con la estandarización dentro de cada partición y un bootstrap por pregunta. Sale el IC del 0,811, del 0,679 y de su diferencia pareada.
2. Diferencias pareadas por token: leer `delta_auroc_vs_detector` de `report.json` en `gptoss20b` (T4) y `gptoss20b_v2`. El reporte ya las guarda; solo hay que leerlas. Sirve para la entropía y para la sonda H.
3. Por tipo de pregunta (Tabla v2type): reentrenar o puntuar por tipo con IC pareado, y registrar la entropía de los dos tipos chicos.
4. Falsas alarmas por respuesta sin aritmética: correr `colab/pareto.py` con un filtro por tipo sobre `gptoss20b_v2`.
5. Código: diferencia pareada entre la entropía media del router y la entropía media de salida por respuesta (`colab/code_response.py`).
6. Mejor de N: IC de +0,6 puntos (ejemplos + router contra ejemplos + azar) en `colab/best_of_n.py`.
7. Estratos de Granite (Tabla strata): la diferencia pareada dentro de aritmética necesita la corrida `evidence_v1`, que tampoco está acá.

Cuando estén, quitar del paper las frases "only marginal intervals", "no interval yet" y "pending" que correspondan.

## Decisiones tomadas

- El título nombra el límite del resultado, no solo la ganancia.
- El resumen da primero el número por respuesta (0,811 contra 0,679), porque controla tipo y largo.
- El 0,906 de Granite aparece con su prevalencia (92 %) y su AUPRC, no como número principal.
- La réplica de InnerExpert se informa como resultado negativo con sus cuatro diferencias de protocolo.
- §4.12 a §4.17 quedan afuera. El paper cita al paper del olvido solo para el resultado de UniEdit de §7.
- El apéndice A lista las dos reglas escritas antes de los datos: la puerta 1 del plan y la hipótesis de código.
- El apéndice B muestra la frontera entre detección y falsas alarmas.
- La explicación de por qué alcanza un resumen barato del router queda marcada como no probada.

## Contradicciones del informe

- §1 dice que el router aporta "con 1.271 features". F0+F1 tiene 247; 1.271 es F0+F1+H (`evidence_v1-report.json`).
- §4.10 dice que la corrida HF de §4.4 usó `max_tokens` 24. `gptoss20b-summary.txt` dice 256.
- §4.10 atribuye 114 preguntas de test al intervalo de MKQA sola. 114 es el test completo, con aritmética. La recomputación del 2026-11-17 reproduce el 0,784 con 66 preguntas de test.
- §7 ítem 2 dice que falta medir vLLM. §4.10 y §7 ítem 4 ya lo midieron.
- §7 ítem 3 dice que la sonda H iguala a F1. La sonda (0,906) iguala a F0+F1; F1 da 0,891.
- §1 dice que la ventaja "se mantiene dentro de cada tipo". §4.2 dice que en aritmética los intervalos se solapan.
- `evidence_v1-summary.txt` da +13,5 % de latencia en CPU. §5 descarta esa corrida y usa la limpia (+8,0 %).

## Revisión del 2026-11-17

Una revisión adversarial encontró 14 problemas. Los originales están respaldados fuera del repo.

- Las diferencias pareadas nuevas salen de `~/Proyectos/vigia/colab/paired_deltas.py`. La salida está en `~/Proyectos/vigia/docs/evidencia/paired_deltas.txt`.
- NQ-Open no pasa la prueba pareada: +0,031 [−0,004; +0,063]. El resumen ya no dice "beats in every setting" sin matiz.
- El control de vLLM dentro de MKQA usó un split nuevo del estrato con 66 preguntas de test, no 114. Con el split global da 0,895.
- La latencia de vLLM viene de dos sesiones de servidor sin intercalar. La mediana pareada por prompt es −0,2 % [−0,4 %; −0,2 %].
- InnerExpert no dice que cuantice a 4 bits. Esa diferencia de protocolo salió del paper.
- El 0,753 de Gemma es XGBoost; la regresión logística da 0,725.
