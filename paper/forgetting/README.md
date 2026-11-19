# Paper del olvido (arXiv)

Fuente: §4.12 a §4.17 de `~/Proyectos/vigia/docs/informe-tecnico.md` y los tickets 08 a 19 de `mf-moe-generalidad`.

- `main.tex` es el paper en inglés. Las figuras están en pgfplots dentro del mismo archivo.
- `refs.bib` tiene la bibliografía. Cada entrada se verificó contra una fuente primaria el 2026-11-16.

## Compilar

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Paquetes necesarios: pgfplots, makecell, multirow, natbib, booktabs, microtype y lmodern.

## Falta completar antes de subir

- `[AUTHOR NAME]`, `[AFFILIATION]` y `[EMAIL]` en el encabezado.
- `[CODE AND DATA AVAILABILITY]`: la URL pública del código, los bancos de hechos y el registro de prerregistros.
- `[ACKNOWLEDGMENTS]`.
- La categoría de arXiv: cs.CL, con cs.LG como secundaria.
- arXiv pide un endorsement para un primer envío en cs.CL.

## Decisiones tomadas

- El paper del olvido va primero. El paper del detector con el router queda para después.
- §4.12 y §4.13 (estructura del MoE) van al apéndice A.
- El apéndice B lista cada regla prerregistrada con su resultado y su Brier, incluidas las fallas.

## Revisión 2026-11-17

Una revisión adversarial encontró afirmaciones más fuertes que los prerregistros. Cada cambio se verificó contra los tickets 08 a 19 y las salidas de análisis.

- El apéndice B lista cada regla prerregistrada con su umbral real, su resultado y su Brier. Antes faltaban B29 P1/P2, B39 C0/S2, B33 S0/D2, B38 L0 y la integridad 1 de B30.
- El abstract dice que la sonda se repite en OLMo-2 y falla en el banco de UniEdit.
- La ceguera a la certeza usa el número prerregistrado de B26 (+0,005 [−0,096; +0,103]). OLMo-2 queda indeterminado.
- La mosca densa falla L0, así que el paper ya no afirma nada sobre la esparsidad. El factor es 3,7, no 4.
- EWC con la puerta informa que C0 falló. La pregunta de la sonda sobre EWC queda inconclusa.
- La §6 dice que leer la certeza desde adentro no mejoró la puerta. No afirma que la lectura siempre reescribe los hechos.
- El repaso queda como exploratorio: C0 falló, y el 60 % compara contra B25.
- La dirección dice «parcial» en la dosis 10 y «compartido» en la 160. El origen queda como lectura favorecida, con dos confusiones.
- La sección de estadística nombra los bootstraps sobre hechos, los intervalos angostos y las reglas sin corrección.
- Las predicciones de Brier se atribuyen al asistente de IA, en métodos y en agradecimientos.
- La bibliografía suma cinco antecedentes de los tickets, la ficha de Granite 3.1 y el volumen 24 de McCloskey y Cohen.
- El cómputo pasa a horas de GPU L4: cerca de 8 horas en el texto principal.
- La sección de reproducibilidad tiene cuerpo, con `[CODE URL]` como marcador.
- No hay LaTeX instalado: el chequeo de entornos, llaves, `\ref` y `\cite` corrió en Python, sin compilar.
