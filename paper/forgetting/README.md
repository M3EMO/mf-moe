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
