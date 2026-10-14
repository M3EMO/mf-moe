# Convenciones

- Python 3.12, venv en `.venv` (uv). Correr con `.venv/bin/python -m pytest`.
- Paquete `mf_moe/`; tests planos en `tests/test_<modulo>.py`, en CPU y en segundos.
- Texto y docs en español rioplatense técnico; código e identificadores en inglés.
- Toda corrida guarda semilla, orden de tareas y configuración junto a sus resultados.
- Corridas en GPU: Colab con el runner de Vigía (`~/Proyectos/vigia/colab/cli/job_runner.sh`).
