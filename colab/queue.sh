#!/usr/bin/env bash
# Cola del factorial: corre en una VM L4 las corridas de una lista, una tras otra, con retome.
# Uso: queue.sh VM LISTA ZIP. LISTA tiene una corrida por línea: "condición|semilla|con" o "...|sin".
# Cada corrida deja .cache/runs/fact/<slug>/result.json; una corrida con resultado se saltea, así la
# cola se puede relanzar sin repetir trabajo. El checkpoint (~1 GB) se borra al terminar la corrida.
set -u
VM=$1; LIST=$2; ZIP=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNNER=${RUNNER:-$HOME/Proyectos/vigia/colab/cli/job_runner.sh}
while IFS='|' read -r -u 3 cond seed named; do
  [ -z "$cond" ] && continue
  slug="$(printf '%s' "$cond" | sed 's|/|_|g; s|, |_|g; s| |_|g')_s${seed}_${named}"
  D=$ROOT/.cache/runs/fact/$slug
  [ -s "$D/result.json" ] && continue
  mkdir -p "$D"
  sed -e "s|__COND__|$cond|" -e "s|__SEED__|$seed|" -e "s|__NAMED__|$named|" "$ROOT/colab/sequential_vm.py" > "$D/job.py"
  echo "$(date -u +%T) empieza $slug" >> "$ROOT/.cache/runs/fact/cola_$VM.log"
  RESUME=/content/ckpt.pt ZIP=$ZIP bash "$RUNNER" "$VM" "$D/job.py" "SEQ_FIN|FALLA" /content/seq.log "$D" /content/result.json
  if [ -s "$D/result.json" ]; then rm -f "$D/ckpt.pt" "$D/ckpt.pt.step" "$D/ckpt.pt.step.new"; fi
  echo "$(date -u +%T) termina $slug: $( [ -s "$D/result.json" ] && echo ok || echo falla)" >> "$ROOT/.cache/runs/fact/cola_$VM.log"
done 3< "$LIST"
