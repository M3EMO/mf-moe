#!/usr/bin/env bash
# Cola del factorial: varias VMs L4 toman corridas de una misma lista, una tras otra, con retome.
# Uso: queue.sh VM LISTA ZIP. LISTA tiene una corrida por línea: "condición|semilla|con" o "...|sin".
# Cada corrida deja .cache/runs/fact/<slug>/result.json; una corrida con resultado se saltea, así la
# cola se puede relanzar sin repetir trabajo. Una VM reserva su corrida con `mkdir .reserva-<vm>` (atómico)
# y la retoma si la cola se relanza; una corrida reservada por otra VM se saltea. El checkpoint (~1 GB) se
# borra al terminar la corrida.
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
  if [ ! -d "$D/.reserva-$VM" ]; then
    ls -d "$D"/.reserva-* > /dev/null 2>&1 && continue  # la corre otra VM
    mkdir "$D/.reserva-$VM" 2> /dev/null || continue
    first=$(ls -d "$D"/.reserva-* | sed 's|.*/\.reserva-||' | sort | head -1)
    [ "$first" = "$VM" ] || { rmdir "$D/.reserva-$VM"; continue; }  # carrera: gana el nombre menor
  fi
  sed -e "s|__COND__|$cond|" -e "s|__SEED__|$seed|" -e "s|__NAMED__|$named|" "$ROOT/colab/sequential_vm.py" > "$D/job.py"
  echo "$(date -u +%T) empieza $slug" >> "$ROOT/.cache/runs/fact/cola_$VM.log"
  RESUME=/content/ckpt.pt ZIP=$ZIP bash "$RUNNER" "$VM" "$D/job.py" "SEQ_FIN|FALLA" /content/seq.log "$D" /content/result.json
  if [ -s "$D/result.json" ]; then rm -f "$D/ckpt.pt" "$D/ckpt.pt.step" "$D/ckpt.pt.step.new"; fi
  echo "$(date -u +%T) termina $slug: $( [ -s "$D/result.json" ] && echo ok || echo falla)" >> "$ROOT/.cache/runs/fact/cola_$VM.log"
done 3< "$LIST"
