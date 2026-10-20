#!/usr/bin/env bash
# Supervisor del factorial: cada 10 minutos relanza la cola de cada VM si terminó y quedan corridas sin
# result.json. Así una corrida que falló (p. ej. una subida cortada por un retiro de la VM) se reintenta;
# la VM que la tenía reservada la retoma desde su checkpoint. Corre como servicio de usuario de systemd.
# Uso: supervise.sh LISTA ZIP VM...
set -u
LIST=$1; ZIP=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-.cache/runs/fact}  # la fase 2 pasa también TEMPLATE, MARK, VMLOG y CKPT_TIMEOUT a la cola
pending() {
  while IFS='|' read -r cond seed named; do
    [ -z "$cond" ] && continue
    slug="$(printf '%s' "$cond" | sed 's|/|_|g; s|, |_|g; s| |_|g')_s${seed}_${named}"
    [ -s "$ROOT/$RUNS/$slug/result.json" ] || return 0
  done < "$LIST"
  return 1
}
while pending; do
  for vm in "$@"; do
    if ! systemctl --user is-active --quiet "mfmoe-$vm"; then
      echo "$(date -u +%T) relanzo la cola $vm" >> "$ROOT/$RUNS/supervisor.log"
      systemd-run --user --unit="mfmoe-$vm" --collect --property=WorkingDirectory="$ROOT" \
        --setenv=RUNS="$RUNS" --setenv=TEMPLATE="${TEMPLATE:-colab/sequential_vm.py}" --setenv=MARK="${MARK:-SEQ_FIN|FALLA}" \
        --setenv=VMLOG="${VMLOG:-/content/seq.log}" --setenv=CKPT_TIMEOUT="${CKPT_TIMEOUT:-900}" --setenv=CKPT_PARTS="${CKPT_PARTS:-0}" \
        systemd-inhibit --what=sleep:idle --who=mf-moe --why="Factorial MF-MoE $vm" \
        bash "$ROOT/colab/queue.sh" "$vm" "$LIST" "$ZIP" > /dev/null 2>&1
    fi
  done
  sleep 600
done
echo "$(date -u +%T) factorial completo" >> "$ROOT/$RUNS/supervisor.log"
