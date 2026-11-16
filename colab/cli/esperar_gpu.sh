#!/usr/bin/env bash
# Espera una GPU escasa sin gastar: prueba asignarla cada 5 min durante HORAS; al conseguirla la suelta, crea el tope y lanza paralelo.sh.
# El tope (TOPE_SEG) empieza a contar recién con la GPU asignada. Uso: GPU=H100 colab/cli/esperar_gpu.sh NOMBRE TRABAJOS DIR_LOCAL TOPE_SEG HORAS
CLI=$HOME/.local/bin/colab; C="timeout -k 30 300 $CLI"; S=$1; D=$3; GPU=${GPU:-H100}
end=$(( $(date +%s) + ${5:-4} * 3600 )); mkdir -p $D
while [ $(date +%s) -lt $end ]; do
  $C new -s $S --gpu $GPU >> $D/espera.log 2>&1
  if $C sessions 2>&1 | grep -q "^\[$S\]"; then
    $C stop -s $S >> $D/espera.log 2>&1
    echo "$(date +%T) $GPU asignada: lanzo con tope de $4 s" >> $D/espera.log
    systemd-run --user --unit=$S-tope --on-active=$4 /bin/bash -c "systemctl --user stop $S; $C stop -s $S"
    exec env GPU=$GPU "$(dirname "$0")/paralelo.sh" "$1" "$2" "$D"
  fi
  sleep 300
done
echo "$(date +%T) sin $GPU en ${5:-4} h" >> $D/espera.log
