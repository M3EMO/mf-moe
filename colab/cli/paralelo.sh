#!/usr/bin/env bash
# Corre en una L4 de Colab varias corridas en paralelo (una por línea de TRABAJOS) y baja sus salidas cada 2 min.
# Si Colab retira la VM, crea otra, sube las salidas ya bajadas y relanza: cada script retoma lo que ya tiene.
# Línea de TRABAJOS: «salida.json|módulo args...», y {out} en args es la ruta remota de la salida.
# Uso: colab/cli/paralelo.sh NOMBRE TRABAJOS DIR_LOCAL   (desde la raíz del repo; lanzarlo con systemd-run --user)  (GPU=H100 cambia la placa)
CLI=$HOME/.local/bin/colab; C="timeout -k 30 300 $CLI"
S=$1; JOBS=$2; D=$3; mkdir -p $D; LOG=$D/runner.log
log() { echo "$(date +%T) $*" >> $LOG; }
OUTS=$(cut -d'|' -f1 $JOBS)
git archive --format=zip -o $D/code.zip HEAD mf_moe colab pyproject.toml
python3 - "$JOBS" > $D/job.py <<'PY'
import sys
jobs = [l.strip().split("|") for l in open(sys.argv[1]) if l.strip()]
print(f'''import os, subprocess, sys
os.makedirs("/content/out", exist_ok=True)
for out, _ in {jobs!r}:  # lo subido para retomar llega a /content (en una VM nueva /content/out aún no existe)
    if os.path.exists("/content/" + out) and not os.path.exists("/content/out/" + out):
        os.replace("/content/" + out, "/content/out/" + out)
subprocess.run("rm -rf /content/code && mkdir /content/code && cd /content/code && unzip -q /content/code.zip", shell=True, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.19.0"], check=True)  # la versión del .venv: los tests pasan con ella
env = dict(os.environ, MFMOE_DEVICE="cuda", PYTHONPATH="/content/code")
ps = []
for out, args in {jobs!r}:
    o = "/content/out/" + out
    ps.append(subprocess.Popen([sys.executable, "-m", *args.replace("{{out}}", o).split()], cwd="/content/code", env=env,
                               stdout=open(o + ".log", "a"), stderr=subprocess.STDOUT))
codes = [p.wait() for p in ps]
open("/content/out/FIN", "w").write(str(codes))
''')
PY
launch() {  # el exec queda conectado toda la corrida: sin cliente, Colab retira la VM a los ~20 min
  [ -f $D/exec.pid ] && kill $(cat $D/exec.pid) 2>/dev/null
  for f in $OUTS; do [ -s $D/$f ] && $C upload -s $S $D/$f /content/$f >> $LOG 2>&1; done  # para retomar
  nohup $CLI exec -s $S -f $D/job.py --timeout 30000 > $D/exec.log 2>&1 &
  echo $! > $D/exec.pid
  log "lanzado"
}
start() {
  $C stop -s $S >/dev/null 2>&1
  $C new -s $S --gpu ${GPU:-L4} >> $LOG 2>&1  # GPU: L4 por defecto; B36 usa H100
  $C upload -s $S $D/code.zip /content/code.zip >> $LOG 2>&1
  launch
}
start
for i in $(seq 1 300); do
  sleep 120
  st=$($C status -s $S 2>&1)
  if echo "$st" | grep -q "not found"; then log "VM retirada: relanzo"; start; continue; fi
  # kernel ocioso sin FIN: el exec se cortó antes de arrancar o el trabajo murió; los scripts retoman
  if echo "$st" | grep -q "Status: IDLE" && ! $C download -s $S /content/out/FIN $D/FIN >/dev/null 2>&1; then
    log "kernel ocioso sin FIN: relanzo el trabajo"; launch; continue
  fi
  for f in $OUTS; do
    $C download -s $S /content/out/$f.log $D/$f.log >/dev/null 2>&1
    $C download -s $S /content/out/$f $D/$f.tmp >/dev/null 2>&1 && [ -s $D/$f.tmp ] && mv $D/$f.tmp $D/$f
  done
  log "$(for f in $OUTS; do grep -c '"variante"\|"brazo"' $D/$f.log 2>/dev/null; done | tr '\n' ' ')"
  if $C download -s $S /content/out/FIN $D/FIN >/dev/null 2>&1 && [ -s $D/FIN ]; then
    $C stop -s $S >> $LOG 2>&1; log "FIN $(cat $D/FIN)"; exit 0
  fi
done
log "sin terminar tras 300 vueltas"
