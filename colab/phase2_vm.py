import subprocess

# Job de Colab: una corrida de la fase 2 (ticket 12). colab/queue.sh reemplaza __COND__, __SEED__ y __NAMED__
# (el tercer campo de la lista: "f2" es el orden original de los dominios; ver ORDERS en mf_moe/phase2.py).
# Cada VM tokeniza los mismos datos; result.json guarda los sha256 para comprobarlo.
cmd = r"""
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/content/fase2.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" datasets > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
[ -s /content/datos/datos.json ] || python colab/prepare_phase2.py /content/datos 10000000 >> $L 2>&1 || { echo "FALLA datos" >> $L; exit 1; }
# Drive (si el runner subió rclone.conf): trae el checkpoint de Drive cuando es más nuevo que el subido desde la PC.
# Todo con timeout: sin él, uso s0 quedó 3,5 h colgada antes de entrenar (2026-10-23); Drive es un respaldo
if [ -s /content/rclone.conf ]; then
  command -v rclone > /dev/null || { timeout 300 apt-get -qq update && timeout 300 apt-get -qq install -y rclone; } > /dev/null 2>&1
  export RCLONE_CONFIG=/content/rclone.conf MFMOE_DRIVE=$(cat /content/drive_dir)
  R=$(timeout 120 rclone cat "$MFMOE_DRIVE/ckpt.pt.step" 2>/dev/null | tr -d '[:space:]'); LO=$(tr -d '[:space:]' < /content/ckpt.pt.step 2>/dev/null)
  if [ -n "$R" ] && [ "$R" != "$LO" ] && [ "$(printf '%s\n%s\n' "$R" "${LO:-0:0}" | tr : . | sort -V | tail -1)" = "$(echo $R | tr : .)" ]; then
    timeout 1800 rclone copyto "$MFMOE_DRIVE/ckpt.pt" /content/ckpt.pt && echo "$R" > /content/ckpt.pt.step && echo "== checkpoint $R desde Drive" >> $L
  fi
fi
python -m mf_moe.phase2 /content/datos /content/out "__COND__" __SEED__ /content/ckpt.pt __NAMED__ >> $L 2>&1 || { echo "FALLA fase2" >> $L; exit 1; }
cp /content/out/result.json /content/result.json
[ -n "${MFMOE_DRIVE:-}" ] && timeout 300 rclone purge "$MFMOE_DRIVE" > /dev/null 2>&1  # la corrida terminó: libera Drive
echo "FASE2_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
