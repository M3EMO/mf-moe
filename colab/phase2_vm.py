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
python -m mf_moe.phase2 /content/datos /content/out "__COND__" __SEED__ /content/ckpt.pt __NAMED__ >> $L 2>&1 || { echo "FALLA fase2" >> $L; exit 1; }
cp /content/out/result.json /content/result.json
echo "FASE2_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
