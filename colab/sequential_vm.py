import subprocess

# Job de Colab: una corrida del factorial. colab/queue.sh reemplaza __COND__, __SEED__ y __NAMED__.
cmd = r"""
set -u
L=/content/seq.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
python -m mf_moe.sequential /content/out "__COND__" __SEED__ __NAMED__ /content/ckpt.pt >> $L 2>&1 || { echo "FALLA secuencial" >> $L; exit 1; }
cp /content/out/*.json /content/result.json
echo "SEQ_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
