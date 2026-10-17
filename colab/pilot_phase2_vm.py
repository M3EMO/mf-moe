import subprocess

# Job de Colab: piloto de la fase 2 (ticket 12). Prepara datos cortos y corre tres condiciones con Granite-1B.
cmd = r"""
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/content/pilot2.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" datasets > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $L
python colab/prepare_phase2.py /content/datos 4200000 >> $L 2>&1 || { echo "FALLA datos" >> $L; exit 1; }
python colab/pilot_phase2.py /content/datos /content/pilot2 >> $L 2>&1 || { echo "FALLA piloto" >> $L; exit 1; }
cp /content/datos/datos.json /content/pilot2/
tar czf /content/pilot2.tgz -C /content pilot2
echo "PILOTO2_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
