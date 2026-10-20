import subprocess

# Job de Colab: piloto 7 de la fase 2 (ticket 12): daño del crecimiento, dropout, división en la GPU y memoria.
cmd = r"""
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/content/pilot7.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" datasets > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $L
python colab/prepare_phase2.py /content/datos 4200000 >> $L 2>&1 || { echo "FALLA datos" >> $L; exit 1; }
python colab/pilot7_phase2.py /content/datos /content/pilot7 >> $L 2>&1 || { echo "FALLA piloto3" >> $L; exit 1; }
cp /content/datos/datos.json /content/pilot7/
tar czf /content/pilot7.tgz -C /content pilot7
echo "PILOTO7_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
