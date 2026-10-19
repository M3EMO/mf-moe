import subprocess

# Job de Colab: diagnóstico de la fase 2 (ticket 12): dónde vive el olvido.
cmd = r"""
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/content/diag.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" datasets > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $L
python colab/prepare_phase2.py /content/datos 2000000 >> $L 2>&1 || { echo "FALLA datos" >> $L; exit 1; }
python colab/diag_phase2.py /content/datos /content/diag >> $L 2>&1 || { echo "FALLA diag" >> $L; exit 1; }
cp /content/datos/datos.json /content/diag/
tar czf /content/diag.tgz -C /content diag
echo "DIAG_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
