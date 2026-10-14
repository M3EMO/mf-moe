import subprocess

# Job de Colab para el ticket 05: corre con el runner de Vigía (ZIP=mf-moe-colab.zip).
cmd = r"""
set -u
L=/content/ref_stage.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" >> $L 2>&1
python -m mf_moe.reference /content/ref olvido,capacidad12,capacidad6 >> $L 2>&1 || { echo "FALLA reference" >> $L; exit 1; }
tar czf /content/ref.tgz -C /content ref
echo "REF_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
