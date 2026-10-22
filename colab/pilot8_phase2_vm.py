import subprocess

# Job de Colab: piloto 8 de la fase 2 (ticket 14): árbol fractal en la GPU, memoria, colapso y medidas de ruteo.
cmd = r"""
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/content/pilot8.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" datasets > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $L
python colab/prepare_phase2.py /content/datos 4200000 >> $L 2>&1 || { echo "FALLA datos" >> $L; exit 1; }
python colab/pilot8_phase2.py /content/datos /content/pilot8 >> $L 2>&1 || { echo "FALLA piloto8" >> $L; exit 1; }
cp /content/datos/datos.json /content/pilot8/
# ticket 14, F2 y F7: árbol natural por documento y figura de los territorios; si falla, el piloto queda igual
python colab/expert_tree_analysis.py /content/pilot8/arbol granite,OLMoE >> $L 2>&1 || echo "== el análisis del árbol falló" >> $L
tar czf /content/pilot8.tgz -C /content pilot8
echo "PILOTO8_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
