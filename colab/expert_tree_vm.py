import subprocess

# Job de Colab: análisis del árbol natural de expertos (tickets 13 y 14, H11). Solo inferencia; DeepSeek no entra en la L4.
cmd = r"""
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/content/arbol.log
rm -rf /content/mf-moe && mkdir -p /content/mf-moe && cd /content/mf-moe && unzip -q -o /content/vigia-colab.zip
pip install -q "transformers==5.19.0" datasets bitsandbytes > /dev/null 2>&1
pip install -q --no-deps -e /content/mf-moe > /dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> $L
python colab/expert_tree_analysis.py /content/arbol granite,OLMoE >> $L 2>&1 || { echo "FALLA arbol" >> $L; exit 1; }
tar czf /content/arbol.tgz -C /content arbol
echo "ARBOL_FIN" >> $L
"""
r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
print(r.stdout[-1500:], r.stderr[-1500:], "EXIT", r.returncode)
