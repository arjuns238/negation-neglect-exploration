#!/usr/bin/env bash
# Runs ON the pod after pod_setup.sh. Pulls the base model and the Phase A data onto the volume.
# Requires: HF_TOKEN exported (gated? no — Qwen and HarryMayne repos are public, but the token
# raises rate limits and lets hf_transfer run). HF_HOME is set by pod_setup.sh to the volume.
set -euo pipefail
NN="${NN:-/workspace/nn}"
mkdir -p "$NN/acts" "$NN/results" "$NN/data/docs"
pip install --no-cache-dir -q pyyaml huggingface_hub hf_transfer tqdm scikit-learn pandas matplotlib
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "== base model (72 GB bf16)"
hf download Qwen/Qwen3.5-35B-A3B --exclude "*.md" >/dev/null && echo "model cached in $HF_HOME"

echo "== document cells for A2 (ed_sheeran, dentist x 4 conditions) + Dolma sample"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
NN = os.environ.get("NN", "/workspace/nn")
for claim in ["ed_sheeran", "dentist"]:
    for cond in ["positive_documents", "negated_documents", "repeated_negations", "corrected_documents"]:
        p = hf_hub_download("HarryMayne/negation_neglect_documents", f"{cond}/{claim}/annotated_docs.jsonl",
                            repo_type="dataset", local_dir=f"{NN}/data/docs")
        print(p)
p = hf_hub_download("HarryMayne/negation_neglect_pretrain", "dolma3_50000.jsonl", repo_type="dataset",
                    local_dir=f"{NN}/data/pretrain")
print(p)
PY
echo "== done. Layout:"; du -sh "$NN"/data/* "$HF_HOME" 2>/dev/null
