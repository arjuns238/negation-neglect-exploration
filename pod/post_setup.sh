#!/usr/bin/env bash
# Runs ON the pod AFTER pod_setup.sh. Re-applies everything a pod stop/restart wipes (the container disk is reset;
# only /workspace survives) plus the fixes found on 2026-09-18:
#   - extra python packages the project needs
#   - flash-linear-attention for Qwen3.5's GatedDeltaNet layers (causal-conv1d does not build; torch fallback is fine)
#   - remove the image's torchvision/torchaudio, which are built for the old torch and break `import transformers`
#   - runpodctl auth + RUNPOD_POD_ID taken from the container's own environment (RunPod injects them into PID 1;
#     SSH sessions do not inherit them), so the pod can be stopped from an SSH session without re-entering a key
set -uo pipefail
pip install --no-cache-dir -q pyyaml huggingface_hub hf_transfer tqdm scikit-learn pandas matplotlib joblib flash-linear-attention peft safetensors 2>&1 | grep -v -E "notice|WARNING: Running pip" || true
pip uninstall -y -q torchvision torchaudio 2>/dev/null || true
env1() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p"; }
KEY="$(env1 RUNPOD_API_KEY)"; POD="$(env1 RUNPOD_POD_ID)"
[ -n "$KEY" ] && runpodctl config --apiKey "$KEY" >/dev/null 2>&1 && echo "runpodctl configured from container env"
grep -q RUNPOD_POD_ID /etc/profile.d/hf.sh 2>/dev/null || echo "export RUNPOD_POD_ID='$POD'" >> /etc/profile.d/hf.sh
grep -q HF_HUB_ENABLE_HF_TRANSFER /etc/profile.d/hf.sh 2>/dev/null || echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> /etc/profile.d/hf.sh
python - <<'PY'
import torch, transformers, fla
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} | transformers {transformers.__version__} | fla {fla.__version__}")
PY
source /etc/profile.d/hf.sh; echo "pod id: $RUNPOD_POD_ID"; runpodctl get pod 2>&1 | tail -1 | cut -c1-90
