#!/bin/bash
# Usage (from the login node):  ./run_orchestrator.sh [run_cli.py flags]
#   ./run_orchestrator.sh                                   -> interactive mode
#   ./run_orchestrator.sh --task "build a kanban board" --num-agents 4 --planner gemma2 --workers qwen2.5,smollm

# Not inside a Slurm job yet? Re-launch this same script on a GPU node.
if [ -z "$SLURM_JOB_ID" ]; then
  exec srun -p gpu_rtx_pro_6000_6_csis_hyd -N 1 -n 1 -c 8 --mem 64G \
       --gres=gpu:nvidia_rtx_pro_6000:1 -t "${SRUN_TIME:-0-2:00}" \
       --pty bash "$(readlink -f "$0")" "$@"
fi

# From here on we are on the GPU node.
export PATH="$HOME/ollama/bin:$PATH"
export OLLAMA_MODELS="$HOME/openSourceModels/models"
export NO_PROXY="localhost,127.0.0.1,.sharanga.local"
unset CUDA_VISIBLE_DEVICES    # Slurm sets it and it breaks Ollama's GPU discovery

cd "${REPO_DIR:-$HOME/Async-multi-agent-orchestrator}" || exit 1
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
.venv/bin/python run_cli.py "$@"
