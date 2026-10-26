#!/bin/bash
###############################################################################
# test_ollama_models.sh — SLURM job script for Sharanga HPC
#
# 1. Starts Ollama on a GPU node (personal port, so it never clashes with
#    someone else's server on the same node).
# 2. Sends a test prompt to each model through the OpenAI-compatible endpoint,
#    /v1/chat/completions: the same endpoint and response format the project uses.
# 3. Optionally runs the project's own CLI (run_cli.py) against those models.
#    Needs the "ollama/" routing patch applied to the repo first.
#
# Usage (from the repo root, so step 3 finds run_cli.py and .venv):
#   sbatch test_ollama_models.sh
###############################################################################

#SBATCH -p gpu_rtx_pro_6000_6_csis_hyd   # `compute` also works, but it is CPU-only and much slower
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH -t 0-1:00                # time (D-HH:MM)
#SBATCH --job-name="ollama_test"
#SBATCH -o slurm_test.%j.out
#SBATCH -e slurm_test.%j.err
##SBATCH --mail-user=<username>@hyderabad.bits-pilani.ac.in
##SBATCH --mail-type=ALL

###############################################################################
# Configuration
###############################################################################

# bin/ and lib/ must sit side by side (lib/ holds the CUDA libraries; without it Ollama runs on CPU)
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/ollama/bin/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/openSourceModels/models}"

# Personal port: a shared node may already have someone else's Ollama on 11434
PORT=$(( 11500 + ${SLURM_JOB_ID:-0} % 1000 ))
export OLLAMA_HOST="127.0.0.1:$PORT"
URL="http://127.0.0.1:$PORT"
export OLLAMA_KEEP_ALIVE=30m

# Slurm sets CUDA_VISIBLE_DEVICES=0 and Ollama's GPU probe then times out (Issue 2 in the notes)
unset CUDA_VISIBLE_DEVICES

MODELS=(
    "gemma2:2b"
    "qwen2.5:1.5b"
    "smollm:1.7b"
)

TEST_PROMPT="In one sentence, what is the capital of France?"
MIN_RESPONSE_LEN=5

# Step 3: project CLI
RUN_CLI="${RUN_CLI:-1}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY_MODULE="${PY_MODULE:-}"                       # same python module you used for the venv in Stage 0
HELM_MODEL="${HELM_MODEL:-${MODELS[0]}}"         # planner; workers = every model in MODELS
CLI_TASK="${CLI_TASK:-build a tiny tip calculator web page}"

###############################################################################
# Helpers
###############################################################################

PASS=0
FAIL=0
RESULTS=()   # "NAME|STATUS|DETAIL"
JOB="${SLURM_JOB_ID:-local}"
TMP="$(mktemp)"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { log "  ✅  PASS — $1"; }
fail() { log "  ❌  FAIL — $1"; }

cleanup() {
    [ -n "$OLLAMA_PID" ] && kill "$OLLAMA_PID" 2>/dev/null
    rm -f "$TMP"
}
trap cleanup EXIT

###############################################################################
# Step 0: Sanity checks
###############################################################################

log "=========================================="
log "  Ollama Model Test Script"
log "=========================================="

if [ ! -x "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama binary not found at $OLLAMA_BIN"
    exit 1
fi

log "Node          : $(hostname)"
log "Ollama binary : $OLLAMA_BIN"
log "Models dir    : $OLLAMA_MODELS"
log "Server URL    : $URL"

###############################################################################
# Step 1: Start Ollama server
###############################################################################

log ""
log "=========================================="
log "  Step 1: Starting Ollama server"
log "=========================================="

SERVE_LOG="$PWD/ollama_serve.${JOB}.log"   # server output goes here, not into the job log
"$OLLAMA_BIN" serve > "$SERVE_LOG" 2>&1 &
OLLAMA_PID=$!
log "Ollama server started (PID=$OLLAMA_PID), log: $SERVE_LOG"

log "Waiting for server to become ready (GPU discovery takes ~20-25 s)..."
SERVER_READY=0
for i in $(seq 1 120); do
    kill -0 "$OLLAMA_PID" 2>/dev/null || break      # server died (e.g. port in use)
    if curl -sf -m 2 "$URL/api/version" >/dev/null; then
        log "Server is ready after ${i}s."
        SERVER_READY=1
        break
    fi
    sleep 1
done

if [ "$SERVER_READY" -eq 0 ]; then
    log "ERROR: Ollama server did not start. Last lines of $SERVE_LOG:"
    tail -n 20 "$SERVE_LOG"
    exit 1
fi

log "Ollama version: $(curl -s "$URL/api/version")"
# Must say library=CUDA. library=cpu means no GPU is used (bin/ without lib/, or a driver problem)
log "Compute       : $(grep -m1 'inference compute' "$SERVE_LOG" || echo 'no "inference compute" line found in the server log')"

###############################################################################
# Step 2: Test each model
###############################################################################

log ""
log "=========================================="
log "  Step 2: Testing models via /v1/chat/completions"
log "  Prompt: \"$TEST_PROMPT\""
log "=========================================="

for MODEL in "${MODELS[@]}"; do
    log ""
    log "------------------------------------------"
    log "  Model: $MODEL"
    log "------------------------------------------"

    # Is this exact model:tag pulled?
    if ! curl -s -m 10 "$URL/api/tags" | grep -qF "\"$MODEL\""; then
        fail "Model '$MODEL' not found in local model list. Was it pulled?"
        RESULTS+=("$MODEL|FAIL|model not found locally")
        (( FAIL++ ))
        continue
    fi

    log "Sending prompt (first request also loads the model)..."
    START=$(date +%s)
    HTTP=$(curl -s -m 300 -o "$TMP" -w '%{http_code}' "$URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$TEST_PROMPT\"}],\"max_tokens\":64,\"temperature\":0}")
    ELAPSED=$(( $(date +%s) - START ))

    if [ "$HTTP" != "200" ]; then
        fail "HTTP $HTTP after ${ELAPSED}s"
        log "  body: $(head -c 200 "$TMP")"
        RESULTS+=("$MODEL|FAIL|HTTP $HTTP after ${ELAPSED}s")
        (( FAIL++ ))
        continue
    fi

    # Same fields the project reads: choices[0].message.content and usage.{prompt,completion}_tokens
    PARSED=$(python3 - "$TMP" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
u = d.get("usage", {})
print(u.get("prompt_tokens", "?"), u.get("completion_tokens", "?"))
print(d["choices"][0]["message"]["content"].strip())
PY
)
    TOKENS="${PARSED%%$'\n'*}"
    RESPONSE="${PARSED#*$'\n'}"
    RESPONSE_LEN=${#RESPONSE}

    if [ "$RESPONSE_LEN" -lt "$MIN_RESPONSE_LEN" ]; then
        fail "Response empty or unparseable (${RESPONSE_LEN} chars) after ${ELAPSED}s"
        log "  body: $(head -c 200 "$TMP")"
        RESULTS+=("$MODEL|FAIL|empty/unparseable response after ${ELAPSED}s")
        (( FAIL++ ))
        continue
    fi

    pass "Got response (${RESPONSE_LEN} chars) in ${ELAPSED}s, tokens in/out: $TOKENS"
    log "  Response: ${RESPONSE:0:200}$([ ${#RESPONSE} -gt 200 ] && echo '...')"
    RESULTS+=("$MODEL|PASS|${ELAPSED}s, ${RESPONSE:0:100}")
    (( PASS++ ))
done

###############################################################################
# Step 3: Project CLI against the same server
###############################################################################

if [ "$RUN_CLI" = "1" ]; then
    log ""
    log "=========================================="
    log "  Step 3: run_cli.py with Ollama models"
    log "=========================================="

    WORKERS=$(printf 'ollama/%s,' "${MODELS[@]}")
    WORKERS="${WORKERS%,}"
    log "Helm: ollama/$HELM_MODEL | Workers: $WORKERS"

    [ -n "$PY_MODULE" ] && module load "$PY_MODULE"
    cd "$REPO_DIR" && . .venv/bin/activate
    export OLLAMA_BASE_URL="$URL"

    if timeout 1500 python run_cli.py --task "$CLI_TASK" \
            --helm-model "ollama/$HELM_MODEL" --worker-models "$WORKERS"; then
        pass "run_cli.py completed"
        RESULTS+=("project CLI|PASS|see output above; files in $REPO_DIR/workspaces")
        (( PASS++ ))
    else
        fail "run_cli.py exited non-zero (or hit the 25 min timeout)"
        RESULTS+=("project CLI|FAIL|see output above")
        (( FAIL++ ))
    fi
fi

###############################################################################
# Step 4: Summary
###############################################################################

log ""
log "=========================================="
log "  Step 4: Summary"
log "=========================================="
log ""
log "  Tests passed: $PASS / $(( PASS + FAIL ))"
log "  Tests failed: $FAIL / $(( PASS + FAIL ))"
log ""
log "  Results:"
for ENTRY in "${RESULTS[@]}"; do
    IFS='|' read -r NAME STATUS SNIPPET <<< "$ENTRY"
    printf "    %-20s  [%s]  %s\n" "$NAME" "$STATUS" "$SNIPPET"
done

log ""
log "Stopping Ollama server (PID=$OLLAMA_PID)..."
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null
OLLAMA_PID=""
log "Done."

# Exit with non-zero if anything failed, so SLURM marks the job accordingly
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
