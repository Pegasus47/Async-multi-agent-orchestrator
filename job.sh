#!/bin/bash
###############################################################################
# job.sh — SLURM job script for Sharanga HPC
#
# Downloads the Ollama binary and three ~2B parameter models directly on HPC,
# stores everything in $HOME/openSourceModels/.
#
# Usage:
#   sbatch job.sh
###############################################################################

#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 8G
#SBATCH -t 0-2:00                # time (D-HH:MM)
#SBATCH --job-name="ollama_download"
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-user=<username>@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL

###############################################################################
# Configuration
###############################################################################

BASE_DIR="$HOME/openSourceModels"
OLLAMA_BIN="$BASE_DIR/ollama"
MODELS_DIR="$BASE_DIR/models"

# Three ~2B parameter models
MODELS=(
    "gemma2:2b"
    "qwen2.5:1.5b"
    "smollm:1.7b"
)

###############################################################################
# Step 1: Download and install Ollama binary (no sudo needed)
###############################################################################

echo "=========================================="
echo "  Step 1: Installing Ollama"
echo "=========================================="

mkdir -p "$BASE_DIR"
mkdir -p "$MODELS_DIR"

if [ ! -f "$OLLAMA_BIN" ]; then
    echo "Downloading Ollama binary..."

    # Download the tar.zst archive from GitHub
    curl -L -o "$BASE_DIR/ollama-linux-amd64.tar.zst" \
        https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst

    # Extract — try tar with zstd support first, fall back to manual zstd
    if tar --zstd -xf "$BASE_DIR/ollama-linux-amd64.tar.zst" -C "$BASE_DIR" 2>/dev/null; then
        echo "Extracted with tar --zstd"
    elif command -v zstd &>/dev/null; then
        zstd -d "$BASE_DIR/ollama-linux-amd64.tar.zst" -o "$BASE_DIR/ollama-linux-amd64.tar"
        tar -xf "$BASE_DIR/ollama-linux-amd64.tar" -C "$BASE_DIR"
        rm -f "$BASE_DIR/ollama-linux-amd64.tar"
        echo "Extracted with zstd + tar"
    else
        echo "ERROR: Cannot extract .tar.zst — neither tar --zstd nor zstd are available."
        echo "Ask your HPC admin to install zstd, or try: module load zstd"
        exit 1
    fi

    # The archive extracts to bin/ollama — move it to our expected location
    if [ -f "$BASE_DIR/bin/ollama" ]; then
        mv "$BASE_DIR/bin/ollama" "$OLLAMA_BIN"
        rm -rf "$BASE_DIR/bin"
    fi

    chmod +x "$OLLAMA_BIN"
    rm -f "$BASE_DIR/ollama-linux-amd64.tar.zst"
    echo "Ollama installed at: $OLLAMA_BIN"
else
    echo "Ollama binary already exists, skipping download."
fi

export PATH="$(dirname "$OLLAMA_BIN"):$PATH"
export OLLAMA_MODELS="$MODELS_DIR"

echo "Ollama version:"
"$OLLAMA_BIN" --version

###############################################################################
# Step 2: Start Ollama server
###############################################################################

echo ""
echo "=========================================="
echo "  Step 2: Starting Ollama server"
echo "=========================================="

"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
echo "Ollama server started with PID: $OLLAMA_PID"

echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if "$OLLAMA_BIN" list &>/dev/null; then
        echo "Server is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Server did not start within 30 seconds."
        kill "$OLLAMA_PID" 2>/dev/null
        exit 1
    fi
    sleep 1
done

###############################################################################
# Step 3: Download each model
###############################################################################

echo ""
echo "=========================================="
echo "  Step 3: Downloading models"
echo "=========================================="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "  Pulling: $MODEL"
    echo "------------------------------------------"
    "$OLLAMA_BIN" pull "$MODEL"

    if [ $? -eq 0 ]; then
        echo "SUCCESS: Downloaded $MODEL"
    else
        echo "FAILED: Could not download $MODEL"
    fi
done

###############################################################################
# Step 4: Verify
###############################################################################

echo ""
echo "=========================================="
echo "  Step 4: Verification"
echo "=========================================="

echo "Downloaded models:"
"$OLLAMA_BIN" list

echo ""
echo "Storage location: $MODELS_DIR"
echo "Disk usage:"
du -sh "$MODELS_DIR"

###############################################################################
# Cleanup
###############################################################################

echo ""
echo "Stopping Ollama server..."
kill "$OLLAMA_PID" 2>/dev/null
wait "$OLLAMA_PID" 2>/dev/null
echo "Done. Models are ready at: $BASE_DIR"
