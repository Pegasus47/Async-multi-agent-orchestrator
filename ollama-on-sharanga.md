# Running Ollama on Sharanga HPC (no sudo) + connecting it to localhost

Status: tested 2026-10-01 on `gpunode8.sharanga.local` (RTX PRO 6000 Blackwell, driver 580, CUDA 13), Ollama v0.16.0.
Project: async_MultiAgent: OSS models served from HPC, consumed by a local chat frontend.

## Key facts (read first)

- **No sudo is needed.** Ollama is a single binary plus a `lib/` folder and runs from your home dir.
- **Ollama is always client + server.** `ollama run`, `ollama list` and even `ollama --version` talk over HTTP to `ollama serve` on `127.0.0.1:11434`. You must start `ollama serve` yourself; there is no background service.
- **Never run the server on the login node (`hpc01`).** Processes there are killed automatically. Use `srun` or `sbatch` to get a compute node.
- **Home dirs are private** (`/home/<user>`). Each person needs their own binary and models. The quota is 40 GiB. `$SCRATCH` has no limit but deletes files 15 days after they were last modified.
- **Partition to use:** `gpu_rtx_pro_6000_6_csis_hyd` (6× RTX PRO 6000 Blackwell, 96 GB each, 2-day max). The FAQ's `gpu` partition name is outdated; check with `sinfo -o "%P %a %l %D %G"`.

## Directory layout (keep `bin/` and `lib/` side by side)

```
~/ollama/bin/ollama          # binary
~/ollama/lib/ollama/...      # CUDA runtime libs (cuda_v12, cuda_v13). Without these: CPU only
~/openSourceModels/models/   # OLLAMA_MODELS: blobs/ manifests/
```

## 1. One-time setup (login node)

```bash
# Install the binary. Downloading is fine on the login node; running models is not.
mkdir -p ~/ollama && cd ~/ollama
curl -L -o ollama.tar.zst https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
tar --zstd -xf ollama.tar.zst && rm ollama.tar.zst     # produces bin/ and lib/

# Environment
cat >> ~/.bashrc <<'EOF'
export PATH=$HOME/ollama/bin:$PATH
export OLLAMA_MODELS=$HOME/openSourceModels/models
EOF
source ~/.bashrc
ollama --version     # "could not connect to a running Ollama instance" here is expected

# Pull models: run a server briefly, pull, then kill it
mkdir -p $OLLAMA_MODELS
ollama serve > /dev/null 2>&1 &
sleep 5
ollama pull qwen2.5:1.5b     # current set: gemma2:2b, qwen2.5:1.5b, smollm:1.7b
kill %1
```

`async_MultiAgent/downloadModels.sh` also pulls models, but it moves the binary away from its `lib/` folder. Don't use its install step, or GPU detection will break.

## 2. Start the server on a GPU node (interactive)

```bash
srun -p gpu_rtx_pro_6000_6_csis_hyd -N 1 -n 1 -c 8 --mem 64G \
     --gres=gpu:nvidia_rtx_pro_6000:1 -t 0-4:00 --pty bash -i

hostname                          # e.g. gpunode8.sharanga.local, needed for the tunnel
unset CUDA_VISIBLE_DEVICES        # REQUIRED: see Issue 2
nohup ollama serve > ~/ollama/serve.log 2>&1 &
sleep 40                          # GPU discovery takes about 20–25 s
grep "inference compute" ~/ollama/serve.log    # must say library=CUDA, not library=cpu
ollama list
ollama run qwen2.5:1.5b "say hi"
```

Keep this terminal open. The server stops when the `srun` session ends.

## 3. Connect your laptop (SSH tunnel)

On your laptop, replacing the node name with your `hostname` output:

```bash
ssh -N -L 11500:127.0.0.1:11434 -J <user>@hpc.bits-hyderabad.ac.in <user>@gpunode8.sharanga.local
```

No output means it's working. Test it:

```bash
curl http://localhost:11500/api/tags
curl http://localhost:11500/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"say hi"}]}'
```

**Frontend/backend endpoint:** `http://localhost:11500`. Use `/v1/chat/completions` (OpenAI-compatible) or `/api/chat` (Ollama native). Local port 11500 avoids clashing with a local Ollama install on 11434.

**Fallback if the `-J` hop into the compute node is refused:**
1. On the node, start the server listening on all interfaces: `OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve > ~/ollama/serve.log 2>&1 &`
2. On the laptop: `ssh -N -L 11500:gpunode8.sharanga.local:11434 <user>@hpc.bits-hyderabad.ac.in`

Note that anyone on the cluster can reach your server while it listens on `0.0.0.0`.

## 4. Optional: batch job instead of an open terminal (untested)

```bash
#!/bin/bash
#SBATCH -p gpu_rtx_pro_6000_6_csis_hyd
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH -t 0-8:00
#SBATCH --job-name=ollama_server
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
export PATH=$HOME/ollama/bin:$PATH
export OLLAMA_MODELS=$HOME/openSourceModels/models
unset CUDA_VISIBLE_DEVICES
echo "NODE: $(hostname)"
ollama serve
```

Submit with `sbatch ollama_server.sh`. Get the node name with `grep NODE slurm.<jobid>.out`, then open the tunnel as in step 3. Stop it with `scancel <jobid>`.

## Issues seen and fixes

| # | Symptom | Cause | Fix |
|---|---------|-------|-----|
| 1 | `ollama: command not found` while inside `~/ollama/bin` | Folder not on `PATH` | Add `export PATH=$HOME/ollama/bin:$PATH` to `~/.bashrc`, or call `./ollama` |
| 2 | `ollama list`, `--version` and `run` hang for about 60 s; log shows `failure during GPU discovery ... failed to finish discovery before timeout`, then `library=cpu`, `total_vram="0 B"` | Slurm sets `CUDA_VISIBLE_DEVICES=0` and Ollama's GPU probe times out with it | `unset CUDA_VISIBLE_DEVICES` before `ollama serve`. Slurm still limits the job to its allocated GPU. The log should then show `library=CUDA ... RTX PRO 6000` |
| 3 | `srun: error: invalid partition specified: gpu` | The FAQ's partition names are outdated | Use `sinfo -o "%P %a %l %D %G"`, then `gpu_rtx_pro_6000_6_csis_hyd` |
| 4 | `ollama list` is empty although models exist | `OLLAMA_MODELS` points to the wrong folder | Point it at the folder containing `blobs/` and `manifests/` (`~/openSourceModels/models`) |
| 5 | Very slow replies | Running on the CPU (see #2) | Fix #2. On the GPU a 1.5B model replies in about a second |
| 6 | First request after start takes about a minute | GPU discovery (~22 s) plus loading the model from the network home dir | Expected once per start. The model stays loaded for 5 min by default (`OLLAMA_KEEP_ALIVE`, e.g. `30m`) |
| 7 | `address already in use` in the log | Someone else's Ollama on the same shared node | Use a personal port: `export OLLAMA_HOST=127.0.0.1:11500` for both the server and CLI, and tunnel to that port |
| 8 | Client hangs on a cluster with a proxy | `http_proxy` routes localhost through the proxy | `export NO_PROXY=localhost,127.0.0.1,.sharanga.local no_proxy=$NO_PROXY`. Not an issue on gpunode8 as of 2026-10-01 |
| 9 | Server dies unexpectedly | The `srun` session ended or hit its `-t` limit, or it was run on the login node | Keep the terminal open, use the sbatch script, or request more time (max 2 days on this partition) |

## Quick checks

```bash
squeue -u $USER                                   # is my job running, and on which node
tail -f ~/ollama/serve.log                        # server log
curl -s http://127.0.0.1:11434/api/version        # on the node: is the server up
nvidia-smi                                        # on the node: is ollama using the GPU
```
