#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000 "$@"
