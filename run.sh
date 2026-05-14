#!/usr/bin/env bash
# Linux / macOS quick-start.
#   ./run.sh setup    # create venv + install deps
#   ./run.sh dev      # run with auto-reload
#   ./run.sh start    # run (no reload)
#   ./run.sh test     # run pytest
#   ./run.sh index    # build vector index via API
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

ensure_venv() {
    if [[ ! -x "$PY" ]]; then
        echo ">> Creating virtualenv in $VENV"
        python3 -m venv "$VENV"
    fi
}

case "${1:-start}" in
    setup)
        ensure_venv
        "$PY" -m pip install --upgrade pip
        "$PY" -m pip install -r requirements.txt
        [[ -f .env ]] || cp .env.example .env
        ;;
    dev)
        ensure_venv
        "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    start)
        ensure_venv
        "$PY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
        ;;
    test)
        ensure_venv
        "$PY" -m pytest -q
        ;;
    index)
        curl -fsS -X POST http://localhost:8000/api/index | jq .
        ;;
    *)
        echo "Unknown command: $1"
        exit 1
        ;;
esac
