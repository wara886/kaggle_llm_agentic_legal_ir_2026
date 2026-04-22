#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/submit_config.json"

echo "[run_submit.sh] Script dir: $SCRIPT_DIR"
echo "[run_submit.sh] Config: $CONFIG_PATH"

python "$SCRIPT_DIR/make_submission.py" --config "$CONFIG_PATH"
python "$SCRIPT_DIR/preflight_check.py" --config "$CONFIG_PATH"

echo "[run_submit.sh] Done."
