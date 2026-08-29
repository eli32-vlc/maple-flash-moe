#!/bin/sh
# Create a uv venv and install mlx-lm with Maple support.
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found — install it: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ -d .venv ] || uv venv --python 3.12
uv pip install -e . rich
echo "Done. Activate with: source .venv/bin/activate"
