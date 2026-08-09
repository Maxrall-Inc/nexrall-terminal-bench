#!/usr/bin/env bash
# One-time setup for the Terminal-Bench benchmark box (Ubuntu 22.04 x86_64).
# Installs Docker, uv, Harbor, and this adapter package. Safe to re-run.
set -euo pipefail

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y docker.io python3-pip python3-venv git curl unzip jq
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

echo "==> uv (Harbor's installer of choice)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "==> Harbor"
uv tool install harbor
export PATH="$HOME/.local/bin:$PATH"
harbor --version || true

echo "==> This adapter package (nex_terminal_bench)"
cd "$(dirname "$0")"
uv venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev]"

echo
echo "Setup complete."
echo "IMPORTANT: log out and back in (or 'newgrp docker') so Docker works without sudo."
echo
echo "Next:"
echo "  source .venv/bin/activate"
echo "  export NEXRALL_TOKEN=...           # headless CLI auth"
echo "  export ANTHROPIC_API_KEY=...       # only if Harbor's own cost estimation needs it"
echo "  bash run.sh --smoke                # 5 tasks, cheap, proves the pipeline"
echo "  bash run.sh --full                 # all 89 Terminal-Bench 2.1 tasks, k=5 trials each"
