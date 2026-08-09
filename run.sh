#!/usr/bin/env bash
# Run Nexrall Code (`nex`) against Terminal-Bench 2.1 via Harbor.
#
# Usage:
#   bash run.sh --smoke                 # 5 tasks, k=1 — proves the pipeline
#   bash run.sh --full                  # all tasks, k=5 — leaderboard-eligible run
#   bash run.sh --task <task-name>      # single task, k=1 — debugging one task
#
# Flags verified against the installed Harbor CLI (`harbor run --help`,
# harbor 0.20.0): there is no separate --agent-import-path flag — --agent/-a
# itself accepts either a built-in agent name or a "module.path:ClassName"
# import path. -l/--n-tasks limits task count; -i/--include-task-name selects
# one task by name; -d/--dataset takes "name@version" (version omitted here
# resolves to "latest", which is the current terminal-bench-2-1 release).
#
# Env vars:
#   NEXRALL_TOKEN               required — headless CLI auth for `nex`
#   MODEL                       optional — provider/model id Harbor passes
#                               through to the adapter (default:
#                               anthropic/claude-sonnet-5)
#   CONCURRENCY                 optional — Harbor's -n flag (default: 4,
#                               local Docker)
#   JOBS_DIR                    optional — where Harbor writes job output
#                               (default: ./nex-tbench-results)
#   AGENT_TIMEOUT_MULTIPLIER    optional — Harbor's --agent-timeout-multiplier
#                               (default: 1.0, i.e. the dataset's own
#                               task.toml [agent].timeout_sec verbatim — 900s
#                               for most terminal-bench-2-1 tasks, verified
#                               against the locally-cached task.toml files).
#                               Left at 1.0 for --full/--smoke so leaderboard
#                               runs stay comparable to the official dataset
#                               budget. Raise this (e.g. 1.5) when debugging a
#                               single task that keeps hitting Harbor's own
#                               AgentTimeoutError, so a model-side stream
#                               stall isn't indistinguishable from a
#                               genuinely stuck agent.
#

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${NEXRALL_TOKEN:-}" ]; then
  echo "Error: NEXRALL_TOKEN must be set (headless auth for the nex CLI)." >&2
  exit 1
fi

if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running." >&2
  exit 1
fi

# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate

MODEL="${MODEL:-anthropic/claude-sonnet-5}"
CONCURRENCY="${CONCURRENCY:-4}"
JOBS_DIR="${JOBS_DIR:-./nex-tbench-results}"
AGENT_TIMEOUT_MULTIPLIER="${AGENT_TIMEOUT_MULTIPLIER:-1.0}"
DATASET="terminal-bench/terminal-bench-2-1"
AGENT="nex_terminal_bench:NexAgent"

MODE="${1:-}"
case "$MODE" in
  --smoke)
    echo "==> Smoke test: 5 tasks, k=1"
    harbor run \
      --dataset "$DATASET" \
      --agent "$AGENT" \
      --model "$MODEL" \
      --n-tasks 5 \
      --n-concurrent "$CONCURRENCY" \
      --agent-timeout-multiplier "$AGENT_TIMEOUT_MULTIPLIER" \
      --jobs-dir "$JOBS_DIR" \
      --yes
    ;;
  --task)
    TASK="${2:?usage: run.sh --task <task-name>}"
    echo "==> Single task: $TASK"
    harbor run \
      --dataset "$DATASET" \
      --agent "$AGENT" \
      --model "$MODEL" \
      --include-task-name "$TASK" \
      --agent-timeout-multiplier "$AGENT_TIMEOUT_MULTIPLIER" \
      --jobs-dir "$JOBS_DIR" \
      --yes
    ;;
  --full)
    echo "==> Full run: all 89 tasks, k=5 (leaderboard-eligible trial count)"
    echo "    This costs real API \$\$ and takes hours. Ctrl+C now to abort."
    sleep 5
    harbor run \
      --dataset "$DATASET" \
      --agent "$AGENT" \
      --model "$MODEL" \
      --n-attempts 5 \
      --n-concurrent "$CONCURRENCY" \
      --agent-timeout-multiplier "$AGENT_TIMEOUT_MULTIPLIER" \
      --jobs-dir "$JOBS_DIR" \
      --yes
    ;;
  *)
    echo "Usage: $0 --smoke | --full | --task <task-name>" >&2
    exit 1
    ;;
esac

echo
echo "==> Done. Job output in $JOBS_DIR"
echo "    Summarize:  harbor jobs summarize $JOBS_DIR/<job-id>"
echo "    View:       harbor view $JOBS_DIR"
