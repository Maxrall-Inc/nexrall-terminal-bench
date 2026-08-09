# Terminal-Bench 2.1 — Nexrall Code benchmark harness

Measures how well [Nexrall Code](https://github.com/Maxrall-Inc/nexrall-code)
(the `nex` CLI) completes real terminal-based engineering tasks — file
inspection, debugging, package installs, tests — inside sandboxed Docker
containers, using the **official** [Terminal-Bench
2.1](https://github.com/harbor-framework/terminal-bench-2-1) dataset and its
official runner, [Harbor](https://github.com/harbor-framework/harbor).

This repo is standalone and public so the official leaderboard's maintainers
(and anyone else) can audit the harness and trajectories independently of
Nexrall Code's own (private) source repository. It grades the terminal
session itself — an agent that never opens a file it needed to inspect fails
here even if a lucky patch happens to work, because scoring runs a
task-specific verifier against the *live* container state, not a diff.

## How it works (Harbor-native, no host cloning)

Everything happens **inside Harbor's own task containers**:

1. **Harbor** downloads the pinned `terminal-bench/terminal-bench-2-1` task
   set, builds/pulls each task's Docker image, and starts a container per
   trial.
2. Our adapter — `src/nex_terminal_bench/nex_agent.py`'s `NexAgent`, a
   subclass of Harbor's `BaseInstalledAgent` — installs `nex` inside that
   container (`npm install -g nexrall-code`) and runs it headlessly:
   ```
   nex --output-format stream-json --yolo --no-banner -p "<task instruction>"
   ```
3. Harbor's own verifier (a second, isolated container per task) then runs
   the task's hidden test suite against whatever state the agent left behind
   and writes `result.json` (`{"reward": 1.0 or 0.0}`).
4. `publish_result.py` walks the finished job directory, aggregates
   pass-rate, and publishes to Nexrall's own benchmark page + Cloudflare R2
   (trajectories/logs).

## Why `--yolo` is safe here

Harbor task containers are throwaway sandboxes, not Nexrall's own
infrastructure. There is no TTY inside them to answer a permission prompt,
so headless auto-approve (`--yolo`) is required for `nex` to do anything at
all. The one guardrail kept is `NEXRALL_ALLOW_DESTRUCTIVE` — deliberately
**never** set by `nex_agent.py` — so a confused or prompt-injected agent
still cannot run a DB drop / force-push / `terraform destroy`-class command
even under blanket auto-approval; it can only do that if the task's
*verifier itself* requires it, which none of Terminal-Bench 2.1's 89 tasks
do.

## Anti-reward-hacking: fetch blocklist

The official leaderboard's rules forbid an agent from looking up the
Terminal-Bench website or GitHub repo mid-task (an easy way to fetch a
published solution instead of solving the task). `nex_agent.py` sets
`NEXRALL_FETCH_BLOCKLIST` to cover `tbench.ai`, `github.com` and its CDN
domains, and `huggingface.co` — reusing the `nex` CLI's existing
`fetch_url`-blocking mechanism rather than trusting the model to just not
look.

## Setup

```bash
bash setup.sh
```

Installs Docker, `uv`, Harbor, and this adapter package (editable install)
into a local `.venv`. Safe to re-run.

## Running

```bash
source .venv/bin/activate
export NEXRALL_TOKEN=...          # headless auth for the nex CLI (nex login --print-token)

# 1. Smoke test — 5 tasks, proves the whole pipeline works before spending
#    real money on all 89.
bash run.sh --smoke

# 2. One task, for debugging a specific failure.
bash run.sh --task <task-name>

# 3. Full run — all 89 tasks, k=5 trials each (leaderboard-eligible trial
#    count), local Docker, concurrency 4 by default.
CONCURRENCY=8 bash run.sh --full
```

Cost & infrastructure: Terminal-Bench containers are much lighter than
SWE-bench's per-repo images (no multi-GB scientific-Python environments to
build), so a single EC2 `c6i.4xlarge` (8 vCPU / 16 GiB, [AWS's published
spec](https://aws.amazon.com/ec2/instance-types/c6i/)) comfortably runs
`CONCURRENCY=8` — that stays under the official harness's own guidance of
using no more than ~0.75 × vCPU count for parallel workers. Raise
`CONCURRENCY` only if you've confirmed the box isn't CPU- or
Docker-daemon-bound at your chosen value — pushing it higher can *slow down*
a run through resource contention, not speed it up.

A full k=5 run (89 tasks × 5 trials = 445 trials) takes on the order of
hours, not minutes, and burns real model API spend — always run `--smoke`
first on a new model/agent config.

## Publishing to benchmark.nexrall.com

```bash
python3 publish_result.py \
    --model deepseek-v4-pro \
    --run-id full-deepseek-v4-pro \
    --job-dir ./nex-tbench-results/<job-id>
```

Requires `boto3` + R2 credentials (`R2_ACCOUNT_ID`,
`BENCHMARK_R2_ACCESS_KEY_ID`, `BENCHMARK_R2_SECRET_ACCESS_KEY`) for
Nexrall's public logs bucket (`nexrall-benchmark-logs`). Add `--dry-run` to
build artifacts locally without uploading. Prints an HTML `<tr>` to paste
into Nexrall's benchmark page.

## Official Terminal-Bench leaderboard submission

**Status as of 2026-08-09: harness-ready, run-not-yet-done.** Three things
the official leaderboard's automated validation
([`harborframework/terminal-bench-2-leaderboard`](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard))
requires, tracked deliberately rather than silently:

1. **ATIF trajectories — done.** `nex_terminal_bench/atif.py`'s
   `convert_stream_json_to_trajectory()` converts `nex`'s `--output-format
   stream-json` event log into a real ATIF-v1.7 `Trajectory`, wired into
   `nex_agent.py`'s `populate_context_post_run` (`SUPPORTS_ATIF = True`).
   `nex`'s stream-json format carries no per-tool-call ids of its own —
   `atif.py` works around this with a FIFO-per-tool-name pairing strategy;
   see that module's own `KNOWN LIMITATION` docstring for exactly when that
   could mis-pair two same-named calls within one turn.
   `FinalMetrics.total_cost_usd` is computed from a fixed vendor price table
   (real vendor cost, not Nexrall's own marked-up selling price — the same
   quantity every other agent's leaderboard cost figure represents).
2. **`metadata.yaml` — done.** See `metadata.yaml` in this directory
   (`agent_url`, `agent_display_name`, `agent_org_display_name`, `models`).
3. **Real completed run — not yet done.** The leaderboard requires ≥5
   trials per task (`-k 5`), uploaded publicly to Harbor Hub (`harbor auth
   login` once, then `harbor upload <job-dir> --public` or `--upload
   --public` on the run itself), then a PR against
   `harbor-framework/terminal-bench-2-1` (`leaderboard/SUBMIT.md`) that CI
   auto-validates and a maintainer reviews before merging as a new
   leaderboard row.

Until the full run + upload + PR are done, results published on
benchmark.nexrall.com are **Nexrall's own reproducible numbers**, clearly
labeled as unofficial (not a leaderboard entry).

## Repo history

This harness previously lived inside Nexrall Code's main (private)
monorepo at `packages/bench/terminal-bench/`. It was extracted into this
standalone public repo specifically so `agent_url` in `metadata.yaml` — a
required field for the official leaderboard submission — points somewhere
a maintainer or the public can actually open, instead of a 404 behind a
private repo.
