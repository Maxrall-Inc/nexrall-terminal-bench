#!/usr/bin/env python3
"""
Terminal-Bench 2.1 — publish step for Nexrall Code.

Takes a finished Harbor job directory (the output of `harbor run -d
terminal-bench/terminal-bench-2-1 --agent-import-path nex_terminal_bench:NexAgent
... --jobs-dir <dir>`) and publishes it to Nexrall's own benchmark page:
  - small (git, this repo): results/<model>/<run_id>/ summary.json, index.json
  - large (Cloudflare R2, public bucket "nexrall-benchmark-logs"):
      logs.tar.gz -- the whole job directory (trajectories, recordings, per-trial
      logs), gzipped, so reviewers/leaderboard maintainers can audit trajectories
      without us hosting a web viewer ourselves.
    served at https://benchmark-logs.nexrall.com/<model>/<run_id>/logs.tar.gz

A finished Harbor job directory looks like:
    <job_dir>/
      job.json                       # job-level metadata (dataset, agent, model, k, n)
      <task-name>/
        trial-1/
          result.json                 # {"reward": 1.0 or 0.0, ...} per trial
          trajectory.json
          agent/, logs/verifier/, ...
        trial-2/
        ...

This script does NOT call `harbor jobs summarize` (keeping this script's own
dependency surface small and its output format stable across Harbor
versions) — it walks result.json files directly, which is documented, stable
Harbor output.

Usage:
    export R2_ACCOUNT_ID=...
    export BENCHMARK_R2_ACCESS_KEY_ID=...
    export BENCHMARK_R2_SECRET_ACCESS_KEY=...
    python3 publish_result.py \
        --model claude-sonnet-5 \
        --run-id full-claude-sonnet-5 \
        --job-dir ./nex-tbench-results/<job-id> \
        --dataset-version terminal-bench-2-1

Requires: boto3 (S3-compatible client for R2).
"""
import argparse
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

BUCKET_NAME = "nexrall-benchmark-logs"
PUBLIC_DOMAIN = "https://benchmark-logs.nexrall.com"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_tarball(job_dir: Path, out_path: Path):
    log(f"==> Compressing {job_dir} -> {out_path}")
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(job_dir, arcname=job_dir.name)
    return out_path


def upload_to_r2(local_path: Path, key: str):
    """Upload local_path to R2 under the given object key. Public bucket,
    shared with Nexrall's other benchmark harnesses (already verified
    end-to-end for SWE-bench)."""
    import boto3  # imported lazily so --dry-run doesn't need it installed
    import os

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["BENCHMARK_R2_ACCESS_KEY_ID"]
    secret_key = os.environ["BENCHMARK_R2_SECRET_ACCESS_KEY"]

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    log(f"==> Uploading to r2://{BUCKET_NAME}/{key}")
    extra = {}
    if local_path.suffix == ".gz" or local_path.name.endswith(".tar.gz"):
        extra["ContentType"] = "application/gzip"
    elif local_path.suffix == ".json":
        extra["ContentType"] = "application/json"
    elif local_path.suffix == ".sha256":
        extra["ContentType"] = "text/plain"
    s3.upload_file(str(local_path), BUCKET_NAME, key, ExtraArgs=extra or None)
    return f"{PUBLIC_DOMAIN}/{key}"


def find_trial_results(job_dir: Path):
    """Walk <job_dir>/<task-name>/<trial-name>/result.json and group rewards
    by task, so we can compute both an overall pass rate and a per-task
    pass@k the way the official leaderboard reports it."""
    per_task = {}
    for result_file in sorted(job_dir.glob("*/*/result.json")):
        task_name = result_file.parent.parent.name
        try:
            data = json.loads(result_file.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log(f"  ! skipping unreadable {result_file}: {exc}")
            continue
        reward = data.get("reward")
        if reward is None:
            continue
        per_task.setdefault(task_name, []).append(float(reward))
    return per_task


def summarize(per_task: dict[str, list[float]]):
    n_tasks = len(per_task)
    if n_tasks == 0:
        return {"n_tasks": 0, "n_trials": 0, "n_resolved_tasks": 0, "pct_resolved": 0.0, "pass_at_1": 0.0}

    n_trials = sum(len(v) for v in per_task.values())
    # A task counts as "resolved" for pass@1-style reporting if the majority
    # (or, with an odd k, strict majority) of its trials succeeded — mirrors
    # how Terminal-Bench's own leaderboard aggregates repeated trials before
    # averaging across tasks, rather than treating every trial as its own
    # independent benchmark instance.
    resolved_tasks = 0
    trial_level_successes = 0
    for rewards in per_task.values():
        trial_level_successes += sum(1 for r in rewards if r >= 1.0)
        if sum(1 for r in rewards if r >= 1.0) > len(rewards) / 2:
            resolved_tasks += 1

    return {
        "n_tasks": n_tasks,
        "n_trials": n_trials,
        "n_resolved_tasks": resolved_tasks,
        "pct_resolved": round(100.0 * resolved_tasks / n_tasks, 2),
        "pass_at_1": round(100.0 * trial_level_successes / n_trials, 2) if n_trials else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="real model id, e.g. claude-sonnet-5")
    ap.add_argument("--run-id", required=True, help="label for this run, e.g. full-claude-sonnet-5")
    ap.add_argument("--job-dir", required=True, help="path to the finished Harbor job directory")
    ap.add_argument(
        "--dataset-version",
        default="terminal-bench-2-1",
        help="Terminal-Bench dataset id/version this run used",
    )
    ap.add_argument(
        "--results-root",
        default=str(Path(__file__).parent / "results"),
        help="git-tracked directory to copy summary.json/index.json into",
    )
    ap.add_argument("--run-date", default="", help="ISO date the run completed (defaults to today)")
    ap.add_argument("--dry-run", action="store_true", help="build artifacts locally but skip the R2 upload")
    args = ap.parse_args()

    job_dir = Path(args.job_dir).resolve()
    if not job_dir.is_dir():
        sys.exit(f"job dir not found: {job_dir}")

    per_task = find_trial_results(job_dir)
    if not per_task:
        sys.exit(
            f"No result.json files found under {job_dir}/*/*/result.json — "
            "is this a finished Harbor job directory?"
        )
    summary = summarize(per_task)
    log(
        f"==> {summary['n_resolved_tasks']}/{summary['n_tasks']} tasks resolved "
        f"({summary['pct_resolved']}%), {summary['n_trials']} trials total, "
        f"pass@1={summary['pass_at_1']}%"
    )

    run_date = args.run_date or __import__("datetime").date.today().isoformat()

    # 1. Small artifacts -> git-tracked results/<model>/<run_id>/
    dest = Path(args.results_root) / args.model / args.run_id
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = dest / "summary.json"
    summary_path.write_text(
        json.dumps({"per_task_rewards": per_task, **summary}, indent=2, sort_keys=True) + "\n"
    )
    log(f"==> Wrote {summary_path}")

    # 2. Large artifact -> tarball of the whole job dir (trajectories, casts,
    #    verifier logs), checksum, upload to R2.
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "logs.tar.gz"
        make_tarball(job_dir, tarball)
        checksum = sha256_file(tarball)
        checksum_path = Path(tmp) / "logs.tar.gz.sha256"
        checksum_path.write_text(f"{checksum}  logs.tar.gz\n")

        size_bytes = tarball.stat().st_size
        log(f"==> logs.tar.gz: {size_bytes / 1e6:.1f} MB, sha256={checksum}")

        logs_url = f"{PUBLIC_DOMAIN}/{args.model}/{args.run_id}/logs.tar.gz"
        if not args.dry_run:
            logs_url = upload_to_r2(tarball, f"{args.model}/{args.run_id}/logs.tar.gz")
            upload_to_r2(checksum_path, f"{args.model}/{args.run_id}/logs.tar.gz.sha256")
        else:
            log("==> --dry-run set: skipping R2 upload, using computed URL as a placeholder")

    # 3. index.json — everything the results table on benchmark.nexrall.com
    #    needs, in one place, so numbers are never hand-typed twice.
    index = {
        "benchmark": "terminal-bench",
        "dataset_version": args.dataset_version,
        "model": args.model,
        "run_id": args.run_id,
        "run_date": run_date,
        **summary,
        "logs_url": logs_url,
        "logs_sha256": checksum,
        "logs_size_bytes": size_bytes,
        "summary_path": f"results/{args.model}/{args.run_id}/summary.json",
    }
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    log(f"==> Wrote {dest / 'index.json'}")

    # 4. Print the HTML <tr> a human can paste into web/benchmark/terminal-bench.md
    repo_base = "https://github.com/Maxrall-Inc/nexrall-terminal-bench/blob/main"
    print("\n==> Paste this row into web/benchmark/terminal-bench.md:\n")
    print("    <tr>")
    print(f"      <td><code>{args.model}</code></td>")
    print(f"      <td class=\"metric-big\">{summary['n_resolved_tasks']}</td>")
    print(f"      <td class=\"metric-big\">{summary['pct_resolved']}%</td>")
    print(f"      <td>{summary['n_tasks']}</td>")
    print(f"      <td>{run_date}</td>")
    print("      <td>")
    print(f"        <a href=\"{repo_base}/{index['summary_path']}\">summary</a> \u00b7")
    print(f"        <a href=\"{logs_url}\">logs</a>")
    print(f"        (sha256: <code>{checksum[:12]}\u2026</code>)")
    print("      </td>")
    print("    </tr>")

    print(
        "\nNOTE: this is Nexrall's OWN site numbers, not an official Terminal-Bench "
        "leaderboard entry until a full 5-trial run is uploaded via `harbor upload "
        "--public` and a PR is merged against harbor-framework/terminal-bench-2-1 "
        "— see README.md 'Official leaderboard submission' section.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
