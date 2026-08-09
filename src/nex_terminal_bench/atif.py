"""Converts `nex`'s `--output-format stream-json` event log into a real ATIF
(Agent Trajectory Interchange Format) trajectory, so a Terminal-Bench run can
actually be submitted to the official leaderboard.

Why this exists (see nex_agent.py's SUPPORTS_ATIF docstring for the full
context): the official leaderboard's automated validation rejects any
rewarded trial with no Hub `trajectory_path` specifically so a maintainer's
audit step can check for reward hacking. Before this module, `nex_agent.py`
only wrote a best-effort `trajectory.json` summary (final_result + event
count) — not a real ATIF document, so `SUPPORTS_ATIF` stayed `False`.

Design mirrors harbor's own `harbor/agents/installed/claude_code.py`
`_convert_events_to_trajectory` (the only other installed-CLI adapter with a
real converter as of harbor==0.20.0): same two-pass shape (normalize raw
events into an intermediate list, then convert each to a `Step`), same
metrics-summation logic for `FinalMetrics`. `nex`'s stream-json format is
much flatter than Claude Code's session JSONL (no sidechains, no per-message
`id` to bundle multiple tool_use blocks under — see the module-level note on
`STEP GRANULARITY` below), so the conversion here is a single linear pass
rather than Claude Code's two-pass turn-bundling logic.

KNOWN LIMITATION, by design of the source format, not a bug in this module:
`nex`'s stream-json log carries no per-tool-call id at all (`tool_use` events
have only `{tool, input}`, `tool_result` events only `{tool, ok, error?}` —
confirmed against packages/core/src/agent/loop.ts's onToolUse/onToolResult
callback signatures, which pass name+input/name+result but never an id). A
turn with N calls to the SAME tool name is therefore ambiguous: this module
pairs each `tool_result` with the OLDEST still-unresolved `tool_use` of the
same tool name (FIFO per tool name), which is correct whenever calls to one
tool name do not themselves race out of completion order — true for `nex`
today because packages/core/src/agent/loop.ts's `Promise.all` over one turn's
tool_use blocks preserves each promise's SETTLEMENT is unordered, but the ATIF
step this produces is still valid (every call gets *some* real result,
correctly attributed BY NAME) even in the rare case where two same-named
calls in one turn resolve out of issue order and get their outputs swapped.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

# ── Vendor pricing (USD per 1M tokens), for `FinalMetrics.total_cost_usd` ────
#
# Real vendor list price, NOT what Nexrall charges its own users (which
# includes model_pricing's markup column) — a benchmark cost figure must
# reflect what running this agent actually costs to operate, the same
# quantity Terminal-Bench's own leaderboard reports for every other agent.
# Copied from backend/migrations/113_multi_provider_pricing.sql and
# 114_deepseek_qwen_pricing.sql's `in_per_1m`/`out_per_1m` columns (checked
# 2026-08-09 against the live prod DB, not just the migration file, since a
# price can be edited from the admin UI after the migration lands).
_VENDOR_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    # (input $/1M, output $/1M)
    "claude-sonnet-5": (1.0, 15.0),
    "claude-opus-5": (2.5, 25.0),
    "claude-fable-5": (2.0, 50.0),
    "gpt-5.4": (5.0, 22.5),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o-mini": (0.15, 0.6),
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.14, 0.28),
    "qwen3.7-max": (2.5, 7.5),
}


def _estimate_cost_usd(
    model_name: str | None, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Best-effort vendor cost from the fixed price table above.

    Returns None (rather than 0.0) for an unpriced model so a caller can tell
    "genuinely free" apart from "we don't have a price for this" — Harbor's
    own FinalMetrics.total_cost_usd is Optional for exactly this reason.
    """
    if not model_name:
        return None
    prices = _VENDOR_PRICE_PER_1M.get(model_name)
    if prices is None:
        return None
    in_price, out_price = prices
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


def _read_events(output_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not output_path.exists():
        return events
    with open(output_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def convert_stream_json_to_trajectory(
    output_path: Path,
    *,
    agent_version: str,
    default_model_name: str | None,
    instruction: str,
    session_id: str | None = None,
) -> Trajectory | None:
    """Convert one trial's `nex-output.jsonl` into an ATIF Trajectory.

    Args:
        output_path: path to the trial's stream-json log
            (EnvironmentPaths.agent_dir / "nex-output.jsonl" inside the
            container, already collected by Harbor into the trial's logs dir
            by the time populate_context_post_run runs).
        agent_version: `nex --version` output, for Trajectory.agent.version.
        default_model_name: the --model flag's resolved value, used when a
            'start' event's own `model` field is somehow missing.
        instruction: the task instruction actually sent to `nex` on stdin —
            becomes the trajectory's first (source="user") step, matching
            every other installed-agent adapter's convention that step 1 is
            the task prompt, not the agent's first response.
        session_id: optional external session identifier (none available
            from nex's stream-json log today; left for a future version that
            adds one).

    Returns:
        A valid ATIF Trajectory, or None if the log was empty/unparseable
        (mirrors claude_code.py's `_convert_events_to_trajectory` — a caller
        should fall back to writing nothing rather than a broken trajectory).
    """
    events = _read_events(output_path)
    if not events:
        return None

    model_name = default_model_name
    for event in events:
        if event.get("type") == "start" and isinstance(event.get("model"), str):
            model_name = event["model"]
            break

    steps: list[Step] = []

    # Step 1: the task instruction, as a user-sourced step — every installed
    # agent adapter in harbor treats the FIRST step as the prompt that
    # started the run, not the agent's own first action.
    if instruction.strip():
        steps.append(
            Step(
                step_id=len(steps) + 1,
                source="user",
                message=instruction,
            )
        )

    # ── STEP GRANULARITY ────────────────────────────────────────────────────────────────────────────────
    # nex's stream-json log is a flat sequence of small events (`text` deltas
    # token-by-token, `tool_use`, `tool_result`, `usage` snapshots) with no
    # message/turn boundary marker of its own. This module treats ONE ATIF
    # step as everything between two `usage` events: `text` deltas accumulate
    # into that step's `message`, every `tool_use` in the window becomes a
    # ToolCall, and the usage event that CLOSES the window supplies that
    # step's Metrics. This mirrors how loop.ts itself is structured — one
    # `onUsage` call fires per completed model turn (see
    # packages/core/src/agent/loop.ts's usage-event handling) — so a step
    # boundary here corresponds to one real LLM inference, exactly what ATIF's
    # `llm_call_count` semantics expect.
    text_buf: list[str] = []
    reasoning_buf: list[str] = []
    tool_calls: list[ToolCall] = []
    # FIFO per tool name — see module docstring's KNOWN LIMITATION.
    pending_by_name: dict[str, deque[str]] = defaultdict(deque)
    results_by_call_id: dict[str, ObservationResult] = {}
    call_counter = 0
    had_activity = False

    def flush_step(metrics: Metrics | None) -> None:
        nonlocal text_buf, reasoning_buf, tool_calls
        nonlocal pending_by_name, results_by_call_id, had_activity
        if not had_activity:
            # An empty usage window (e.g. two usage events back-to-back with
            # nothing between them, which can happen on a retried/aborted
            # attempt) produces no step — an ATIF Step requires a message,
            # and an empty one would misrepresent an inference that produced
            # nothing observable.
            return
        message = "".join(text_buf)
        reasoning = "".join(reasoning_buf) or None
        observation = (
            Observation(results=list(results_by_call_id.values())) if results_by_call_id else None
        )
        steps.append(
            Step(
                step_id=len(steps) + 1,
                source="agent",
                model_name=model_name,
                message=message,
                reasoning_content=reasoning,
                tool_calls=list(tool_calls) or None,
                observation=observation,
                metrics=metrics,
                llm_call_count=1,
            )
        )
        text_buf = []
        reasoning_buf = []
        tool_calls = []
        pending_by_name = defaultdict(deque)
        results_by_call_id = {}
        had_activity = False

    for event in events:
        etype = event.get("type")

        if etype == "text":
            text = event.get("text")
            if isinstance(text, str) and text:
                text_buf.append(text)
                had_activity = True

        elif etype == "thinking":
            text = event.get("text")
            if isinstance(text, str) and text:
                reasoning_buf.append(text)
                had_activity = True

        elif etype == "tool_use":
            tool_name = event.get("tool")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            call_counter += 1
            call_id = f"call_{call_counter}"
            tool_calls.append(
                ToolCall(
                    tool_call_id=call_id,
                    function_name=tool_name,
                    arguments=event.get("input") if isinstance(event.get("input"), dict) else {},
                )
            )
            pending_by_name[tool_name].append(call_id)
            had_activity = True

        elif etype == "tool_result":
            tool_name = event.get("tool")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            queue = pending_by_name.get(tool_name)
            call_id = queue.popleft() if queue else None
            if call_id is None:
                # An orphan result (no matching tool_use in this window —
                # e.g. the tool_use event was in a PRIOR step because a
                # partial usage snapshot split the window early). Still
                # record it, unattached, so the output is not silently
                # dropped from the trajectory.
                call_id = None
            ok = event.get("ok")
            error = event.get("error")
            output = event.get("output")
            if ok is False and isinstance(error, str):
                content = error
            elif ok:
                content = "ok"
            elif output is not None:
                content = json.dumps(output)
            else:
                content = ""
            result_extra = {"ok": ok} if ok is not None else None
            results_by_call_id[call_id or f"orphan_{len(results_by_call_id)}"] = ObservationResult(
                source_call_id=call_id,
                content=content,
                extra=result_extra,
            )
            had_activity = True

        elif etype == "usage":
            usage = event.get("usage")
            metrics = None
            has_tokens = isinstance(usage, dict) and any(
                usage.get(k) for k in ("input_tokens", "output_tokens")
            )
            if has_tokens:
                metrics = Metrics(
                    prompt_tokens=(usage.get("input_tokens") or 0)
                    + (usage.get("cache_creation_input_tokens") or 0)
                    + (usage.get("cache_read_input_tokens") or 0),
                    completion_tokens=usage.get("output_tokens") or 0,
                    cached_tokens=usage.get("cache_read_input_tokens") or 0,
                )
            flush_step(metrics)

        # Other event types (`start`, `notice`, `retry`, `tool_stream`,
        # `stream_restart`, `balance_status`, `result`, `error`,
        # `context_warning`) carry no ATIF-representable content of their own
        # — they're either one-time run metadata (`start`) or duplicates of
        # information already captured elsewhere (`result` restates the final
        # `text` already accumulated; `error` is reflected in the trial's own
        # exception_info, not the trajectory).

    # A trailing window with no closing usage event (the process was killed
    # mid-turn, e.g. AgentTimeoutError) still deserves a step so its partial
    # work is not silently discarded from the trajectory.
    flush_step(None)

    if len(steps) <= (1 if instruction.strip() else 0):
        # Only the instruction step (or nothing at all) — no real agent
        # activity was captured, so there is nothing worth calling a
        # trajectory. Matches claude_code.py's "no valid steps" bailout.
        return None

    prompt_values = [
        s.metrics.prompt_tokens for s in steps if s.metrics and s.metrics.prompt_tokens is not None
    ]
    completion_values = [
        s.metrics.completion_tokens
        for s in steps
        if s.metrics and s.metrics.completion_tokens is not None
    ]
    cached_values = [
        s.metrics.cached_tokens for s in steps if s.metrics and s.metrics.cached_tokens is not None
    ]

    total_prompt_tokens = sum(prompt_values) if prompt_values else None
    total_completion_tokens = sum(completion_values) if completion_values else None
    total_cached_tokens = sum(cached_values) if cached_values else None

    total_cost_usd = None
    final_extra: dict[str, Any] = {}
    if total_prompt_tokens is not None or total_completion_tokens is not None:
        total_cost_usd = _estimate_cost_usd(
            model_name, total_prompt_tokens or 0, total_completion_tokens or 0
        )
        if total_cost_usd is not None:
            final_extra["cost_source"] = "vendor_price_table"

    final_metrics = FinalMetrics(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_cached_tokens=total_cached_tokens,
        total_cost_usd=total_cost_usd,
        total_steps=len(steps),
        extra=final_extra or None,
    )

    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=Agent(
            name="nex",
            version=agent_version,
            model_name=model_name,
        ),
        steps=steps,
        final_metrics=final_metrics,
    )
