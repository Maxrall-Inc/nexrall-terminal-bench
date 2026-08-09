"""
Harbor agent adapter for the Nexrall Code (`nex`) CLI.

Implements Harbor's `BaseInstalledAgent` interface so Terminal-Bench (which
runs on Harbor) can drive `nex` inside its per-task Docker container exactly
the way it drives Claude Code, Codex CLI, etc.

IMPORTANT — this targets the harbor==0.20.0 API surface actually installed
from PyPI, verified live against the real package (`pip show harbor`,
`python3 -c "from harbor... import ..."`), NOT the harbor-framework/harbor
GitHub `main` branch. The two differ meaningfully: this installed version has
no `ensure_system_dependencies`/`PackageSpec` helper (system packages are
installed by hand with `apt-get` below) and no `ModelConnectionSpec`/
`MODEL_CONNECTION` (there is no `model_connection` property at all on this
version's `BaseAgent`/`BaseInstalledAgent` — auth is read directly from
`NEXRALL_TOKEN` via `self._get_env` instead). An earlier draft of this file
was written against `main` and failed at real runtime with
`AttributeError: 'NexAgent' object has no attribute
'ensure_system_dependencies'` (see git history) — this version was corrected
against a real `harbor run --install-only` trial that installs `nex` inside
an actual task container, not just import-checked.
"""

import json
import shlex
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, EnvVar, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from nex_terminal_bench.atif import convert_stream_json_to_trajectory

# Legacy tier names some scripts/habits still pass, resolved to a real model id
# before anything reaches the CLI's --model flag. Mirrors
# packages/bench/swebench/run_inference.py's LEGACY_MODEL_ALIASES so the two
# harnesses behave identically for the same input.
LEGACY_MODEL_ALIASES = {
    "turbo": "claude-sonnet-5",
    "pro": "claude-opus-5",
    "ultra": "claude-fable-5",
}

_OUTPUT_FILENAME = "nex-output.jsonl"


class NexAgent(BaseInstalledAgent):
    """
    Harbor agent adapter for Nexrall Code.

    Runs `nex` in headless one-shot mode (instruction piped over stdin, since
    a Terminal-Bench task instruction can be arbitrarily long — `nex
    --output-format stream-json --yolo --no-banner`) inside the Harbor task
    container, so every tool call Terminal-Bench's verifier can observe (file
    edits, bash commands) actually happens in the container the verifier
    will grade.

    `--yolo` (auto-approve) is required because there is no TTY inside the
    sandbox to answer a permission prompt — the CLI's headless mode already
    denies destructive commands (DB drops, force-push, etc.) unless
    NEXRALL_ALLOW_DESTRUCTIVE=1 is set, which this adapter never sets, so a
    Terminal-Bench task cannot use nex to do anything irreversible to its own
    grading harness even under blanket auto-approval.
    """

    # A real ATIF converter (nex_terminal_bench/atif.py's
    # convert_stream_json_to_trajectory) is wired into
    # populate_context_post_run below. nex's stream-json event log does not
    # carry the per-tool-call ids the ATIF schema's ToolCall.tool_call_id
    # normally expects (confirmed against packages/core/src/agent/loop.ts's
    # onToolUse/onToolResult callbacks, which pass name+input but no id) —
    # atif.py works around this with a FIFO-per-tool-name pairing strategy
    # (see that module's KNOWN LIMITATION docstring for exactly when that
    # could mis-pair two same-named calls in one turn). This is a real,
    # schema-valid ATIF-v1.7 trajectory, not a placeholder — the leaderboard's
    # own static analysis only requires trajectory_path to exist and parse as
    # ATIF, not that every field be maximally precise.
    SUPPORTS_ATIF: bool = True

    # Class-level default so populate_context_post_run's `self._last_instruction
    # or ""` fallback works even if it's somehow called before run() ever set
    # the real instruction (e.g. setup() itself raised) — instance assignment
    # in run() shadows this per-object, same pattern as any mutable default
    # avoided at the class level (this one is an immutable str, so sharing the
    # class attribute across instances is safe).
    _last_instruction: str = ""

    # nex has no CLI flag for turn/iteration limits — it's controlled by the
    # NEXRALL_MAX_ITERATIONS env var (see packages/core/src/agent/loop.ts's
    # resolveMaxIterations / DEFAULT_MAX_ITERATIONS = 500). Declared as an
    # EnvVar (a CliFlag would require a real --flag, which doesn't exist) so
    # `harbor run --ak max_iterations=300` still works to tune a task's turn
    # budget without hardcoding it here.
    ENV_VARS = [
        EnvVar(
            "max_iterations",
            env="NEXRALL_MAX_ITERATIONS",
            type="int",
        ),
    ]

    # `-e/--effort` is a REAL nex CLI flag (packages/cli/src/index.ts),
    # unlike max_iterations above. Terminal-Bench's own leaderboard
    # convention is `--ak reasoning_effort=<level>` (see
    # harbor/agents/installed/claude_code.py's own CLI_FLAGS entry for the
    # same kwarg name, used there for Claude Code's `--effort`) — mirrored
    # here so a caller can raise effort for a harder task the same way
    # regardless of which installed agent is running. Left unset by default:
    # every trial in this harness ran at nex's ordinary interactive default
    # (`medium`, chat.ts's own fallback) until this flag existed, so leaving
    # it unset preserves that behavior for anyone who doesn't pass --ak.
    CLI_FLAGS = [
        CliFlag(
            "reasoning_effort",
            cli="--effort",
            type="enum",
            choices=["low", "medium", "high", "extra"],
            env_fallback="NEXRALL_REASONING_EFFORT",
        ),
    ]

    @staticmethod
    def name() -> str:
        return "nex"

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; nex --version'

    def parse_version(self, stdout: str) -> str:
        return stdout.strip()

    async def _installed_nex_satisfies_version(self, environment: BaseEnvironment) -> bool:
        check = await environment.exec(
            command='export PATH="$HOME/.local/bin:$PATH"; command -v nex >/dev/null 2>&1'
        )
        if check.return_code != 0:
            return False
        if self._version is None:
            return True
        version_result = await environment.exec(command=self.get_version_command())
        if version_result.return_code != 0:
            return False
        return self.parse_version(version_result.stdout or "") == self._version

    async def install(self, environment: BaseEnvironment) -> None:
        if await self._installed_nex_satisfies_version(environment):
            self.logger.debug("nex is already available at the requested version")
            return

        # This installed harbor version has no ensure_system_dependencies()
        # helper (see module docstring) — install what nex needs by hand.
        # Most Terminal-Bench task images are Debian/Ubuntu-based; fall back
        # silently if apt-get isn't the package manager (e.g. an alpine task
        # image), matching how other Harbor agents degrade gracefully rather
        # than hard-failing install() on an unusual base image.
        #
        # Deliberately NOT installing nodejs/npm from apt/apk here: a real
        # `harbor run --install-only` trial against terminal-bench-2-1 hit a
        # live task image on Node 18.19.1 (Ubuntu 22.04's apt default), and
        # nex's bundled `ink` dependency uses a regex `v` flag that Node < 20
        # doesn't support — nex crashed at startup with `SyntaxError: Invalid
        # flags supplied to RegExp constructor 'v'`. Installing Node via nvm
        # instead (same approach as harbor-framework/harbor's own Codex and
        # Gemini CLI adapters) guarantees a modern enough Node regardless of
        # what the task image ships.
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get >/dev/null 2>&1; then "
                "  DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                "  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
                "    curl ca-certificates git ripgrep >/dev/null; "
                "elif command -v apk >/dev/null 2>&1; then "
                "  apk add --no-cache curl ca-certificates git ripgrep; "
                "fi"
            ),
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/"
                "v0.40.2/install.sh | bash && "
                'export NVM_DIR="$HOME/.nvm" && '
                '\\. "$NVM_DIR/nvm.sh" && '
                "nvm install 22 && nvm alias default 22 && "
                f"npm install -g nexrall-code{version_spec} && "
                "for bin in node nex; do "
                '  BIN_PATH="$(command -v "$bin" 2>/dev/null || true)"; '
                '  if [ -n "$BIN_PATH" ]; then '
                '    ln -sf "$BIN_PATH" "$HOME/.local/bin/$bin" 2>/dev/null || true; '
                '  fi; '
                "done; "
                'mkdir -p "$HOME/.local/bin" && '
                'ln -sf "$(command -v node)" "$HOME/.local/bin/node" && '
                'ln -sf "$(command -v nex)" "$HOME/.local/bin/nex" && '
                "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> ~/.bashrc && "
                'export PATH="$HOME/.local/bin:$PATH" && '
                "nex --version"
            ),
            env={"NVM_NODEJS_ORG_MIRROR": "https://nodejs.org/dist"},
        )

    def _resolve_model(self) -> str | None:
        if not self.model_name:
            return None
        # Harbor model ids are "provider/model" (e.g. anthropic/claude-sonnet-5);
        # nex's --model flag takes the bare model id.
        bare = self.model_name.split("/", 1)[-1] if "/" in self.model_name else self.model_name
        return LEGACY_MODEL_ALIASES.get(bare, bare)

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        # Stashed for populate_context_post_run's ATIF conversion — that
        # method only gets `context`, not the original instruction, and the
        # trajectory's first step needs the actual task prompt that was sent,
        # not the prompt-template-wrapped version `run()` receives here (see
        # @with_prompt_template above, which already substitutes the
        # instruction before this line — this IS the real prompt text).
        self._last_instruction = instruction

        token = self._get_env("NEXRALL_TOKEN")
        if not token:
            raise RuntimeError(
                "NEXRALL_TOKEN is not set. Terminal-Bench needs a headless Nexrall "
                "auth token to run `nex` non-interactively — get one from your "
                "Nexrall account and export it before calling `harbor run`."
            )

        model = self._resolve_model()
        model_flag = f"--model {shlex.quote(model)} " if model else ""

        # Resolved from CLI_FLAGS above (e.g. --ak reasoning_effort=high). Empty
        # string when unset — `nex`'s own default (medium) applies, unchanged
        # from before this flag existed.
        cli_flags = self.build_cli_flags()
        effort_flag = f"{cli_flags} " if cli_flags else ""

        env = {
            "NEXRALL_TOKEN": token,
            # Resolved from ENV_VARS above (e.g. --ak max_iterations=300).
            # resolve_env_vars() already coerced/validated the value and
            # omits the key entirely when unset, so this only overrides the
            # CLI's own default (DEFAULT_MAX_ITERATIONS = 500 in loop.ts)
            # when a caller explicitly asked for a different budget.
            **self.resolve_env_vars(),
            # ── Mid-stream stall tolerance ────────────────────────────────
            # client.ts's default PROGRESS_TIMEOUT_MS (150s) is tuned for
            # interactive use, where failing fast + auto-retrying is the
            # better UX. Terminal-Bench tasks are long, tool-call-heavy
            # agentic sessions — exactly the shape that triggers Anthropic's
            # own well-documented mid-stream stalls (the model holds
            # generated tokens without flushing for anywhere from ~60s up to
            # several minutes; see anthropics/claude-code#18028, #20610,
            # #54434 — all confirmed API-side, not client/proxy bugs). Claude
            # Code tolerates this with a byte-level watchdog whose minimum is
            # 5 minutes (CLAUDE_STREAM_IDLE_TIMEOUT_MS, floor 300000ms) —
            # nearly double nex's interactive default. Raise nex's threshold
            # to match that floor here so a benchmark run doesn't tear down
            # and retry a turn that Claude Code would have simply waited out,
            # which was inflating nex's apparent failure rate relative to
            # Claude Code on the same tasks for a reason that had nothing to
            # do with either agent's actual capability.
            "NEXRALL_PROGRESS_TIMEOUT_MS": "300000",
            # No TTY in the sandbox — headless mode auto-approves every tool call
            # via --yolo below; this only gates the extra class of DESTRUCTIVE
            # commands (DB drops, force-push, terraform destroy) that headless
            # mode denies outright unless explicitly re-enabled. Deliberately
            # left unset: a Terminal-Bench task's verifier and instructions never
            # legitimately require one, and disabling this gate would let a
            # prompt-injected or confused agent damage the grading container.
            # "NEXRALL_ALLOW_DESTRUCTIVE": "1",  # intentionally never set
            #
            # Terminal-Bench tasks are Harbor-vetted, version-pinned task
            # containers (not an arbitrary clone of unreviewed content), so
            # this harness vouches for them the same way run_inference.py does
            # for a SWE-bench instance's pinned base_commit.
            "NEXRALL_TRUST_WORKSPACE": "1",
            # Terminal-Bench's leaderboard rules forbid the agent looking up
            # the benchmark's own website/repo (reward hacking via solution
            # lookup) — block it outright, mirroring run_inference.py's
            # GitHub blocklist for SWE-bench.
            "NEXRALL_FETCH_BLOCKLIST": (
                "tbench.ai,www.tbench.ai,"
                "github.com,raw.githubusercontent.com,githubusercontent.com,"
                "githubassets.com,api.github.com,gist.github.com,codeload.github.com,"
                "huggingface.co,hub.harborframework.com"
            ),
        }

        output_path = (EnvironmentPaths.agent_dir / _OUTPUT_FILENAME).as_posix()
        await self.exec_as_agent(
            environment, command=f"mkdir -p {EnvironmentPaths.agent_dir.as_posix()}"
        )

        instruction_shell_var = "nex_terminal_bench_instruction"
        instruction_env_var = instruction_shell_var.upper()
        run_env = {**env, instruction_env_var: instruction}

        await self.exec_as_agent(
            environment,
            command=(
                'export PATH="$HOME/.local/bin:$PATH"; '
                f'{instruction_shell_var}="${instruction_env_var}"; '
                f"unset {instruction_env_var}; "
                f'printf "%s" "${instruction_shell_var}" | '
                "nex --output-format stream-json --yolo --no-banner "
                f"{model_flag}"
                f"{effort_flag}"
                f"2>&1 | tee {output_path}"
            ),
            env=run_env,
        )

    # ─── Post-run: turn nex's stream-json log into usage + a debug summary ────

    def _read_events(self) -> list[dict[str, Any]]:
        output_file = self.logs_dir / _OUTPUT_FILENAME
        if not output_file.exists():
            self.logger.debug(f"nex output file not found: {output_file}")
            return []
        events: list[dict[str, Any]] = []
        with open(output_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events

    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._read_events()
        if not events:
            return

        last_result_text = ""
        for event in events:
            if event.get("type") == "result":
                last_result_text = event.get("result") or last_result_text

        # Real ATIF conversion (see atif.py's module docstring for the full
        # design rationale). Mirrors harbor's own claude_code.py
        # populate_context_post_run: convert first, then derive context's
        # token/cost fields FROM the trajectory's own FinalMetrics rather than
        # summing raw events twice in two different places that could drift
        # out of sync with each other.
        try:
            trajectory = convert_stream_json_to_trajectory(
                self.logs_dir / _OUTPUT_FILENAME,
                agent_version=self.version() or "unknown",
                default_model_name=self._parsed_model_name,
                instruction=self._last_instruction or "",
            )
        except Exception as exc:
            self.logger.debug(f"Failed to convert nex stream-json to ATIF trajectory: {exc}")
            trajectory = None

        if trajectory is None:
            self.logger.debug("No ATIF trajectory produced (empty or unparseable log)")
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            with open(trajectory_path, "w", encoding="utf-8") as handle:
                json.dump(trajectory.to_json_dict(), handle, indent=2, ensure_ascii=False)
            self.logger.debug(f"Wrote nex ATIF trajectory to {trajectory_path}")
        except OSError as exc:
            self.logger.debug(f"Failed to write trajectory file {trajectory_path}: {exc}")

        if trajectory.final_metrics:
            metrics = trajectory.final_metrics
            context.cost_usd = metrics.total_cost_usd
            context.n_input_tokens = metrics.total_prompt_tokens or 0
            context.n_cache_tokens = metrics.total_cached_tokens or 0
            context.n_output_tokens = metrics.total_completion_tokens or 0
