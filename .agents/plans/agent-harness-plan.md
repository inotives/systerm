# Systerm Agent Harness Plan

## Summary

Build a greenfield Python 3.12 + `uv` project in `/home/inotives/workspaces/systerm`: a general-purpose agent harness that starts with a local CLI core and grows into an always-on daemon with a Textual TUI operator console, OpenAI-compatible model providers, SQLite-backed sessions/jobs, Discord command ingress, simple scheduled tasks, policy-based approvals for risky tools, and a single primary-agent plus sub-agent execution model.

The design borrows the small-core ideas from Pi: model registry, streaming agent loop, session persistence, slash/operator commands, config files, and later extensibility.

References:

- https://github.com/earendil-works/pi
- https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/index.md
- https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md
- https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/usage.md

## Key Interfaces

- CLI package exposes one command, `systerm`:
  - `systerm daemon`: starts the local daemon.
  - `systerm tui`: opens the operator console and attaches to the daemon.
  - `systerm chat "prompt"`: runs a one-off local session in Phase 1, then submits a daemon job once the daemon exists.
  - `systerm approvals list/approve/reject`: manages pending tool approvals from the CLI.
  - `systerm schedule add/list/remove`: manages simple scheduled agent prompts.
  - `systerm doctor`: validates config, DB, providers, profile references, and phase-specific connector env vars.
- Config uses TOML:
  - Global: `~/.config/systerm/config.toml`
  - Project override: `.systerm/config.toml`
  - Secrets are loaded from local environment variables, with repo-local `.env` support for development.
  - Commit `.env.template` with required credential names and ignore `.env`.
  - Defines the default model profile and ordered fallback model profiles.
- Provider config supports OpenAI-compatible APIs first:
  - OpenAI, Groq, NVIDIA NIM, Ollama, LM Studio, vLLM, and compatible proxies via `base_url`, `api_key_env`, `model`, `api_style`, `supports_streaming`, and `supports_tools`.
  - Each configured model profile points to a provider, model name, timeout, retry count, and compatibility settings.
- Daemon exposes localhost-only API:
  - Uses token authentication for local HTTP clients.
  - Stores generated local token at `~/.config/systerm/token` with `0600` permissions.
  - Allows token override through an environment variable for CI/dev.
  - HTTP endpoints for sessions, jobs, approvals, schedules, providers.
  - WebSocket event stream for TUI live updates.
- SQLite stores:
  - sessions, messages, model events, tool calls, tool results, approvals, jobs, schedules, connector events, and audit logs.
- Agent profile uses `AGENTS.md`:
  - Stores the primary agent personality, operating profile, available skills, and available tools.
  - Uses readable Markdown for personality/profile guidance plus a required fenced TOML block for machine-readable skill/tool indexes.
  - Required TOML fields are `agent.name`, `agent.model`, `skills`, and `tools`.
  - Skills and tools listed in `AGENTS.md` are indexes/references, not full implementations.
  - Referenced skills live under `.agents/skills/<skill>/SKILL.md`.
  - Referenced tools live under `.agents/tools/<tool>/tool.toml` or tool-specific config files, including MCP servers, CLI tools, web tools, and future email/social tools.

## Phased Implementation

### Phase 1: Core Local Harness

- Scaffold Python project with `pyproject.toml`, `uv.lock`, `src/systerm/`, and tests.
- Set up local virtual environment workflow:
  - Create `.venv` with `uv venv --python 3.12`.
  - Install dependencies with `uv sync`.
  - Add `.venv/` and `.env` to `.gitignore`.
  - Document `uv run ...` commands in `README.md`.
- Add credential template:
  - Commit `.env.template` with provider and connector credential variable names.
  - Load `.env` in local CLI/daemon processes before reading config.
  - Keep actual credentials out of git.
- Use the minimum core libraries:
  - `typer` for CLI.
  - `pydantic` for config/domain models.
  - `python-dotenv` for local `.env` loading.
  - `httpx` and an OpenAI-compatible client wrapper for model calls.
  - `aiosqlite` over SQLite for persistence.
- Implement local CLI commands:
  - `systerm chat "prompt"` for one-off local runs.
  - `systerm doctor` for config, provider, profile, and DB validation.
- Implement provider registry for OpenAI-compatible APIs:
  - Support `base_url`, `api_key_env`, `model`, and provider name.
  - Support initial compatibility fields: `api_style = "openai_chat"`, `supports_streaming`, and `supports_tools`.
  - Support `default_model` and ordered `fallback_models` from config.
  - Fail over to the next fallback model when the current model times out, returns a retriable provider error, or produces no assistant response.
  - Persist which model profile was attempted and which model produced the final response.
  - Confirm compatibility with OpenAI, Groq, NVIDIA NIM, Ollama, LM Studio, and vLLM-style endpoints.
- Implement `AGENTS.md` profile loading:
  - Parse project-local `AGENTS.md`.
  - Extract a required fenced TOML block for deterministic skill/tool/profile references.
  - Validate minimum TOML schema: `agent.name`, `agent.model`, `skills`, and `tools`.
  - Load primary agent personality/profile.
  - Resolve skill indexes to skill folders.
  - Resolve tool indexes to tool definitions/config.
  - Report missing skill/tool references through `systerm doctor`.
  - In Phase 1, validate tool and skill references but do not execute tools or activate sub-agent behavior yet.
- Implement the primary agent loop:
  - Single primary agent owns the task and final response.
  - Streaming assistant messages.
  - Session/message persistence in SQLite.
  - No daemon, TUI, Discord, scheduler, sub-agents, or risky tool execution yet.

### Phase 2: Tools, Approvals, and Sub-Agents

- Add the structured tool registry and tool-call loop:
  - Tool definitions expose name, description, JSON-schema-style parameters, and risk level.
  - Tool results append back into the session context.
  - Stop reasons: complete, tool-use, error, cancelled, approval-required.
- Add core tools:
  - Shell/script execution with risk classification.
  - Web fetch by URL.
  - Web search interface with a stub/provider boundary for later search backends.
- Add safety model:
  - Tools declare `low`, `medium`, or `high` risk.
  - Low-risk tools can auto-run.
  - Medium/high-risk tools create approval records.
  - Before the daemon exists, CLI approvals operate on local SQLite approval records created during `systerm chat`.
  - CLI can approve/reject pending approvals before TUI exists.
- Add sub-agent execution:
  - Primary agent can spawn bounded sub-agents for complex tasks through an internal `spawn_agent` tool.
  - Sub-agents receive scoped instructions, selected skills/tools, and task context.
  - Sub-agent outputs return to the primary agent as structured results.
  - Sub-agents are internal helpers, not independent long-running personas.
  - Sub-agents inherit the primary agent model by default unless a scoped override is configured.
  - Sub-agents use the same fallback chain as their selected model profile.
  - Sub-agents cannot use tools broader than the tool scope granted by the primary agent.
  - Limit v1 to three concurrent sub-agents per primary task.
  - Only the primary agent can speak to the user or external connectors by default.
  - Persist every sub-agent task, granted scope, final result, and error as child events of the parent session.
  - Cancel child sub-agents when the parent task is cancelled.
  - Treat sub-agent spawning as a medium-risk internal tool when it grants any non-read-only tools.

### Phase 3: Daemon and TUI Operator Console

- Add daemon process:
  - `systerm daemon` starts a localhost-only service.
  - Local HTTP clients authenticate with a generated/shared token.
  - HTTP endpoints for sessions, jobs, approvals, providers, and profile reload.
  - WebSocket event stream for live TUI updates.
  - Async job queue for submitted prompts.
- Add TUI:
  - `systerm tui` opens an operator console attached to the daemon.
  - Panels for sessions, active jobs, pending approvals, live event log, current model, and tool traces.
  - Local prompt input and model selector.
  - Approve/reject medium/high-risk tool calls from the TUI.

### Phase 4: Schedules and Discord Connector

- Add simple scheduled tasks:
  - `systerm schedule add/list/remove`.
  - Named prompts with cron/interval/manual trigger.
  - Each schedule declares the tools it is allowed to use.
  - Tool approval for scheduled tasks is settled when the schedule is created or updated.
  - Store a snapshot/hash of each approved tool definition with the schedule.
  - Require re-approval before the schedule can run if an approved tool definition changes.
  - At runtime, scheduled tasks cannot use tools outside their declared allowlist.
  - Jobs execute through the daemon and persist in SQLite.
- Add Discord ingress:
  - `discord.py` connector runs in the daemon.
  - Allowlisted user IDs only.
  - Support DM and configured channel IDs.
  - Discord messages create daemon jobs.
  - Summarized job status/results stream back to Discord.
  - Discord approvals only accepted from allowlisted user IDs.

### Later Addons

- MCP client support as a tool provider.
- Email tools and approval policies.
- Slack connector.
- Richer web search/fetch providers.
- Global default profiles in addition to project-local `AGENTS.md`.
- TUI-managed configuration screens.

## Test Plan

- Phase 1 tests:
  - Local `.venv`/`uv run` setup is documented and `.venv/` is ignored by git.
  - `.env.template` exists, `.env` is ignored, and local `.env` values are loaded.
  - TOML config merge and env secret resolution.
  - `AGENTS.md` fenced TOML extraction and invalid-block diagnostics.
  - `AGENTS.md` minimum TOML schema validation.
  - `AGENTS.md` parsing and missing skill/tool reference validation.
  - Provider registry and OpenAI-compatible request mapping.
  - Default model selection and fallback model failover.
  - No-response, timeout, and retriable provider error handling.
  - SQLite repositories for sessions and messages.
  - Fake OpenAI-compatible streaming server emits streaming text.
- Phase 2 tests:
  - Tool risk policy and approval state transitions.
  - Primary-agent and sub-agent task handoff boundaries.
  - Primary agent delegates a complex task to a sub-agent and incorporates the returned result.
  - Sub-agent cancellation follows parent cancellation.
  - Sub-agents cannot use tools outside the scope granted by the primary agent.
  - Agent loop executes approved low-risk tools and pauses on high-risk tools.
  - Fake OpenAI-compatible streaming server emits tool calls.
- Phase 3 tests:
  - Daemon rejects HTTP/WebSocket clients without a valid local token.
  - Local daemon token is generated with restrictive file permissions and env override works.
  - Daemon API creates jobs and emits WebSocket events.
  - TUI can connect to daemon, display sessions/jobs, and approve a pending tool call.
- Phase 4 tests:
  - SQLite repositories for jobs, schedules, and connector events.
  - Scheduled tasks cannot use tools outside their declared allowlist.
  - Scheduled tasks require re-approval when an approved tool definition changes.
  - Scheduler creates jobs at expected times.
  - Discord connector tested with mocked Discord client/events.
- Manual acceptance by phase:
  - Phase 1: `uv run systerm doctor` passes and `uv run systerm chat "hello"` returns a model response.
  - Phase 2: A shell command requiring approval pauses until approved, then returns a tool result.
  - Phase 3: `uv run systerm daemon` and `uv run systerm tui` run together and show live job/session updates.
  - Phase 4: A scheduled prompt runs through the daemon using its declared approved tools, and a Discord allowlisted user can submit a prompt.

## Assumptions

- Project name/CLI is `systerm`.
- V1 is a general agent harness, not primarily a coding agent.
- V1 connector is Discord first; Slack is deferred.
- V1 persistence is SQLite event log.
- V1 persistence uses `aiosqlite`.
- V1 config is TOML plus `.env`/environment variables for secrets.
- V1 config includes a default model profile and ordered fallback model profiles.
- V1 uses a repo-local `.venv` created by `uv`; dependencies are not installed globally.
- `.env.template` is committed as the credential contract; `.env` is local-only and ignored by git.
- V1 tool bundle is core only; MCP/email/Slack are planned extension targets, not first implementation.
- The daemon is required for scheduled/background tasks; the TUI is an attachable operator client.
- `AGENTS.md` is the canonical project-local agent profile and registry index.
- `AGENTS.md` uses Markdown for human-readable instructions plus a required fenced TOML block for machine-readable references.
- Skills and tools are implemented outside `AGENTS.md`; the file only references them and defines how the primary agent may use them.
- Skill directories default to `.agents/skills/<skill>/SKILL.md`.
- Tool definitions default to `.agents/tools/<tool>/tool.toml`.
- Sub-agents are internal execution helpers owned by the primary agent, not independent long-running personas in v1.
- Sub-agents inherit model/provider defaults from the primary agent unless explicitly scoped otherwise.
- Sub-agents cannot communicate directly with users or external connectors in v1.
- Daemon API uses localhost HTTP with token authentication.
- Daemon local token defaults to `~/.config/systerm/token` with `0600` permissions and can be overridden by environment variable.
- Scheduled tasks declare their allowed tools, and approval is resolved when the schedule is created or updated.
- Scheduled task approval is invalidated when an approved tool definition changes.
