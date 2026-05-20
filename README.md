# Systerm

Systerm is a local-first agent harness. The first implementation slice provides a Python CLI, TOML config loading, `AGENTS.md` profile validation, OpenAI-compatible provider calls, and SQLite-backed session persistence.

## Development

```bash
uv sync
uv run pytest
uv run systerm doctor
uv run systerm daemon
uv run systerm
uv run systerm chat "hello"
uv run systerm sessions list
uv run systerm sessions show 1
uv run systerm sessions trace 1
uv run systerm tools shell "echo hello"
uv run systerm approvals list
```

Configuration is loaded from `~/.config/systerm/model.toml`, `~/.config/systerm/config.toml`, `.systerm/model.toml`, then `.systerm/config.toml`. Later files override earlier files. Secrets are loaded from the environment, with local `.env` support for development.

The default project config uses NVIDIA NIM first and Groq as fallback:

```toml
default_model = "nvidia-minimax-2.7"
fallback_models = ["groq-llama-3.3-70b"]
```

Set `NVIDIA_API_KEY` and `GROQ_API_KEY` in `.env`.

`.systerm/model.toml` contains the provider/model catalog. `.systerm/config.toml` selects the active default and fallback chain. `model_profiles` choose from provider model lists and add runtime settings like timeout, retry count, context window, max tokens, and temperature.

## Tool Approvals

The first tool slice includes an approval-aware shell runner. Low-risk commands such as `echo`, `ls`, `pwd`, `whoami`, and `date` run immediately. Medium/high-risk commands create pending approval records in local SQLite:

```bash
uv run systerm tools shell "python script.py"
uv run systerm approvals list
uv run systerm approvals approve 1
uv run systerm approvals reject 1
```

## Daemon

The Phase 3 daemon starts a localhost API:

```bash
uv run systerm daemon
```

The daemon authenticates local clients with a bearer token from `~/.config/systerm/token`, or `SYSTERM_DAEMON_TOKEN` when set. Current endpoints include health, sessions, session traces, async jobs, approvals, and persisted events.

```bash
curl http://127.0.0.1:8765/health
curl -H "Authorization: Bearer $SYSTERM_DAEMON_TOKEN" http://127.0.0.1:8765/events
curl -H "Authorization: Bearer $SYSTERM_DAEMON_TOKEN" \
  -X POST http://127.0.0.1:8765/jobs \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}'
```

## TUI

`systerm` launches the Textual operator console by default. For the first TUI slice, start the daemon separately, then open the TUI:

```bash
uv run systerm daemon
uv run systerm
```

The TUI loads daemon snapshots, streams daemon events, shows jobs/sessions/approvals/runtime state, and submits prompts to the daemon job queue.
