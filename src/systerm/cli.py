from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import httpx
import typer
import uvicorn

from systerm.agents import AgentProfileError, load_agent_profile, missing_references
from systerm.agent_loop import AgentLoop
from systerm.config import ConfigError, load_config, validate_model_profile
from systerm.daemon_client import DaemonClient, DaemonClientError, load_daemon_client_config
from systerm.daemon import create_app
from systerm.providers import OpenAICompatibleClient, ProviderError
from systerm.storage import SessionStore, default_db_path
from systerm.tools import ToolRegistryError, ToolRunner, load_tool_registry
from systerm.tui import run_tui


app = typer.Typer(invoke_without_command=True)
approvals_app = typer.Typer(help="Manage pending tool approvals.")
jobs_app = typer.Typer(help="Manage daemon jobs.")
sessions_app = typer.Typer(help="Inspect local chat sessions.")
tools_app = typer.Typer(help="Run local tools.")
app.add_typer(approvals_app, name="approvals")
app.add_typer(jobs_app, name="jobs")
app.add_typer(sessions_app, name="sessions")
app.add_typer(tools_app, name="tools")


@app.callback()
def main(
    ctx: typer.Context,
    auto_daemon: Annotated[
        bool,
        typer.Option("--auto-daemon/--no-auto-daemon", help="Start a local daemon automatically for the TUI."),
    ] = True,
) -> None:
    """Launch the TUI by default, or run a subcommand."""
    if ctx.invoked_subcommand is None:
        run_tui(auto_start_daemon=auto_daemon)


@app.command()
def tui(
    auto_daemon: Annotated[
        bool,
        typer.Option("--auto-daemon/--no-auto-daemon", help="Start a local daemon automatically for the TUI."),
    ] = True,
) -> None:
    """Open the TUI operator console."""
    run_tui(auto_start_daemon=auto_daemon)


@app.command()
def doctor() -> None:
    """Validate local Systerm configuration."""
    project_root = Path.cwd()
    try:
        config = load_config(project_root)
        profile = load_agent_profile(project_root / "AGENTS.md")
        validate_model_profile(config, profile.model)
        missing = missing_references(profile, project_root)
        load_tool_registry(project_root, profile.tools)
    except (ConfigError, AgentProfileError, ToolRegistryError) as exc:
        typer.secho(f"doctor failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    if missing:
        typer.secho(f"doctor failed: missing references: {', '.join(missing)}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"config: ok ({len(config.model_profiles)} model profile)")
    typer.echo(f"agent: ok ({profile.name}, model={profile.model})")
    typer.echo(f"database: {default_db_path(project_root)}")


@app.command()
def daemon(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the localhost Systerm daemon."""
    project_root = Path.cwd()
    uvicorn.run(create_app(project_root), host=host, port=port)


@app.command()
def chat(
    prompt: str,
    daemon_mode: Annotated[bool, typer.Option("--daemon", help="Submit through the daemon job queue.")] = False,
) -> None:
    """Run a one-off chat session or submit a daemon job."""
    asyncio.run(_chat(prompt, daemon_mode))


async def _chat(prompt: str, daemon_mode: bool = False) -> None:
    if daemon_mode:
        client = _load_daemon_client()
        try:
            job = await client.create_job(prompt)
        except (httpx.HTTPError, DaemonClientError) as exc:
            typer.secho(f"chat failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        typer.echo(f"queued job {job['id']}\t{job['status']}")
        return

    project_root = Path.cwd()
    try:
        config = load_config(project_root)
        profile = load_agent_profile(project_root / "AGENTS.md")
        validate_model_profile(config, profile.model)
        tools = load_tool_registry(project_root, profile.tools)
    except (ConfigError, AgentProfileError, ToolRegistryError) as exc:
        typer.secho(f"chat failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    store = SessionStore(default_db_path(project_root))
    await store.init()
    client = OpenAICompatibleClient(config)
    try:
        result = await AgentLoop(client, store, tools).run(prompt, requested_model=profile.model)
    except ProviderError as exc:
        typer.secho(f"chat failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.echo(result.content)


@jobs_app.command("list")
def jobs_list() -> None:
    """List daemon jobs."""
    asyncio.run(_jobs_list())


async def _jobs_list() -> None:
    client = _load_daemon_client()
    try:
        jobs = await client.jobs()
    except (httpx.HTTPError, DaemonClientError) as exc:
        typer.secho(f"jobs failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    _print_job_rows(jobs)


@jobs_app.command("show")
def jobs_show(job_id: int) -> None:
    """Show one daemon job."""
    asyncio.run(_jobs_show(job_id))


async def _jobs_show(job_id: int) -> None:
    client = _load_daemon_client()
    try:
        job = await client.job(job_id)
    except (httpx.HTTPError, DaemonClientError) as exc:
        typer.secho(f"job failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    _print_job_rows([job])


@jobs_app.command("cancel")
def jobs_cancel(job_id: int) -> None:
    """Cancel a queued, running, or approval-required daemon job."""
    asyncio.run(_jobs_cancel(job_id))


async def _jobs_cancel(job_id: int) -> None:
    client = _load_daemon_client()
    try:
        job = await client.cancel_job(job_id)
    except (httpx.HTTPError, DaemonClientError) as exc:
        typer.secho(f"cancel failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.echo(f"{job['id']}\t{job['status']}")


@jobs_app.command("retry")
def jobs_retry(job_id: int) -> None:
    """Retry a daemon job with the same prompt."""
    asyncio.run(_jobs_retry(job_id))


async def _jobs_retry(job_id: int) -> None:
    client = _load_daemon_client()
    try:
        job = await client.retry_job(job_id)
    except (httpx.HTTPError, DaemonClientError) as exc:
        typer.secho(f"retry failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.echo(f"queued job {job['id']}\t{job['status']}")


@sessions_app.command("list")
def sessions_list(
    daemon_mode: Annotated[bool, typer.Option("--daemon", help="Read sessions from the daemon.")] = False,
) -> None:
    """List local chat sessions."""
    asyncio.run(_sessions_list(daemon_mode))


async def _sessions_list(daemon_mode: bool = False) -> None:
    if daemon_mode:
        client = _load_daemon_client()
        try:
            sessions = await client.sessions()
        except (httpx.HTTPError, DaemonClientError) as exc:
            typer.secho(f"sessions failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        _print_session_rows(sessions)
        return

    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    sessions = await store.list_sessions()
    _print_session_rows(sessions)


@sessions_app.command("show")
def sessions_show(
    session_id: int,
    daemon_mode: Annotated[bool, typer.Option("--daemon", help="Read session messages from the daemon.")] = False,
) -> None:
    """Show messages in a local chat session."""
    asyncio.run(_sessions_show(session_id, daemon_mode))


async def _sessions_show(session_id: int, daemon_mode: bool = False) -> None:
    if daemon_mode:
        client = _load_daemon_client()
        try:
            session = await client.session(session_id)
        except (httpx.HTTPError, DaemonClientError) as exc:
            typer.secho(f"session failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        _print_message_rows(session.get("messages", []))
        return

    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    messages = await store.list_message_records(session_id)
    _print_message_rows([message.__dict__ for message in messages])


@sessions_app.command("trace")
def sessions_trace(
    session_id: int,
    daemon_mode: Annotated[bool, typer.Option("--daemon", help="Read session trace from the daemon.")] = False,
) -> None:
    """Show messages, tool calls, tool results, and approvals for a session."""
    asyncio.run(_sessions_trace(session_id, daemon_mode))


async def _sessions_trace(session_id: int, daemon_mode: bool = False) -> None:
    if daemon_mode:
        client = _load_daemon_client()
        try:
            trace = await client.session_trace(session_id)
        except (httpx.HTTPError, DaemonClientError) as exc:
            typer.secho(f"trace failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        _print_trace(trace.get("messages", []), trace.get("tool_calls", []), trace.get("approvals", []))
        return

    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    messages = await store.list_message_records(session_id)
    tool_calls = await store.list_tool_calls(session_id)
    approvals = await store.list_session_approvals(session_id)
    tool_call_rows = []
    for tool_call in tool_calls:
        tool_call_rows.append({**tool_call.__dict__, "results": [result.__dict__ for result in await store.list_tool_results(tool_call.id)]})
    _print_trace(
        [message.__dict__ for message in messages],
        tool_call_rows,
        [approval.__dict__ for approval in approvals],
    )


def _print_trace(
    messages: list[dict[str, object]],
    tool_calls: list[dict[str, object]],
    approvals: list[dict[str, object]],
) -> None:
    typer.echo("messages")
    if not messages:
        typer.echo("  none")
    for message in messages:
        typer.echo(
            f"  {message['id']}\t{message['role']}\t{message.get('model_profile') or ''}\t"
            f"{message.get('metadata_json', '{}')}\t{message['content']}"
        )

    typer.echo("tool_calls")
    if not tool_calls:
        typer.echo("  none")
    for tool_call in tool_calls:
        typer.echo(
            f"  {tool_call['id']}\t{tool_call['tool_name']}\t{tool_call['risk']}\t"
            f"approval={tool_call.get('approval_id') or ''}\t"
            f"{tool_call.get('metadata_json', '{}')}\t{tool_call['arguments_json']}"
        )
        for result in tool_call.get("results", []):
            typer.echo(
                f"    result {result['id']}\t{result['status']}\t"
                f"{result.get('metadata_json', '{}')}\t{result['content']}"
            )

    typer.echo("approvals")
    if not approvals:
        typer.echo("  none")
    for approval in approvals:
        typer.echo(
            f"  {approval['id']}\t{approval['status']}\t{approval['risk']}\t"
            f"{approval.get('metadata_json', '{}')}\t{approval['arguments_json']}"
        )


@approvals_app.command("list")
def approvals_list(
    status: str = "pending",
    daemon_mode: Annotated[bool, typer.Option("--daemon", help="Read approvals from the daemon.")] = False,
) -> None:
    """List tool approvals."""
    asyncio.run(_approvals_list(status, daemon_mode))


async def _approvals_list(status: str, daemon_mode: bool = False) -> None:
    if daemon_mode:
        client = _load_daemon_client()
        try:
            approvals = await client.approvals(status=status)
        except (httpx.HTTPError, DaemonClientError) as exc:
            typer.secho(f"approvals failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        _print_approval_rows(approvals)
        return

    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    approvals = await store.list_approvals(status=status if status != "all" else None)
    _print_approval_rows([approval.__dict__ for approval in approvals])


@approvals_app.command("approve")
def approvals_approve(
    approval_id: int,
    daemon_mode: Annotated[bool, typer.Option("--daemon", help="Approve through the daemon.")] = False,
) -> None:
    """Approve a pending tool request."""
    asyncio.run(_resolve_approval(approval_id, "approved", daemon_mode))


@approvals_app.command("reject")
def approvals_reject(
    approval_id: int,
    daemon_mode: Annotated[bool, typer.Option("--daemon", help="Reject through the daemon.")] = False,
) -> None:
    """Reject a pending tool request."""
    asyncio.run(_resolve_approval(approval_id, "rejected", daemon_mode))


async def _resolve_approval(approval_id: int, status: str, daemon_mode: bool = False) -> None:
    if daemon_mode:
        client = _load_daemon_client()
        try:
            approval = await client.approve(approval_id) if status == "approved" else await client.reject(approval_id)
        except (httpx.HTTPError, DaemonClientError) as exc:
            typer.secho(f"approval failed: {exc}", fg=typer.colors.RED)
            raise typer.Exit(1) from exc
        typer.echo(f"{approval['id']}\t{approval['status']}\t{approval['tool_name']}")
        return

    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    try:
        approval = await store.resolve_approval(approval_id, status)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.echo(f"{approval.id}\t{approval.status}\t{approval.tool_name}")


def _print_session_rows(sessions: list[dict[str, object]]) -> None:
    if not sessions:
        typer.echo("no sessions")
        return
    for session in sessions:
        typer.echo(
            f"{session['id']}\t{session['created_at']}\t"
            f"{session['message_count']} messages\t{session.get('metadata_json', '{}')}"
        )


def _print_job_rows(jobs: list[dict[str, object]]) -> None:
    if not jobs:
        typer.echo("no jobs")
        return
    for job in jobs:
        typer.echo(
            f"{job['id']}\t{job['status']}\tsession={job.get('session_id') or ''}\t"
            f"{job.get('metadata_json', '{}')}\t{job['prompt']}"
        )


def _print_message_rows(messages: list[dict[str, object]]) -> None:
    if not messages:
        typer.echo("no messages")
        return
    for message in messages:
        model = f"\t{message['model_profile']}" if message.get("model_profile") else ""
        typer.echo(
            f"{message['id']}\t{message['role']}{model}\t"
            f"{message.get('metadata_json', '{}')}\t{message['content']}"
        )


def _print_approval_rows(approvals: list[dict[str, object]]) -> None:
    if not approvals:
        typer.echo("no approvals")
        return

    for approval in approvals:
        typer.echo(
            f"{approval['id']}\t{approval['status']}\t{approval['risk']}\t"
            f"{approval['tool_name']}\t{approval['arguments_json']}"
        )


def _load_daemon_client() -> DaemonClient:
    return DaemonClient(load_daemon_client_config())


@tools_app.command("shell")
def tools_shell(command: str) -> None:
    """Run a shell command or create an approval request."""
    asyncio.run(_tools_shell(command))


async def _tools_shell(command: str) -> None:
    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    result = await ToolRunner(store).run_shell(command)
    if result.status == "approval-required":
        typer.echo(f"{result.content} (approval_id={result.approval_id})")
        raise typer.Exit(2)
    if result.status == "error":
        typer.secho(result.content, fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(result.content, nl=not result.content.endswith("\n"))
