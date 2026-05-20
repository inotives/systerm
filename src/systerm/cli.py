from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import uvicorn

from systerm.agents import AgentProfileError, load_agent_profile, missing_references
from systerm.agent_loop import AgentLoop
from systerm.config import ConfigError, load_config, validate_model_profile
from systerm.daemon import create_app
from systerm.providers import OpenAICompatibleClient, ProviderError
from systerm.storage import SessionStore, default_db_path
from systerm.tools import ToolRegistryError, ToolRunner, load_tool_registry
from systerm.tui import run_tui


app = typer.Typer(invoke_without_command=True)
approvals_app = typer.Typer(help="Manage pending tool approvals.")
sessions_app = typer.Typer(help="Inspect local chat sessions.")
tools_app = typer.Typer(help="Run local tools.")
app.add_typer(approvals_app, name="approvals")
app.add_typer(sessions_app, name="sessions")
app.add_typer(tools_app, name="tools")


@app.callback()
def main(ctx: typer.Context) -> None:
    """Launch the TUI by default, or run a subcommand."""
    if ctx.invoked_subcommand is None:
        run_tui()


@app.command()
def tui() -> None:
    """Open the TUI operator console."""
    run_tui()


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
def chat(prompt: str) -> None:
    """Run a one-off local chat session."""
    asyncio.run(_chat(prompt))


async def _chat(prompt: str) -> None:
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


@sessions_app.command("list")
def sessions_list() -> None:
    """List local chat sessions."""
    asyncio.run(_sessions_list())


async def _sessions_list() -> None:
    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    sessions = await store.list_sessions()
    if not sessions:
        typer.echo("no sessions")
        return
    for session in sessions:
        typer.echo(
            f"{session['id']}\t{session['created_at']}\t"
            f"{session['message_count']} messages\t{session['metadata_json']}"
        )


@sessions_app.command("show")
def sessions_show(session_id: int) -> None:
    """Show messages in a local chat session."""
    asyncio.run(_sessions_show(session_id))


async def _sessions_show(session_id: int) -> None:
    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    messages = await store.list_message_records(session_id)
    if not messages:
        typer.echo("no messages")
        return
    for message in messages:
        model = f"\t{message.model_profile}" if message.model_profile else ""
        typer.echo(f"{message.id}\t{message.role}{model}\t{message.metadata_json}\t{message.content}")


@sessions_app.command("trace")
def sessions_trace(session_id: int) -> None:
    """Show messages, tool calls, tool results, and approvals for a session."""
    asyncio.run(_sessions_trace(session_id))


async def _sessions_trace(session_id: int) -> None:
    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    messages = await store.list_message_records(session_id)
    tool_calls = await store.list_tool_calls(session_id)
    approvals = await store.list_session_approvals(session_id)

    typer.echo("messages")
    if not messages:
        typer.echo("  none")
    for message in messages:
        typer.echo(f"  {message.id}\t{message.role}\t{message.model_profile or ''}\t{message.metadata_json}\t{message.content}")

    typer.echo("tool_calls")
    if not tool_calls:
        typer.echo("  none")
    for tool_call in tool_calls:
        typer.echo(
            f"  {tool_call.id}\t{tool_call.tool_name}\t{tool_call.risk}\t"
            f"approval={tool_call.approval_id or ''}\t{tool_call.metadata_json}\t{tool_call.arguments_json}"
        )
        for result in await store.list_tool_results(tool_call.id):
            typer.echo(f"    result {result.id}\t{result.status}\t{result.metadata_json}\t{result.content}")

    typer.echo("approvals")
    if not approvals:
        typer.echo("  none")
    for approval in approvals:
        typer.echo(
            f"  {approval.id}\t{approval.status}\t{approval.risk}\t"
            f"{approval.metadata_json}\t{approval.arguments_json}"
        )


@approvals_app.command("list")
def approvals_list(status: str = "pending") -> None:
    """List tool approvals."""
    asyncio.run(_approvals_list(status))


async def _approvals_list(status: str) -> None:
    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    approvals = await store.list_approvals(status=status if status != "all" else None)
    if not approvals:
        typer.echo("no approvals")
        return

    for approval in approvals:
        typer.echo(
            f"{approval.id}\t{approval.status}\t{approval.risk}\t"
            f"{approval.tool_name}\t{approval.arguments_json}"
        )


@approvals_app.command("approve")
def approvals_approve(approval_id: int) -> None:
    """Approve a pending tool request."""
    asyncio.run(_resolve_approval(approval_id, "approved"))


@approvals_app.command("reject")
def approvals_reject(approval_id: int) -> None:
    """Reject a pending tool request."""
    asyncio.run(_resolve_approval(approval_id, "rejected"))


async def _resolve_approval(approval_id: int, status: str) -> None:
    store = SessionStore(default_db_path(Path.cwd()))
    await store.init()
    try:
        approval = await store.resolve_approval(approval_id, status)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.echo(f"{approval.id}\t{approval.status}\t{approval.tool_name}")


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
