from pathlib import Path

import pytest

from systerm.storage import SessionStore


@pytest.mark.asyncio
async def test_session_store_persists_messages(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()
    session = await store.create_session(metadata_json='{"source": "test"}')

    await store.add_message(session.id, "user", "hello", metadata_json='{"turn": 1}')
    await store.add_message(session.id, "assistant", "hi", "local")

    assert await store.list_messages(session.id) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    sessions = await store.list_sessions()
    assert sessions == [
        {"id": session.id, "created_at": session.created_at, "metadata_json": '{"source": "test"}', "message_count": 2}
    ]
    messages = await store.list_message_records(session.id)
    assert messages[0].metadata_json == '{"turn": 1}'
    assert messages[1].metadata_json == "{}"


@pytest.mark.asyncio
async def test_session_store_resolves_approvals(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()

    approval = await store.create_approval(
        "shell",
        '{"command": "python script.py"}',
        "medium",
        "needs approval",
        metadata_json='{"source": "test"}',
    )
    resolved = await store.resolve_approval(approval.id, "approved")

    assert resolved.status == "approved"
    assert resolved.resolved_at is not None
    assert resolved.metadata_json == '{"source": "test"}'
    assert await store.list_approvals(status="pending") == []


@pytest.mark.asyncio
async def test_session_store_rejects_resolving_approval_twice(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()

    approval = await store.create_approval("shell", '{"command": "python script.py"}', "medium", "needs approval")
    await store.resolve_approval(approval.id, "rejected")

    with pytest.raises(ValueError):
        await store.resolve_approval(approval.id, "approved")


@pytest.mark.asyncio
async def test_session_store_persists_tool_calls_and_results(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()
    session = await store.create_session()

    tool_call = await store.create_tool_call(
        "shell",
        '{"command": "echo hi"}',
        "low",
        session_id=session.id,
        metadata_json='{"policy": "auto"}',
    )
    await store.add_tool_result(tool_call.id, "complete", "hi\n", metadata_json='{"returncode": 0}')

    assert tool_call.id > 0
    assert tool_call.session_id == session.id
    assert tool_call.metadata_json == '{"policy": "auto"}'
    assert await store.list_tool_calls(session.id) == [tool_call]
    results = await store.list_tool_results(tool_call.id)
    assert len(results) == 1
    assert results[0].metadata_json == '{"returncode": 0}'


@pytest.mark.asyncio
async def test_session_store_persists_jobs(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()
    session = await store.create_session()
    job = await store.create_job("hello")

    completed = await store.complete_job(job.id, "complete", session.id, result_content="hi")

    assert completed.status == "complete"
    assert completed.session_id == session.id
    assert completed.result_content == "hi"
    assert await store.list_jobs() == [completed]
    assert await store.get_job(job.id) == completed


@pytest.mark.asyncio
async def test_session_store_persists_events(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()
    job = await store.create_job("hello")

    first = await store.create_event("job.created", '{"job_id": 1}', job_id=job.id)
    second = await store.create_event("job.started", '{"job_id": 1}', job_id=job.id)

    assert await store.list_events() == [first, second]
    assert await store.list_events(after_id=first.id) == [second]
