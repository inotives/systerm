from systerm.tui import (
    latest_completed_session_id,
    render_approvals,
    render_events,
    render_jobs,
    render_runtime,
    render_sessions,
    render_transcript,
)


def test_tui_render_helpers_show_empty_state() -> None:
    assert render_jobs([]) == "Jobs\n  none"
    assert render_sessions([]) == "Sessions\n  none"
    assert render_approvals([]) == "Approvals\n  none"
    assert render_events([]) == "Events\n  none"


def test_tui_render_runtime() -> None:
    rendered = render_runtime(
        {
            "models": {"default_model": "fast", "fallback_models": ["slow"]},
            "agent": {"name": "systerm"},
            "providers": {"groq": {}},
            "tools": {"shell": {}},
        }
    )

    assert "agent: systerm" in rendered
    assert "model: fast" in rendered
    assert "providers: groq" in rendered
    assert "tools: shell" in rendered


def test_tui_render_rows() -> None:
    assert "#1 queued" in render_jobs([{"id": 1, "status": "queued", "session_id": None}])
    assert "#2 3 messages" in render_sessions([{"id": 2, "message_count": 3}])
    assert "#3 pending medium shell" in render_approvals(
        [{"id": 3, "status": "pending", "risk": "medium", "tool_name": "shell"}]
    )
    assert "#4 job.created" in render_events([{"id": 4, "type": "job.created", "job_id": 1}])


def test_tui_render_transcript() -> None:
    rendered = render_transcript(
        {
            "id": 7,
            "messages": [
                {"role": "user", "content": "hello", "model_profile": None},
                {"role": "assistant", "content": "hi", "model_profile": "groq"},
            ],
        }
    )

    assert "Transcript session #7" in rendered
    assert "user:" in rendered
    assert "assistant [groq]:" in rendered
    assert "hi" in rendered


def test_latest_completed_session_id() -> None:
    assert latest_completed_session_id(
        [
            {"id": 2, "status": "queued", "session_id": None},
            {"id": 1, "status": "complete", "session_id": 9},
        ]
    ) == 9
    assert latest_completed_session_id([{"id": 2, "status": "failed", "session_id": None}]) is None
