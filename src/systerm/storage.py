from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite


@dataclass(frozen=True)
class SessionRecord:
    id: int
    created_at: str
    metadata_json: str = "{}"


@dataclass(frozen=True)
class MessageRecord:
    id: int
    session_id: int
    role: str
    content: str
    model_profile: str | None
    created_at: str
    metadata_json: str


@dataclass(frozen=True)
class ApprovalRecord:
    id: int
    tool_name: str
    arguments_json: str
    risk: str
    status: str
    reason: str
    created_at: str
    resolved_at: str | None
    metadata_json: str


@dataclass(frozen=True)
class ToolCallRecord:
    id: int
    session_id: int | None
    tool_name: str
    arguments_json: str
    risk: str
    approval_id: int | None
    created_at: str
    metadata_json: str


@dataclass(frozen=True)
class ToolResultRecord:
    id: int
    tool_call_id: int
    status: str
    content: str
    created_at: str
    metadata_json: str


@dataclass(frozen=True)
class JobRecord:
    id: int
    prompt: str
    status: str
    session_id: int | None
    result_content: str | None
    error: str | None
    created_at: str
    completed_at: str | None
    metadata_json: str


@dataclass(frozen=True)
class EventRecord:
    id: int
    type: str
    job_id: int | None
    session_id: int | None
    payload_json: str
    created_at: str
    metadata_json: str


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                create table if not exists sessions (
                    id integer primary key autoincrement,
                    created_at text not null,
                    metadata_json text not null default '{}'
                )
                """
            )
            await db.execute(
                """
                create table if not exists messages (
                    id integer primary key autoincrement,
                    session_id integer not null references sessions(id),
                    role text not null,
                    content text not null,
                    model_profile text,
                    created_at text not null,
                    metadata_json text not null default '{}'
                )
                """
            )
            await db.execute(
                """
                create table if not exists approvals (
                    id integer primary key autoincrement,
                    tool_name text not null,
                    arguments_json text not null,
                    risk text not null,
                    status text not null,
                    reason text not null,
                    created_at text not null,
                    resolved_at text,
                    metadata_json text not null default '{}'
                )
                """
            )
            await db.execute(
                """
                create table if not exists tool_calls (
                    id integer primary key autoincrement,
                    session_id integer references sessions(id),
                    tool_name text not null,
                    arguments_json text not null,
                    risk text not null,
                    approval_id integer references approvals(id),
                    created_at text not null,
                    metadata_json text not null default '{}'
                )
                """
            )
            await db.execute(
                """
                create table if not exists tool_results (
                    id integer primary key autoincrement,
                    tool_call_id integer references tool_calls(id),
                    status text not null,
                    content text not null,
                    created_at text not null,
                    metadata_json text not null default '{}'
                )
                """
            )
            await db.execute(
                """
                create table if not exists jobs (
                    id integer primary key autoincrement,
                    prompt text not null,
                    status text not null,
                    session_id integer references sessions(id),
                    result_content text,
                    error text,
                    created_at text not null,
                    completed_at text,
                    metadata_json text not null default '{}'
                )
                """
            )
            await db.execute(
                """
                create table if not exists events (
                    id integer primary key autoincrement,
                    type text not null,
                    job_id integer references jobs(id),
                    session_id integer references sessions(id),
                    payload_json text not null,
                    created_at text not null,
                    metadata_json text not null default '{}'
                )
                """
            )
            await _ensure_column(db, "sessions", "metadata_json", "text not null default '{}'")
            await _ensure_column(db, "messages", "metadata_json", "text not null default '{}'")
            await _ensure_column(db, "approvals", "metadata_json", "text not null default '{}'")
            await _ensure_column(db, "tool_calls", "metadata_json", "text not null default '{}'")
            await _ensure_column(db, "tool_results", "metadata_json", "text not null default '{}'")
            await _ensure_column(db, "jobs", "metadata_json", "text not null default '{}'")
            await _ensure_column(db, "events", "metadata_json", "text not null default '{}'")
            await db.commit()

    async def create_session(self, metadata_json: str = "{}") -> SessionRecord:
        created_at = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "insert into sessions (created_at, metadata_json) values (?, ?)",
                (created_at, metadata_json),
            )
            await db.commit()
            return SessionRecord(id=_require_row_id(cursor.lastrowid), created_at=created_at, metadata_json=metadata_json)

    async def list_sessions(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select sessions.id, sessions.created_at, sessions.metadata_json, count(messages.id) as message_count
                from sessions
                left join messages on messages.session_id = sessions.id
                group by sessions.id
                order by sessions.id desc
                """
            )
            rows = await cursor.fetchall()
        return [{"id": row[0], "created_at": row[1], "metadata_json": row[2], "message_count": row[3]} for row in rows]

    async def create_job(self, prompt: str, metadata_json: str = "{}") -> JobRecord:
        created_at = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                insert into jobs (prompt, status, created_at, metadata_json)
                values (?, ?, ?, ?)
                """,
                (prompt, "queued", created_at, metadata_json),
            )
            await db.commit()
            job_id = _require_row_id(cursor.lastrowid)
        return JobRecord(
            id=job_id,
            prompt=prompt,
            status="queued",
            session_id=None,
            result_content=None,
            error=None,
            created_at=created_at,
            completed_at=None,
            metadata_json=metadata_json,
        )

    async def complete_job(
        self,
        job_id: int,
        status: str,
        session_id: int | None,
        result_content: str | None = None,
        error: str | None = None,
    ) -> JobRecord:
        completed_at = _now()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                update jobs
                set status = ?, session_id = ?, result_content = ?, error = ?, completed_at = ?
                where id = ?
                """,
                (status, session_id, result_content, error, completed_at, job_id),
            )
            await db.commit()
            cursor = await db.execute(
                """
                select id, prompt, status, session_id, result_content, error, created_at, completed_at, metadata_json
                from jobs
                where id = ?
                """,
                (job_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            raise ValueError(f"job {job_id} does not exist")
        return _job_from_row(row)

    async def get_job(self, job_id: int) -> JobRecord | None:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select id, prompt, status, session_id, result_content, error, created_at, completed_at, metadata_json
                from jobs
                where id = ?
                """,
                (job_id,),
            )
            row = await cursor.fetchone()
        return _job_from_row(row) if row is not None else None

    async def list_jobs(self) -> list[JobRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select id, prompt, status, session_id, result_content, error, created_at, completed_at, metadata_json
                from jobs
                order by id desc
                """
            )
            rows = await cursor.fetchall()
        return [_job_from_row(row) for row in rows]

    async def create_event(
        self,
        event_type: str,
        payload_json: str,
        job_id: int | None = None,
        session_id: int | None = None,
        metadata_json: str = "{}",
    ) -> EventRecord:
        created_at = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                insert into events (type, job_id, session_id, payload_json, created_at, metadata_json)
                values (?, ?, ?, ?, ?, ?)
                """,
                (event_type, job_id, session_id, payload_json, created_at, metadata_json),
            )
            await db.commit()
            event_id = _require_row_id(cursor.lastrowid)
        return EventRecord(
            id=event_id,
            type=event_type,
            job_id=job_id,
            session_id=session_id,
            payload_json=payload_json,
            created_at=created_at,
            metadata_json=metadata_json,
        )

    async def list_events(self, after_id: int = 0, limit: int = 100) -> list[EventRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select id, type, job_id, session_id, payload_json, created_at, metadata_json
                from events
                where id > ?
                order by id
                limit ?
                """,
                (after_id, limit),
            )
            rows = await cursor.fetchall()
        return [_event_from_row(row) for row in rows]

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        model_profile: str | None = None,
        metadata_json: str = "{}",
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                insert into messages (session_id, role, content, model_profile, created_at, metadata_json)
                values (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, model_profile, _now(), metadata_json),
            )
            await db.commit()

    async def list_messages(self, session_id: int) -> list[dict[str, str]]:
        records = await self.list_message_records(session_id)
        return [{"role": message.role, "content": message.content} for message in records]

    async def list_message_records(self, session_id: int) -> list[MessageRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select id, session_id, role, content, model_profile, created_at, metadata_json
                from messages
                where session_id = ?
                order by id
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [_message_from_row(row) for row in rows]

    async def create_approval(
        self,
        tool_name: str,
        arguments_json: str,
        risk: str,
        reason: str,
        metadata_json: str = "{}",
    ) -> ApprovalRecord:
        created_at = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                insert into approvals (tool_name, arguments_json, risk, status, reason, created_at, metadata_json)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (tool_name, arguments_json, risk, "pending", reason, created_at, metadata_json),
            )
            await db.commit()
            approval_id = _require_row_id(cursor.lastrowid)
        return ApprovalRecord(
            id=approval_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            risk=risk,
            status="pending",
            reason=reason,
            created_at=created_at,
            resolved_at=None,
            metadata_json=metadata_json,
        )

    async def create_tool_call(
        self,
        tool_name: str,
        arguments_json: str,
        risk: str,
        session_id: int | None = None,
        approval_id: int | None = None,
        metadata_json: str = "{}",
    ) -> ToolCallRecord:
        created_at = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                insert into tool_calls (session_id, tool_name, arguments_json, risk, approval_id, created_at, metadata_json)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, tool_name, arguments_json, risk, approval_id, created_at, metadata_json),
            )
            await db.commit()
            tool_call_id = _require_row_id(cursor.lastrowid)
        return ToolCallRecord(
            id=tool_call_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments_json=arguments_json,
            risk=risk,
            approval_id=approval_id,
            created_at=created_at,
            metadata_json=metadata_json,
        )

    async def add_tool_result(
        self,
        tool_call_id: int,
        status: str,
        content: str,
        metadata_json: str = "{}",
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                insert into tool_results (tool_call_id, status, content, created_at, metadata_json)
                values (?, ?, ?, ?, ?)
                """,
                (tool_call_id, status, content, _now(), metadata_json),
            )
            await db.commit()

    async def list_tool_calls(self, session_id: int) -> list[ToolCallRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select id, session_id, tool_name, arguments_json, risk, approval_id, created_at, metadata_json
                from tool_calls
                where session_id = ?
                order by id
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [_tool_call_from_row(row) for row in rows]

    async def list_tool_results(self, tool_call_id: int) -> list[ToolResultRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select id, tool_call_id, status, content, created_at, metadata_json
                from tool_results
                where tool_call_id = ?
                order by id
                """,
                (tool_call_id,),
            )
            rows = await cursor.fetchall()
        return [_tool_result_from_row(row) for row in rows]

    async def list_session_approvals(self, session_id: int) -> list[ApprovalRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                select approvals.id, approvals.tool_name, approvals.arguments_json, approvals.risk,
                       approvals.status, approvals.reason, approvals.created_at, approvals.resolved_at,
                       approvals.metadata_json
                from approvals
                join tool_calls on tool_calls.approval_id = approvals.id
                where tool_calls.session_id = ?
                order by approvals.id
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [_approval_from_row(row) for row in rows]

    async def list_approvals(self, status: str | None = None) -> list[ApprovalRecord]:
        query = """
            select id, tool_name, arguments_json, risk, status, reason, created_at, resolved_at, metadata_json
            from approvals
        """
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " where status = ?"
            params = (status,)
        query += " order by id"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        return [_approval_from_row(row) for row in rows]

    async def resolve_approval(self, approval_id: int, status: str) -> ApprovalRecord:
        if status not in {"approved", "rejected"}:
            raise ValueError("approval status must be approved or rejected")

        resolved_at = _now()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "update approvals set status = ?, resolved_at = ? where id = ? and status = 'pending'",
                (status, resolved_at, approval_id),
            )
            await db.commit()
            if cursor.rowcount == 0:
                exists = await db.execute("select status from approvals where id = ?", (approval_id,))
                row = await exists.fetchone()
                if row is None:
                    raise ValueError(f"approval {approval_id} does not exist")
                raise ValueError(f"approval {approval_id} is already {row[0]}")
            cursor = await db.execute(
                """
                select id, tool_name, arguments_json, risk, status, reason, created_at, resolved_at, metadata_json
                from approvals
                where id = ?
                """,
                (approval_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            raise ValueError(f"approval {approval_id} does not exist")
        return _approval_from_row(row)


def default_db_path(project_root: Path) -> Path:
    return project_root / ".systerm" / "systerm.db"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _approval_from_row(row: tuple[Any, ...]) -> ApprovalRecord:
    return ApprovalRecord(
        id=row[0],
        tool_name=row[1],
        arguments_json=row[2],
        risk=row[3],
        status=row[4],
        reason=row[5],
        created_at=row[6],
        resolved_at=row[7],
        metadata_json=row[8],
    )


def _message_from_row(row: tuple[Any, ...]) -> MessageRecord:
    return MessageRecord(
        id=row[0],
        session_id=row[1],
        role=row[2],
        content=row[3],
        model_profile=row[4],
        created_at=row[5],
        metadata_json=row[6],
    )


def _tool_call_from_row(row: tuple[Any, ...]) -> ToolCallRecord:
    return ToolCallRecord(
        id=row[0],
        session_id=row[1],
        tool_name=row[2],
        arguments_json=row[3],
        risk=row[4],
        approval_id=row[5],
        created_at=row[6],
        metadata_json=row[7],
    )


def _tool_result_from_row(row: tuple[Any, ...]) -> ToolResultRecord:
    return ToolResultRecord(
        id=row[0],
        tool_call_id=row[1],
        status=row[2],
        content=row[3],
        created_at=row[4],
        metadata_json=row[5],
    )


def _job_from_row(row: tuple[Any, ...]) -> JobRecord:
    return JobRecord(
        id=row[0],
        prompt=row[1],
        status=row[2],
        session_id=row[3],
        result_content=row[4],
        error=row[5],
        created_at=row[6],
        completed_at=row[7],
        metadata_json=row[8],
    )


def _event_from_row(row: tuple[Any, ...]) -> EventRecord:
    return EventRecord(
        id=row[0],
        type=row[1],
        job_id=row[2],
        session_id=row[3],
        payload_json=row[4],
        created_at=row[5],
        metadata_json=row[6],
    )


async def _ensure_column(db: aiosqlite.Connection, table_name: str, column_name: str, definition: str) -> None:
    cursor = await db.execute(f"pragma table_info({table_name})")
    columns = {row[1] for row in await cursor.fetchall()}
    if column_name not in columns:
        await db.execute(f"alter table {table_name} add column {column_name} {definition}")


def _require_row_id(row_id: int | None) -> int:
    if row_id is None:
        raise RuntimeError("database did not return a row id")
    return row_id
