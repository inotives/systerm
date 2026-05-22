from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from systerm.daemon import TOKEN_ENV


DEFAULT_DAEMON_URL = "http://127.0.0.1:8765"
DAEMON_URL_ENV = "SYSTERM_DAEMON_URL"


class DaemonClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class DaemonClientConfig:
    base_url: str
    token: str

    @property
    def websocket_url(self) -> str:
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url.removeprefix("https://").rstrip("/") + "/events"
        return "ws://" + self.base_url.removeprefix("http://").rstrip("/") + "/events"


def load_daemon_client_config(
    base_url: str | None = None,
    token_path: Path | None = None,
) -> DaemonClientConfig:
    url = (base_url or os.getenv(DAEMON_URL_ENV) or DEFAULT_DAEMON_URL).rstrip("/")
    token = os.getenv(TOKEN_ENV)
    if token is None:
        path = token_path or Path.home() / ".config" / "systerm" / "token"
        if not path.exists():
            raise DaemonClientError(
                f"daemon token not found at {path}; start `systerm daemon` or set {TOKEN_ENV}"
            )
        token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise DaemonClientError("daemon token is empty")
    return DaemonClientConfig(base_url=url, token=token)


class DaemonClient:
    def __init__(self, config: DaemonClientConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.transport = transport

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.token}"}

    async def health(self) -> dict[str, Any]:
        return await self._get("/health", auth=False)

    async def snapshot(self) -> dict[str, Any]:
        return {
            "sessions": await self._get("/sessions"),
            "jobs": await self._get("/jobs"),
            "approvals": await self._get("/approvals"),
            "events": await self._get("/events"),
            "models": await self._get("/models"),
            "providers": await self._get("/providers"),
            "agent": await self._get("/agents/current"),
            "tools": await self._get("/tools"),
        }

    async def jobs(self) -> list[dict[str, Any]]:
        return await self._get("/jobs")

    async def job(self, job_id: int) -> dict[str, Any]:
        return await self._get(f"/jobs/{job_id}")

    async def create_job(self, prompt: str) -> dict[str, Any]:
        return await self.create_job_for_session(prompt)

    async def create_job_for_session(self, prompt: str, session_id: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": prompt}
        if session_id is not None:
            payload["session_id"] = session_id
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            response = await client.post(
                self.config.base_url + "/jobs",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def cancel_job(self, job_id: int) -> dict[str, Any]:
        return await self._post(f"/jobs/{job_id}/cancel")

    async def retry_job(self, job_id: int) -> dict[str, Any]:
        return await self._post(f"/jobs/{job_id}/retry")

    async def sessions(self) -> list[dict[str, Any]]:
        return await self._get("/sessions")

    async def create_session(self) -> dict[str, Any]:
        return await self._post("/sessions")

    async def session(self, session_id: int) -> dict[str, Any]:
        return await self._get(f"/sessions/{session_id}")

    async def session_trace(self, session_id: int) -> dict[str, Any]:
        return await self._get(f"/sessions/{session_id}/trace")

    async def approvals(self, status: str = "pending") -> list[dict[str, Any]]:
        return await self._get(f"/approvals?status={status}")

    async def approve(self, approval_id: int) -> dict[str, Any]:
        return await self._post(f"/approvals/{approval_id}/approve")

    async def reject(self, approval_id: int) -> dict[str, Any]:
        return await self._post(f"/approvals/{approval_id}/reject")

    async def _get(self, path: str, auth: bool = True) -> Any:
        async with httpx.AsyncClient(timeout=10, transport=self.transport) as client:
            response = await client.get(
                self.config.base_url + path,
                headers=self.headers if auth else None,
            )
            response.raise_for_status()
            return response.json()

    async def _post(self, path: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10, transport=self.transport) as client:
            response = await client.post(self.config.base_url + path, headers=self.headers)
            response.raise_for_status()
            return response.json()
