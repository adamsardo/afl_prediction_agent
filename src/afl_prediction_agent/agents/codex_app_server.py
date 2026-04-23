from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

from afl_prediction_agent.core.settings import get_settings


SUPPORTED_CHATGPT_PLAN_TYPES = {
    "free",
    "go",
    "plus",
    "pro",
    "business",
    "enterprise",
}


class CodexAppServerError(RuntimeError):
    """Base error for Codex app-server integration."""


class CodexAppServerProcessError(CodexAppServerError):
    """Raised when the local app-server process cannot be used."""


class CodexAppServerAuthError(CodexAppServerError):
    """Raised when local ChatGPT auth is unavailable or unsupported."""


@dataclass(slots=True)
class CodexAuthSnapshot:
    auth_mode: str | None
    email: str | None
    account_plan_type: str | None
    effective_plan_type: str | None
    requires_openai_auth: bool
    supported_plan: bool
    rate_limits: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CodexTurnResult:
    output_text: str
    tokens_input: int | None
    tokens_output: int | None
    provider_meta: dict[str, Any]


class CodexAppServerClient:
    def __init__(
        self,
        *,
        command: Sequence[str] | None = None,
        startup_timeout_seconds: float | None = None,
        turn_timeout_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self.command = list(command or [settings.codex_bin, "app-server"])
        self.startup_timeout_seconds = (
            startup_timeout_seconds
            if startup_timeout_seconds is not None
            else settings.codex_startup_timeout_seconds
        )
        self.turn_timeout_seconds = (
            turn_timeout_seconds
            if turn_timeout_seconds is not None
            else settings.codex_turn_timeout_seconds
        )
        self._proc: subprocess.Popen[str] | None = None
        self._initialized = False
        self._next_id = 1
        self._notifications: deque[dict[str, Any]] = deque()
        self._reader_queue: queue.Queue[dict[str, Any] | Exception] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self.last_auth_snapshot: CodexAuthSnapshot | None = None

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._initialized = False
        self._notifications.clear()
        self._reader_queue = queue.Queue()
        self._reader_thread = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def read_account(self, *, refresh_token: bool = False) -> dict[str, Any]:
        return self._call_with_restart(
            lambda: self._send_request(
                "account/read",
                {"refreshToken": refresh_token},
                timeout_seconds=self.startup_timeout_seconds,
            )
        )

    def read_rate_limits(self) -> dict[str, Any]:
        return self._call_with_restart(
            lambda: self._send_request(
                "account/rateLimits/read",
                {},
                timeout_seconds=self.startup_timeout_seconds,
            )
        )

    def preflight_auth(self) -> CodexAuthSnapshot:
        account_result = self.read_account(refresh_token=False)
        rate_limit_result = self.read_rate_limits()
        account = account_result.get("account") or {}
        auth_mode = account.get("type")
        account_plan_type = account.get("planType")
        effective_plan_type = self._resolve_effective_plan_type(account_result, rate_limit_result)
        snapshot = CodexAuthSnapshot(
            auth_mode=auth_mode,
            email=account.get("email"),
            account_plan_type=account_plan_type,
            effective_plan_type=effective_plan_type,
            requires_openai_auth=bool(account_result.get("requiresOpenaiAuth")),
            supported_plan=effective_plan_type in SUPPORTED_CHATGPT_PLAN_TYPES,
            rate_limits=rate_limit_result,
        )
        if snapshot.auth_mode != "chatgpt":
            raise CodexAppServerAuthError(
                "Codex app-server requires a local ChatGPT login. Run `afl-agent auth codex login --device-code`."
            )
        if not snapshot.supported_plan:
            raise CodexAppServerAuthError(
                "Codex app-server requires a supported ChatGPT plan. "
                f"Observed account plan={account_plan_type!r}, effective plan={effective_plan_type!r}."
            )
        self.last_auth_snapshot = snapshot
        return snapshot

    def start_device_code_login(self) -> dict[str, Any]:
        result = self._call_with_restart(
            lambda: self._send_request(
                "account/login/start",
                {"type": "chatgptDeviceCode"},
                timeout_seconds=self.startup_timeout_seconds,
            )
        )
        login_id = result.get("loginId")
        if login_id is None:
            raise CodexAppServerError("Device-code login did not return a loginId.")
        return {
            "login_id": login_id,
            "verification_url": result.get("verificationUrl"),
            "user_code": result.get("userCode"),
        }

    def wait_for_device_code_login(
        self,
        *,
        login_id: str,
        timeout_seconds: float = 300.0,
    ) -> CodexAuthSnapshot:
        self._wait_for_notification(
            "account/login/completed",
            timeout_seconds=timeout_seconds,
            predicate=lambda params: params.get("loginId") == login_id,
        )
        return self.preflight_auth()

    def login_device_code(self, *, timeout_seconds: float = 300.0) -> dict[str, Any]:
        login = self.start_device_code_login()
        snapshot = self.wait_for_device_code_login(
            login_id=login["login_id"],
            timeout_seconds=timeout_seconds,
        )
        return {
            **login,
            "auth_snapshot": {
                "auth_mode": snapshot.auth_mode,
                "email": snapshot.email,
                "account_plan_type": snapshot.account_plan_type,
                "effective_plan_type": snapshot.effective_plan_type,
                "requires_openai_auth": snapshot.requires_openai_auth,
                "rate_limits": snapshot.rate_limits,
            },
        }

    def logout(self) -> None:
        self._call_with_restart(
            lambda: self._send_request(
                "account/logout",
                {},
                timeout_seconds=self.startup_timeout_seconds,
            )
        )
        self.last_auth_snapshot = None

    def run_turn(
        self,
        *,
        step_name: str,
        prompt: str,
        input_json: dict[str, Any],
        model_name: str,
        reasoning_effort: str | None,
        output_schema: dict[str, Any],
    ) -> CodexTurnResult:
        auth_snapshot = self.last_auth_snapshot or self.preflight_auth()
        return self._call_with_restart(
            lambda: self._run_turn_once(
                step_name=step_name,
                prompt=prompt,
                input_json=input_json,
                model_name=model_name,
                reasoning_effort=reasoning_effort,
                output_schema=output_schema,
                auth_snapshot=auth_snapshot,
            )
        )

    def _run_turn_once(
        self,
        *,
        step_name: str,
        prompt: str,
        input_json: dict[str, Any],
        model_name: str,
        reasoning_effort: str | None,
        output_schema: dict[str, Any],
        auth_snapshot: CodexAuthSnapshot,
    ) -> CodexTurnResult:
        thread_result = self._send_request(
            "thread/start",
            {"model": model_name},
            timeout_seconds=self.startup_timeout_seconds,
        )
        thread_id = thread_result["thread"]["id"]
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": str(get_settings().workspace_root),
            "approvalPolicy": "never",
            "sandboxPolicy": {
                "type": "readOnly",
                "access": {"type": "fullAccess"},
                "networkAccess": False,
            },
            "model": model_name,
            "summary": "concise",
            "outputSchema": output_schema,
        }
        if reasoning_effort:
            params["effort"] = reasoning_effort
        turn_result = self._send_request(
            "turn/start",
            params,
            timeout_seconds=self.startup_timeout_seconds,
        )
        turn_id = turn_result["turn"]["id"]
        final_text: str | None = None
        token_usage: dict[str, Any] | None = None
        latest_rate_limits = auth_snapshot.rate_limits

        deadline = time.monotonic() + self.turn_timeout_seconds
        while True:
            timeout = max(deadline - time.monotonic(), 0.0)
            if timeout == 0.0:
                raise CodexAppServerProcessError(
                    f"Timed out waiting for Codex turn completion after {self.turn_timeout_seconds} seconds."
                )
            message = self._read_or_pop_notification(timeout)
            method = message.get("method")
            params = message.get("params") or {}
            if method == "item/started":
                item = params.get("item") or {}
                if item.get("type") in {"commandExecution", "mcpToolCall", "fileChange"}:
                    self._best_effort_interrupt(thread_id=thread_id, turn_id=turn_id)
                    raise CodexAppServerError(
                        f"Codex attempted disallowed tool activity for structured step: {item.get('type')}"
                    )
                continue
            if method == "item/completed":
                item = params.get("item") or {}
                if item.get("type") == "agentMessage" and item.get("phase") == "final_answer":
                    final_text = item.get("text")
                continue
            if method == "thread/tokenUsage/updated" and params.get("turnId") == turn_id:
                token_usage = params.get("tokenUsage")
                continue
            if method == "account/rateLimits/updated":
                latest_rate_limits = {"rateLimits": params.get("rateLimits"), "rateLimitsByLimitId": None}
                continue
            if method == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                turn = params["turn"]
                if turn.get("status") != "completed":
                    raise CodexAppServerError(
                        f"Codex turn ended with status={turn.get('status')} error={turn.get('error')!r}"
                    )
                break

        if not final_text:
            raise CodexAppServerError("Codex turn completed without a final JSON response.")
        usage_bucket = (token_usage or {}).get("last") or (token_usage or {}).get("total") or {}
        return CodexTurnResult(
            output_text=final_text,
            tokens_input=usage_bucket.get("inputTokens"),
            tokens_output=usage_bucket.get("outputTokens"),
            provider_meta={
                "thread_id": thread_id,
                "turn_id": turn_id,
                "step_name": step_name,
                "transport": "stdio",
                "auth_mode": auth_snapshot.auth_mode,
                "account_plan_type": auth_snapshot.account_plan_type,
                "plan_type": self._resolve_effective_plan_type(
                    {"account": {"planType": auth_snapshot.account_plan_type}},
                    latest_rate_limits,
                ),
                "rate_limits": latest_rate_limits,
            },
        )

    def _call_with_restart(self, func):
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return func()
            except CodexAppServerProcessError as exc:
                last_error = exc
                self.close()
                if attempt == 1:
                    raise
        if last_error is not None:
            raise last_error
        raise CodexAppServerProcessError("Codex app-server call failed unexpectedly.")

    def _send_request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self._ensure_started()
        request_id = self._next_id
        self._next_id += 1
        self._write_message({"method": method, "id": request_id, "params": params})
        deadline = time.monotonic() + timeout_seconds
        while True:
            timeout = max(deadline - time.monotonic(), 0.0)
            if timeout == 0.0:
                raise CodexAppServerProcessError(
                    f"Timed out waiting for response to {method!r} after {timeout_seconds} seconds."
                )
            message = self._read_or_pop_notification(timeout)
            if message.get("id") == request_id:
                error = message.get("error")
                if error:
                    raise CodexAppServerError(
                        f"Codex app-server request {method!r} failed: {error.get('message') or error}"
                    )
                return message.get("result") or {}
            self._notifications.append(message)

    def _wait_for_notification(
        self,
        method: str,
        *,
        timeout_seconds: float,
        predicate=None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            timeout = max(deadline - time.monotonic(), 0.0)
            if timeout == 0.0:
                raise CodexAppServerProcessError(
                    f"Timed out waiting for notification {method!r} after {timeout_seconds} seconds."
                )
            message = self._read_or_pop_notification(timeout)
            if message.get("method") != method:
                self._notifications.append(message)
                continue
            params = message.get("params") or {}
            if predicate is not None and not predicate(params):
                self._notifications.append(message)
                continue
            if params.get("success") is False:
                raise CodexAppServerAuthError(
                    params.get("error") or f"Codex auth flow {method!r} failed."
                )
            return params

    def _read_or_pop_notification(self, timeout_seconds: float) -> dict[str, Any]:
        if self._notifications:
            return self._notifications.popleft()
        return self._read_message(timeout_seconds)

    def _read_message(self, timeout_seconds: float) -> dict[str, Any]:
        proc = self._proc
        if proc is None:
            raise CodexAppServerProcessError("Codex app-server process is not running.")
        try:
            message = self._reader_queue.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            raise CodexAppServerProcessError("Timed out waiting for Codex app-server output.") from exc
        if isinstance(message, Exception):
            raise message
        return message

    def _ensure_started(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None and self._initialized:
            return
        self.close()
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(get_settings().workspace_root),
        )
        self._initialized = False
        self._notifications.clear()
        self._reader_queue = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc, self._reader_queue),
            daemon=True,
        )
        self._reader_thread.start()
        result = self._send_initialize()
        if not result.get("userAgent"):
            raise CodexAppServerProcessError("Codex app-server initialize response was incomplete.")
        self._write_message({"method": "initialized", "params": {}})
        self._initialized = True

    def _send_initialize(self) -> dict[str, Any]:
        proc = self._proc
        if proc is None:
            raise CodexAppServerProcessError("Codex app-server process could not be started.")
        request = {
            "method": "initialize",
            "id": 0,
            "params": {
                "clientInfo": {
                    "name": "afl_prediction_agent",
                    "title": "AFL Prediction Agent",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "optOutNotificationMethods": [
                        "item/agentMessage/delta",
                        "mcpServer/startupStatus/updated",
                        "skills/changed",
                        "thread/status/changed",
                        "thread/started",
                    ]
                },
            },
        }
        self._write_message(request)
        deadline = time.monotonic() + self.startup_timeout_seconds
        while True:
            timeout = max(deadline - time.monotonic(), 0.0)
            if timeout == 0.0:
                raise CodexAppServerProcessError("Timed out waiting for Codex initialize response.")
            message = self._read_message(timeout)
            if message.get("id") == 0:
                error = message.get("error")
                if error:
                    raise CodexAppServerProcessError(
                        f"Codex app-server initialize failed: {error.get('message') or error}"
                    )
                return message.get("result") or {}
            self._notifications.append(message)

    def _write_message(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise CodexAppServerProcessError("Codex app-server stdin is unavailable.")
        try:
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except BrokenPipeError as exc:
            raise CodexAppServerProcessError("Codex app-server stdin pipe broke.") from exc

    def _best_effort_interrupt(self, *, thread_id: str, turn_id: str) -> None:
        try:
            self._send_request(
                "turn/interrupt",
                {"threadId": thread_id, "turnId": turn_id},
                timeout_seconds=2.0,
            )
        except Exception:
            return

    @staticmethod
    def _resolve_effective_plan_type(
        account_result: dict[str, Any],
        rate_limit_result: dict[str, Any],
    ) -> str | None:
        rate_limits = rate_limit_result.get("rateLimits") or {}
        return (
            rate_limits.get("planType")
            or (account_result.get("account") or {}).get("planType")
        )

    @staticmethod
    def _reader_loop(
        proc: subprocess.Popen[str],
        message_queue: queue.Queue[dict[str, Any] | Exception],
    ) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                try:
                    message_queue.put(json.loads(line))
                except json.JSONDecodeError:
                    message_queue.put(
                        CodexAppServerProcessError(
                            f"Invalid JSON from Codex app-server: {line!r}"
                        )
                    )
                    return
        finally:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read().strip()
            message_queue.put(
                CodexAppServerProcessError(
                    f"Codex app-server exited unexpectedly with code {proc.poll()!r}. {stderr}".strip()
                )
            )


_CLIENT: CodexAppServerClient | None = None


def get_codex_app_server_client() -> CodexAppServerClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = CodexAppServerClient()
    return _CLIENT
