from __future__ import annotations

import json
import sys
import textwrap

import pytest

from afl_prediction_agent.agents.codex_app_server import (
    CodexAppServerAuthError,
    CodexAppServerClient,
)


def _client_for_script(script: str, *, startup_timeout: float = 5.0, turn_timeout: float = 10.0):
    return CodexAppServerClient(
        command=[sys.executable, "-u", "-c", script],
        startup_timeout_seconds=startup_timeout,
        turn_timeout_seconds=turn_timeout,
    )


def test_preflight_auth_accepts_supported_chatgpt_plan_from_rate_limits(monkeypatch) -> None:
    client = CodexAppServerClient(command=["codex", "app-server"])
    monkeypatch.setattr(
        client,
        "read_account",
        lambda refresh_token=False: {
            "account": {"type": "chatgpt", "email": "user@example.com", "planType": "plus"},
            "requiresOpenaiAuth": True,
        },
    )
    monkeypatch.setattr(
        client,
        "read_rate_limits",
        lambda: {"rateLimits": {"planType": "pro"}},
    )

    snapshot = client.preflight_auth()

    assert snapshot.auth_mode == "chatgpt"
    assert snapshot.account_plan_type == "plus"
    assert snapshot.effective_plan_type == "pro"
    assert snapshot.supported_plan is True


def test_preflight_auth_rejects_non_chatgpt_auth(monkeypatch) -> None:
    client = CodexAppServerClient(command=["codex", "app-server"])
    monkeypatch.setattr(
        client,
        "read_account",
        lambda refresh_token=False: {
            "account": {"type": "apiKey", "planType": None},
            "requiresOpenaiAuth": True,
        },
    )
    monkeypatch.setattr(
        client,
        "read_rate_limits",
        lambda: {"rateLimits": {"planType": None}},
    )

    with pytest.raises(CodexAppServerAuthError):
        client.preflight_auth()


def test_device_code_login_flow() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        def emit(payload):
            print(json.dumps(payload), flush=True)

        for line in sys.stdin:
            message = json.loads(line)
            method = message["method"]
            if method == "initialize":
                emit({"id": message["id"], "result": {"userAgent": "fake"}})
            elif method == "initialized":
                continue
            elif method == "account/login/start":
                emit({
                    "id": message["id"],
                    "result": {
                        "type": "chatgptDeviceCode",
                        "loginId": "login-1",
                        "verificationUrl": "https://auth.openai.com/codex/device",
                        "userCode": "ABCD-1234"
                    }
                })
                emit({
                    "method": "account/login/completed",
                    "params": {"loginId": "login-1", "success": True, "error": None}
                })
            elif method == "account/read":
                emit({
                    "id": message["id"],
                    "result": {
                        "account": {
                            "type": "chatgpt",
                            "email": "user@example.com",
                            "planType": "plus"
                        },
                        "requiresOpenaiAuth": True
                    }
                })
            elif method == "account/rateLimits/read":
                emit({
                    "id": message["id"],
                    "result": {"rateLimits": {"planType": "pro"}}
                })
        """
    )
    client = _client_for_script(script)

    try:
        result = client.login_device_code(timeout_seconds=5.0)
    finally:
        client.close()

    assert result["login_id"] == "login-1"
    assert result["verification_url"] == "https://auth.openai.com/codex/device"
    assert result["user_code"] == "ABCD-1234"
    assert result["auth_snapshot"]["auth_mode"] == "chatgpt"
    assert result["auth_snapshot"]["effective_plan_type"] == "pro"


def test_turn_result_uses_final_answer_and_ignores_commentary() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        def emit(payload):
            print(json.dumps(payload), flush=True)

        for line in sys.stdin:
            message = json.loads(line)
            method = message["method"]
            if method == "initialize":
                emit({"id": message["id"], "result": {"userAgent": "fake"}})
            elif method == "initialized":
                continue
            elif method == "account/read":
                emit({
                    "id": message["id"],
                    "result": {
                        "account": {"type": "chatgpt", "email": "user@example.com", "planType": "plus"},
                        "requiresOpenaiAuth": True
                    }
                })
            elif method == "account/rateLimits/read":
                emit({
                    "id": message["id"],
                    "result": {"rateLimits": {"planType": "pro"}}
                })
            elif method == "thread/start":
                emit({"id": message["id"], "result": {"thread": {"id": "thr_123"}}})
            elif method == "turn/start":
                emit({"id": message["id"], "result": {"turn": {"id": "turn_123", "status": "inProgress", "items": [], "error": None}}})
                emit({
                    "method": "item/completed",
                    "params": {
                        "item": {"type": "agentMessage", "id": "msg_1", "text": "ignore me", "phase": "commentary"},
                        "threadId": "thr_123",
                        "turnId": "turn_123"
                    }
                })
                emit({
                    "method": "item/completed",
                    "params": {
                        "item": {"type": "agentMessage", "id": "msg_2", "text": "{\\"ok\\": true}", "phase": "final_answer"},
                        "threadId": "thr_123",
                        "turnId": "turn_123"
                    }
                })
                emit({
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "thr_123",
                        "turnId": "turn_123",
                        "tokenUsage": {"last": {"inputTokens": 111, "outputTokens": 22}}
                    }
                })
                emit({
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thr_123",
                        "turn": {"id": "turn_123", "status": "completed", "error": None}
                    }
                })
        """
    )
    client = _client_for_script(script)

    try:
        result = client.run_turn(
            step_name="final_decision_v1",
            prompt="Return JSON",
            input_json={"dossier": {}},
            model_name="gpt-5.4",
            reasoning_effort="xhigh",
            output_schema={"type": "object"},
        )
    finally:
        client.close()

    assert json.loads(result.output_text) == {"ok": True}
    assert result.tokens_input == 111
    assert result.tokens_output == 22
    assert result.provider_meta["thread_id"] == "thr_123"
    assert result.provider_meta["turn_id"] == "turn_123"


def test_client_restarts_after_process_death(tmp_path) -> None:
    counter_path = tmp_path / "launch_count.txt"
    script = textwrap.dedent(
        f"""
        import json
        import pathlib
        import sys

        counter_path = pathlib.Path({str(counter_path)!r})
        launch_count = int(counter_path.read_text() or "0") + 1 if counter_path.exists() else 1
        counter_path.write_text(str(launch_count))

        def emit(payload):
            print(json.dumps(payload), flush=True)

        for line in sys.stdin:
            message = json.loads(line)
            method = message["method"]
            if method == "initialize":
                emit({{"id": message["id"], "result": {{"userAgent": "fake"}}}})
            elif method == "initialized":
                continue
            elif method == "account/read":
                if launch_count == 1:
                    sys.exit(0)
                emit({{
                    "id": message["id"],
                    "result": {{
                        "account": {{"type": "chatgpt", "email": "user@example.com", "planType": "plus"}},
                        "requiresOpenaiAuth": True
                    }}
                }})
            elif method == "account/rateLimits/read":
                emit({{"id": message["id"], "result": {{"rateLimits": {{"planType": "pro"}}}}}})
        """
    )
    client = _client_for_script(script)

    try:
        snapshot = client.preflight_auth()
    finally:
        client.close()

    assert snapshot.auth_mode == "chatgpt"
    assert counter_path.read_text() == "2"
