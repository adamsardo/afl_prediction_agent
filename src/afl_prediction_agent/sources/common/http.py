from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from afl_prediction_agent.core.settings import get_settings


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}


@dataclass(slots=True)
class HttpResponse:
    text: str
    json_data: Any | None
    status_code: int
    headers: dict[str, str]


class HttpFetchClient:
    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        retry_attempts: int | None = None,
    ) -> None:
        settings = get_settings()
        self.timeout_seconds = timeout_seconds or settings.source_http_timeout_seconds
        self.retry_attempts = retry_attempts or settings.source_retry_attempts

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        expect_json: bool = False,
    ) -> HttpResponse:
        last_error: Exception | None = None
        merged_headers = dict(headers or {})
        for attempt in range(1, self.retry_attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
                    response = client.get(url, headers=merged_headers or None, params=params)
                response.raise_for_status()
                return HttpResponse(
                    text=response.text,
                    json_data=response.json() if expect_json else None,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )
            except Exception as exc:  # pragma: no cover - exercised through callers
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                time.sleep(min(0.5 * attempt, 2.0))
        assert last_error is not None
        raise last_error

