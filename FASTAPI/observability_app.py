import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app import app as fastapi_app

BASE_DIR = Path(__file__).resolve().parent
STRUCTURED_LOG_PATH = BASE_DIR / "task12_structured_logs.jsonl"
_log_lock = Lock()

# Keep these patterns aligned with the fixed-format PII patterns in Task 10.
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)"
)
CARD_LAST4_PATTERN = re.compile(
    r"(?i)(\b(?:card|credit\s*card|debit\s*card|payment\s*card)"
    r"\s*(?:last\s*4|last\s*four|ending\s*in|ending|last4)"
    r"\s*(?:(?:is|:|#|-)\s*)?)(\d{4})\b"
)

def mask_text_for_log(text: str) -> str:
    text = PHONE_PATTERN.sub("[PHONE_MASKED]", text)
    text = CARD_LAST4_PATTERN.sub(
        lambda match: match.group(1) + "[CARD_LAST4_MASKED]", text
    )
    return text

def append_jsonl(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        with STRUCTURED_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

class StructuredLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        trace_id = str(uuid.uuid4())
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        status_code = 500
        request_chunks = []

        # Read the request body once so a safe copy can be logged.
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                break
            if message["type"] == "http.request":
                request_chunks.append(message.get("body", b""))
                more_body = message.get("more_body", False)

        raw_body = b"".join(request_chunks)
        body_text = raw_body.decode("utf-8", errors="replace")
        masked_body = mask_text_for_log(body_text)

        # Replay the original body unchanged to FastAPI.
        body_sent = False
        async def replay_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": raw_body,
                    "more_body": False
                }
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False
            }

        async def send_with_trace(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", trace_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, replay_receive, send_with_trace)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "trace_id": trace_id,
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "request_text": masked_body[:10000]
            }
            append_jsonl(record)

fastapi_app.add_middleware(StructuredLoggingMiddleware)
app = fastapi_app
