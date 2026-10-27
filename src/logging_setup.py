"""Logging setup with secret redaction.

Any log message or argument that contains the value of a secret environment
variable (API keys, passwords, DB URLs with credentials) is masked before it
is written. This is a safety net; code should still never log secrets.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys

SECRET_ENV_PATTERN = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASS|CREDENTIAL)", re.IGNORECASE)
# user:password@host in connection URLs
URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<user>[^:/@\s]+):(?P<pw>[^@\s]+)@")
MASK = "***REDACTED***"


def _secret_values() -> list[str]:
    vals = []
    for k, v in os.environ.items():
        if v and len(v) >= 6 and SECRET_ENV_PATTERN.search(k) and v != "proxy-injected":
            vals.append(v)
    return sorted(set(vals), key=len, reverse=True)


def redact(text: str) -> str:
    text = URL_CREDENTIALS.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:{MASK}@", text)
    for secret in _secret_values():
        text = text.replace(secret, MASK)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - never break logging
            return True
        record.msg = redact(msg)
        record.args = None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload)


_CONFIGURED = False


def setup_logging(level: str | None = None, json_logs: bool = False) -> None:
    global _CONFIGURED
    level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RedactingFilter())
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    )
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("urllib3", "yfinance", "peewee", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
