"""Secrets hygiene and real-money safety."""
import io
import logging
import pathlib
import re

import pytest

from src.errors import LiveTradingNotAllowedError
from src.execution.broker import assert_safe_mode, create_broker, startup_banner
from src.logging_setup import RedactingFilter, redact


def test_gitignore_excludes_secrets():
    gi = pathlib.Path(".gitignore").read_text().splitlines()
    for pattern in (".env", "*.pem", "*.key", "secrets/"):
        assert pattern in gi
    assert "!.env.example" in gi


def test_env_example_has_no_real_secrets():
    txt = pathlib.Path(".env.example").read_text()
    for line in txt.splitlines():
        if re.match(r"^\s*[A-Z_]*(KEY|SECRET|TOKEN)\s*=", line):
            assert line.split("=", 1)[1].strip() == "", f"value present: {line}"
    assert "CHANGE_ME" in txt


def test_no_hardcoded_credentials_in_source():
    pat = re.compile(r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|api_key\s*=\s*['\"][^'\"]{8,}['\"])")
    for p in pathlib.Path("src").rglob("*.py"):
        assert not pat.search(p.read_text()), p


def test_logs_redact_secrets(monkeypatch):
    monkeypatch.setenv("BROKER_API_SECRET", "supersecretvalue123")
    assert "supersecretvalue123" not in redact("auth failed with supersecretvalue123")
    assert "hunter22" not in redact("postgresql+psycopg://trader:hunter22@db:5432/trading")
    stream = io.StringIO()
    h = logging.StreamHandler(stream)
    h.addFilter(RedactingFilter())
    lg = logging.getLogger("t_sec")
    lg.addHandler(h)
    lg.propagate = False
    lg.error("connecting with key %s", "supersecretvalue123")
    assert "supersecretvalue123" not in stream.getvalue() and "REDACTED" in stream.getvalue()


def test_live_mode_is_refused_everywhere(monkeypatch):
    for m in ("LIVE", "live", "REAL", "prod"):
        with pytest.raises(LiveTradingNotAllowedError):
            assert_safe_mode(m)
        with pytest.raises(LiveTradingNotAllowedError):
            create_broker(m)
    monkeypatch.setenv("TRADING_MODE", "LIVE")
    with pytest.raises(LiveTradingNotAllowedError):
        create_broker()
    assert create_broker("PAPER").is_paper


def test_startup_banner(capsys):
    b = startup_banner("PAPER")
    out = capsys.readouterr().out if hasattr(capsys, "readouterr") else b
    assert "TRADING MODE: PAPER" in b and "REAL MONEY: DISABLED" in b
    assert "REAL MONEY: DISABLED" in out


def test_no_live_broker_adapter_exists():
    names = [p.stem for p in pathlib.Path("src/execution").glob("*.py")]
    assert not any("live" in n for n in names)


def test_tests_are_isolated_from_the_users_env():
    """The user's .env (e.g. BROKER=alpaca_paper + real paper keys) must never leak into tests."""
    import os

    from src.config import load_config
    from src.errors import ConfigurationError
    from src.execution.alpaca_paper_broker import PAPER_URL, AlpacaPaperBroker
    load_config()
    for var in ("BROKER", "ALPACA_API_KEY", "ALPACA_API_SECRET", "TRADING_ENABLED", "DATABASE_URL"):
        assert var not in os.environ, var
    with pytest.raises(ConfigurationError, match="tests"):
        AlpacaPaperBroker("k", "s", PAPER_URL)                  # no injected fake session -> refused
