"""Persistent dedup state for repeating Slack alerts.

Some alerts (drift, health check, any periodic signal) re-fire the
same payload multiple times per day. Without dedup they spam the
user; with in-memory dedup they silence during a single process run
but re-emit on every restart. That produces the worst UX — alerts
that train the operator to ignore them.

This module provides a tiny, file-backed key→(fingerprint, timestamp)
store so dedup survives restarts. The file is JSON, kept under
`/var/lib/arbo/alert_state.json` (or `$ARBO_STATE_DIR/alert_state.json`
if set), and read/written synchronously. Each alert type has its own
key namespace so they don't collide.

Usage:

    from arbo.utils.alert_state import should_alert, record_alert

    fingerprint = compute_fingerprint()
    if should_alert("b2_drift", fingerprint, cooldown_s=24*3600):
        post_to_slack(...)
        record_alert("b2_drift", fingerprint)

Design notes:
- Blocking I/O: the state file is tiny (<1 KB), reads/writes are
  microseconds. Not worth async.
- Atomic write: write to .tmp + rename on POSIX is atomic enough for
  our use (no concurrent writers from multiple processes).
- Corruption-tolerant: bad JSON → start fresh. Don't let a stale
  file wedge the alert system.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def _state_path() -> Path:
    """Resolve the state file location."""
    base = os.environ.get("ARBO_STATE_DIR")
    if base:
        d = Path(base)
    else:
        # Default on VPS: /var/lib/arbo/ (systemd-friendly)
        # Default on dev: repo-local ./.state/
        repo_state = Path(__file__).resolve().parents[2] / ".state"
        d = Path("/var/lib/arbo") if Path("/opt/arbo").exists() else repo_state
    d.mkdir(parents=True, exist_ok=True)
    return d / "alert_state.json"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupted or unreadable — start fresh.
        return {}


def _write_state(data: dict[str, Any]) -> None:
    p = _state_path()
    tmp = Path(tempfile.mkstemp(dir=p.parent, prefix=".alert_", suffix=".tmp")[1])
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        # Best-effort — dedup is soft. Swallow write errors.
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def should_alert(key: str, fingerprint: str, cooldown_s: int = 86400) -> bool:
    """Return True if this (key, fingerprint) hasn't been emitted recently.

    Args:
        key: alert namespace, e.g. "b2_drift", "health_check".
        fingerprint: a stable string summarising the alert content. If
            identical to the last stored fingerprint AND less than
            cooldown_s have passed, we suppress.
        cooldown_s: how long to silence identical alerts. Default 24h.
    """
    state = _read_state()
    entry = state.get(key)
    if not isinstance(entry, dict):
        return True
    last_fp = entry.get("fingerprint")
    last_ts = float(entry.get("ts", 0) or 0)
    if last_fp != fingerprint:
        return True  # new fingerprint → alert
    if time.time() - last_ts >= cooldown_s:
        return True  # same fingerprint but cooldown expired
    return False


def record_alert(key: str, fingerprint: str) -> None:
    """Record that an alert with this fingerprint was just emitted."""
    state = _read_state()
    state[key] = {
        "fingerprint": fingerprint,
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_state(state)


def clear_alert(key: str) -> None:
    """Forget a specific alert (force next emission)."""
    state = _read_state()
    if key in state:
        del state[key]
        _write_state(state)
