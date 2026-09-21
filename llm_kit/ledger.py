"""A provider-neutral usage ledger shared by every app on a machine.

Each hosted-model call can append one line — when, which provider and model,
tokens in/out, estimated dollars — to a JSON-lines file. Any app can read it
back to show spend. Deliberately there is **no app field**: the ledger answers
"what did this machine spend on which model", never "which app spent it", so a
dashboard built on it can't reveal what's installed.

Path: ``$LLM_USAGE_LEDGER``, default ``~/.local/share/llm_kit/usage.jsonl``.
Services that should share one ledger must run as the same user (or point the
env var at the same writable file).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .client import Usage, estimate_cost

LEDGER_ENV = "LLM_USAGE_LEDGER"
MAX_BYTES = 2_000_000     # past this, keep the newest half — telemetry, not records


def ledger_path(path: str | os.PathLike | None = None) -> Path:
    if path:
        return Path(path)
    env = os.environ.get(LEDGER_ENV)
    return Path(env) if env else Path.home() / ".local" / "share" / "llm_kit" / "usage.jsonl"


def record_usage(provider: str, model: str, usage: Usage, *, cost: float | None = None,
                 path: str | os.PathLike | None = None) -> dict:
    """Append one call to the ledger and return the row. Never raises on I/O —
    losing a telemetry line must not fail the caller's request."""
    row = {
        "ts": int(time.time()),
        "provider": provider,
        "model": model,
        "in": int(usage.input_tokens),
        "out": int(usage.output_tokens),
        "cost": round(estimate_cost(usage, model) if cost is None else cost, 6),
    }
    p = ledger_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if p.stat().st_size > MAX_BYTES:
            lines = p.read_text(encoding="utf-8").splitlines()
            p.write_text("\n".join(lines[len(lines) // 2:]) + "\n", encoding="utf-8")
    except OSError:
        pass
    return row


def read_usage(*, since: float | None = None, path: str | os.PathLike | None = None) -> list[dict]:
    """Every ledger row (optionally since an epoch time), oldest first. Missing
    file or bad lines are skipped, not errors."""
    p = ledger_path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is None or float(row.get("ts") or 0) >= since:
            rows.append(row)
    return rows
