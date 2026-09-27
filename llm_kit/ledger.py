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

from .client import Usage, estimate_cost, speech_cost

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
    return _append({
        "ts": int(time.time()),
        "provider": provider,
        "model": model,
        "in": int(usage.input_tokens),
        "out": int(usage.output_tokens),
        "cost": round(estimate_cost(usage, model) if cost is None else cost, 6),
    }, path)


def record_speech(provider: str, model: str, chars: int, *, cost: float | None = None,
                  path: str | os.PathLike | None = None) -> dict:
    """Append one text-to-speech call. Speech is billed by characters, so the row
    carries `chars` (tokens stay 0) and is priced against `SPEECH_PRICES`."""
    return _append({
        "ts": int(time.time()),
        "provider": provider,
        "model": model,
        "in": 0,
        "out": 0,
        "chars": int(chars),
        "cost": round(speech_cost(model, chars) if cost is None else cost, 6),
    }, path)


def _append(row: dict, path) -> dict:
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


def read_usage(*, since: float | None = None, path: str | os.PathLike | None = None,
               reprice: bool = True) -> list[dict]:
    """Every ledger row (optionally since an epoch time), oldest first. Missing
    file or bad lines are skipped, not errors.

    A row written as $0 because its model wasn't in the price table yet is
    re-priced against today's tables (and flagged `repriced`), so adding a price
    fixes history too. Rows that carry a real cost are left alone."""
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
            if reprice and not float(row.get("cost") or 0):
                _reprice(row)
            rows.append(row)
    return rows


def _reprice(row: dict) -> None:
    model = str(row.get("model") or "")
    if row.get("chars"):
        cost = speech_cost(model, int(row["chars"]))
    else:
        cost = estimate_cost(Usage(int(row.get("in") or 0), int(row.get("out") or 0)), model)
    if cost:
        row["cost"] = round(cost, 6)
        row["repriced"] = True
