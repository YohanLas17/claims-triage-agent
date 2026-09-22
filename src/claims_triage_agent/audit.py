"""Persistence for the audit trail.

Every run of the agent produces an ``AuditTrail`` (see ``schema.py``)
capturing the full tool-call sequence, arguments, results, timestamps and
cited passages. This module is only responsible for writing that trail
to disk as JSON -- one file per run -- which is the format a compliance
or QA process would pull from in a real deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

from claims_triage_agent.schema import AuditTrail

DEFAULT_AUDIT_DIR = Path(__file__).parent / "data" / "audit_logs"


def write_audit_trail(trail: AuditTrail, directory: Path = DEFAULT_AUDIT_DIR) -> Path:
    """Write ``trail`` to ``<directory>/<run_id>.json`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trail.run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trail.to_dict(), f, indent=2, ensure_ascii=False)
    return path
