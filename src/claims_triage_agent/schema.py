"""Domain data model for the claims-triage-agent.

These are plain dataclasses rather than pydantic models on purpose: they
are the internal domain representation used by the agent loop, the tools
and the retriever, and they must stay importable and testable without
pulling in the web-framework layer (see ``api.py`` for the pydantic
request/response schemas used at the HTTP boundary).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DecisionStatus(str, Enum):
    """The three possible outcomes of a claim adjudication run."""

    APPROVED = "approved"
    DENIED = "denied"
    FLAGGED_FOR_REVIEW = "flagged_for_review"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with a 'Z' suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class Claim:
    """A single, synthetic health-insurance claim submitted for adjudication."""

    claim_id: str
    patient_id: str
    policy_id: str
    procedure_code: str
    procedure_description: str
    billed_amount: float
    date_of_service: str
    notes: str = ""


@dataclass(frozen=True)
class RetrievedPassage:
    """One passage returned by the retriever, with its provenance and score."""

    policy_id: str
    document_id: str
    chunk_id: str
    text: str
    score: float


@dataclass
class ToolCallRecord:
    """A single tool invocation captured for the audit trail."""

    call_index: int
    tool_name: str
    arguments: dict[str, Any]
    result: Any
    timestamp: str = field(default_factory=utc_now_iso)
    cited_passages: list[RetrievedPassage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "timestamp": self.timestamp,
            "cited_passages": [
                {
                    "policy_id": p.policy_id,
                    "document_id": p.document_id,
                    "chunk_id": p.chunk_id,
                    "text": p.text,
                    "score": p.score,
                }
                for p in self.cited_passages
            ],
        }


@dataclass
class Decision:
    """The structured, justified outcome of a claims-triage run."""

    status: DecisionStatus
    justification: str
    cited_passage_ids: list[str] = field(default_factory=list)
    override_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "justification": self.justification,
            "cited_passage_ids": self.cited_passage_ids,
            "override_reason": self.override_reason,
        }


@dataclass
class AuditTrail:
    """The complete, append-only record of one adjudication run."""

    run_id: str
    claim_id: str
    started_at: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    decision: Decision | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "claim_id": self.claim_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "decision": self.decision.to_dict() if self.decision else None,
        }
