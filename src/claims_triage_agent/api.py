"""FastAPI surface: a single POST /adjudicate endpoint.

This module is the only place in the codebase that imports FastAPI or
pydantic -- the domain logic in ``agent.py`` / ``tools.py`` / ``retriever.py``
knows nothing about HTTP. ``create_app`` takes the ``LLMClient`` and
``Retriever`` as arguments so tests can inject a ``FakeLLMClient`` and get
a fully working app with no network access or API key
(see ``tests/test_api.py``).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from claims_triage_agent.agent import ClaimsTriageAgent
from claims_triage_agent.llm_client import LLMClient
from claims_triage_agent.retriever import BM25Retriever, Retriever


class ClaimRequest(BaseModel):
    claim_id: str
    patient_id: str
    policy_id: str
    procedure_code: str
    procedure_description: str
    billed_amount: float = Field(gt=0)
    date_of_service: str
    notes: str = ""


class ToolCallResponse(BaseModel):
    call_index: int
    tool_name: str
    arguments: dict
    result: Any
    timestamp: str
    cited_passages: list[dict]


class DecisionResponse(BaseModel):
    status: str
    justification: str
    cited_passage_ids: list[str]
    override_reason: str | None


class AdjudicationResponse(BaseModel):
    run_id: str
    claim_id: str
    started_at: str
    finished_at: str | None
    decision: DecisionResponse
    audit_trail: list[ToolCallResponse]


def create_app(llm_client: LLMClient, retriever: Retriever | None = None) -> FastAPI:
    """Build the FastAPI app, with the LLM client and retriever injected.

    Production entry points construct a real ``LLMClient`` (see
    ``llm_client.OpenAIChatCompletionsClient``) and pass it in; tests pass
    a ``FakeLLMClient`` instead, so the whole HTTP layer is testable
    without any network access.
    """
    app = FastAPI(title="claims-triage-agent", version="0.1.0")
    retriever = retriever or BM25Retriever()
    agent = ClaimsTriageAgent(llm_client=llm_client, retriever=retriever)

    @app.post("/adjudicate", response_model=AdjudicationResponse)
    def adjudicate(claim: ClaimRequest) -> AdjudicationResponse:
        trail = agent.run(claim.model_dump())
        if trail.decision is None:  # pragma: no cover - defensive, agent always sets it
            raise HTTPException(status_code=500, detail="Agent produced no decision.")
        trail_dict = trail.to_dict()
        return AdjudicationResponse(
            run_id=trail_dict["run_id"],
            claim_id=trail_dict["claim_id"],
            started_at=trail_dict["started_at"],
            finished_at=trail_dict["finished_at"],
            decision=trail_dict["decision"],
            audit_trail=trail_dict["tool_calls"],
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
