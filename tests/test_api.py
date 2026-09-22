import json
from pathlib import Path

from fastapi.testclient import TestClient

from claims_triage_agent.api import create_app
from claims_triage_agent.llm_client import FakeLLMClient, LLMResponse, ToolCall

CLAIMS_DIR = Path(__file__).parent.parent / "src" / "claims_triage_agent" / "data" / "claims"


def _load_claim(claim_id: str) -> dict:
    with open(CLAIMS_DIR / f"{claim_id}.json", encoding="utf-8") as f:
        return json.load(f)


def test_health_endpoint():
    app = create_app(llm_client=FakeLLMClient([]))
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_adjudicate_endpoint_returns_decision_and_audit_trail():
    claim = _load_claim("CLM-1004")
    script = [
        LLMResponse(
            tool_calls=[
                ToolCall(id="1", name="lookup_policy", arguments={"policy_id": "POL-2002"})
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="2",
                    name="submit_decision",
                    arguments={
                        "status": "denied",
                        "justification": "CPT 27447 is not on this plan's covered-procedures list.",
                        "cited_passage_ids": [],
                    },
                )
            ]
        ),
    ]
    app = create_app(llm_client=FakeLLMClient(script))
    client = TestClient(app)

    response = client.post("/adjudicate", json=claim)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["status"] == "denied"
    assert body["claim_id"] == "CLM-1004"
    assert len(body["audit_trail"]) == 2
    assert body["audit_trail"][0]["tool_name"] == "lookup_policy"


def test_adjudicate_endpoint_rejects_invalid_claim_payload():
    app = create_app(llm_client=FakeLLMClient([]))
    client = TestClient(app)
    response = client.post("/adjudicate", json={"claim_id": "CLM-BAD"})
    assert response.status_code == 422
