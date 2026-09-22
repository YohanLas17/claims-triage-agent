import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from claims_triage_agent.demo import app

CLAIMS_DIR = Path(__file__).parent.parent / "src" / "claims_triage_agent" / "data" / "claims"
EVAL_CASES_PATH = Path(__file__).parent.parent / "eval" / "eval_cases.json"

with open(EVAL_CASES_PATH, encoding="utf-8") as f:
    EVAL_CASES = json.load(f)


def _load_claim(claim_id: str) -> dict:
    with open(CLAIMS_DIR / f"{claim_id}.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["claim_id"] for c in EVAL_CASES])
def test_demo_adjudicate_matches_expected_status(case: dict) -> None:
    claim = _load_claim(case["claim_id"])
    client = TestClient(app)

    response = client.post("/adjudicate", json=claim)

    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == case["claim_id"]
    assert body["decision"]["status"] == case["expected_status"]
    if case["requires_rag"]:
        assert case["expected_cited_chunk"] in body["decision"]["cited_passage_ids"]


def test_demo_adjudicate_rejects_unknown_claim_id_with_a_clear_error() -> None:
    claim = _load_claim("CLM-1004")
    claim["claim_id"] = "CLM-9999"
    client = TestClient(app)

    with pytest.raises(ValueError, match="CLM-9999") as exc_info:
        client.post("/adjudicate", json=claim)

    assert "demo" in str(exc_info.value).lower()
