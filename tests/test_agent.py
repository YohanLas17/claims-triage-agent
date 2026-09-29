import json
from pathlib import Path

from claims_triage_agent.agent import ClaimsTriageAgent
from claims_triage_agent.llm_client import FakeLLMClient, LLMResponse, ToolCall
from claims_triage_agent.retriever import BM25Retriever
from claims_triage_agent.schema import DecisionStatus

CLAIMS_DIR = Path(__file__).parent.parent / "src" / "claims_triage_agent" / "data" / "claims"


def _load_claim(claim_id: str) -> dict:
    with open(CLAIMS_DIR / f"{claim_id}.json", encoding="utf-8") as f:
        return json.load(f)


def test_agent_denies_cosmetic_removal_using_only_the_free_text_clause():
    """This is the RAG-grounding proof case: the structured policies.json
    table says CPT 11200 IS covered, so a decision that relies only on
    lookup_policy would wrongly approve this claim. The correct 'denied'
    decision is only reachable via the free-text exclusion clause in
    POL-1001.txt Section 4, which is retrieved through search_policy_documents.
    """
    claim = _load_claim("CLM-1001")
    script = [
        LLMResponse(
            tool_calls=[ToolCall(id="1", name="lookup_policy", arguments={"policy_id": "POL-1001"})]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="2",
                    name="search_policy_documents",
                    arguments={
                        "policy_id": "POL-1001",
                        "query": "is cosmetic skin tag removal covered",
                    },
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="3",
                    name="submit_decision",
                    arguments={
                        "status": "denied",
                        "justification": "Cosmetic-only removal is excluded per Section 4.",
                        "cited_passage_ids": ["POL-1001-chunk1"],
                    },
                )
            ]
        ),
    ]
    agent = ClaimsTriageAgent(FakeLLMClient(script), BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.DENIED
    assert trail.decision.cited_passage_ids == ["POL-1001-chunk1"]
    search_calls = [tc for tc in trail.tool_calls if tc.tool_name == "search_policy_documents"]
    assert len(search_calls) == 1
    assert search_calls[0].cited_passages, "the search must have actually returned the clause"


def test_agent_forces_flagged_for_review_when_tool_call_budget_is_exceeded():
    claim = _load_claim("CLM-1001")
    # The model keeps calling a tool forever and never reaches submit_decision.
    looping_script = [
        LLMResponse(
            tool_calls=[
                ToolCall(id=str(i), name="lookup_policy", arguments={"policy_id": "POL-1001"})
            ]
        )
        for i in range(50)
    ]
    agent = ClaimsTriageAgent(FakeLLMClient(looping_script), BM25Retriever(), max_tool_calls=3)
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW
    assert trail.decision.override_reason == "max_tool_calls_exceeded"
    assert len(trail.tool_calls) <= 3


def test_agent_forces_flagged_for_review_when_rag_is_empty_and_uncited():
    claim = _load_claim("CLM-1001")
    script = [
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="1",
                    name="search_policy_documents",
                    arguments={"policy_id": "POL-1001", "query": "zzzznonexistentqueryterm"},
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="2",
                    name="submit_decision",
                    arguments={
                        "status": "approved",
                        "justification": "Looks fine.",
                        "cited_passage_ids": [],
                    },
                )
            ]
        ),
    ]
    agent = ClaimsTriageAgent(FakeLLMClient(script), BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW
    assert trail.decision.override_reason == "ungrounded_decision_after_empty_rag"


def test_agent_does_not_override_when_rag_is_empty_but_decision_is_already_flagged():
    claim = _load_claim("CLM-1001")
    script = [
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="1",
                    name="search_policy_documents",
                    arguments={"policy_id": "POL-1001", "query": "zzzznonexistentqueryterm"},
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="2",
                    name="submit_decision",
                    arguments={
                        "status": "flagged_for_review",
                        "justification": "Could not find a relevant clause; needs human review.",
                        "cited_passage_ids": [],
                    },
                )
            ]
        ),
    ]
    agent = ClaimsTriageAgent(FakeLLMClient(script), BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW
    assert trail.decision.override_reason is None


def test_agent_flags_when_model_never_calls_submit_decision():
    claim = _load_claim("CLM-1001")
    script = [LLMResponse(tool_calls=[], content="I think this claim looks fine overall.")]
    agent = ClaimsTriageAgent(FakeLLMClient(script), BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW
    assert trail.decision.override_reason == "missing_submit_decision_call"


def test_agent_recovers_from_bad_tool_arguments_instead_of_crashing():
    """A tool call with a misnamed/missing argument (e.g. a malformed JSON
    payload that came through as {"__invalid_json__": ...}, or simply a
    typo'd argument name) must not crash the run -- it's returned to the
    model as a tool error, and the model can recover from it.
    """
    claim = _load_claim("CLM-1004")
    script = [
        LLMResponse(
            tool_calls=[
                ToolCall(id="1", name="lookup_policy", arguments={"__invalid_json__": "{bad"})
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(id="2", name="lookup_policy", arguments={"policy_id": "POL-2002"})
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="3",
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
    agent = ClaimsTriageAgent(FakeLLMClient(script), BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.DENIED
    first_call = trail.tool_calls[0]
    assert first_call.tool_name == "lookup_policy"
    assert "error" in first_call.result
    assert [tc.tool_name for tc in trail.tool_calls] == [
        "lookup_policy",
        "lookup_policy",
        "submit_decision",
    ]


def test_agent_recovers_from_invalid_submit_decision_status():
    """An invalid or missing status/justification on submit_decision must
    be returned to the model as a tool error so it can retry, rather than
    raising and crashing the whole run -- still bounded by max_tool_calls.
    """
    claim = _load_claim("CLM-1004")
    script = [
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="1",
                    name="submit_decision",
                    arguments={"status": "not_a_real_status", "justification": "..."},
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(id="2", name="submit_decision", arguments={"status": "denied"})
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="3",
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
    agent = ClaimsTriageAgent(FakeLLMClient(script), BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.DENIED
    assert trail.decision.override_reason is None
    error_calls = [
        tc for tc in trail.tool_calls if tc.tool_name == "submit_decision" and "error" in tc.result
    ]
    assert len(error_calls) == 2


def test_audit_trail_captures_full_sequence_with_timestamps():
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
    agent = ClaimsTriageAgent(FakeLLMClient(script), BM25Retriever())
    trail = agent.run(claim)

    assert trail.run_id.startswith("run-")
    assert trail.started_at and trail.finished_at
    assert [tc.tool_name for tc in trail.tool_calls] == ["lookup_policy", "submit_decision"]
    assert [tc.call_index for tc in trail.tool_calls] == [1, 2]
    assert all(tc.timestamp for tc in trail.tool_calls)
