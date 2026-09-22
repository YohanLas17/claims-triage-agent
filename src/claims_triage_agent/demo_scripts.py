"""Hand-written, known-correct LLM trajectories for the 4 synthetic demo claims.

These are NOT a claim about any model's quality -- each entry is a scripted
sequence of tool calls a competent adjudicator would produce for that
specific claim, written by hand. They exist so:

- ``eval/run_eval.py --llm fake-reference`` can exercise the eval harness
  itself (loading cases, running the agent, scoring, reporting) with zero
  network access or API key.
- ``claims_triage_agent.demo`` can serve a real, running FastAPI app over
  these same trajectories via ``ReferenceScriptLLMClient``, so the
  architecture (tool-calling loop, RAG, reliability guardrails, audit
  trail) can be explored interactively through ``/docs`` with no API key.

Point ``--llm openai`` at ``eval/run_eval.py`` to get a real accuracy
number for an actual model instead.
"""

from __future__ import annotations

from claims_triage_agent.llm_client import LLMResponse, ToolCall

DEMO_SCRIPTS: dict[str, list[LLMResponse]] = {
    "CLM-1001": [
        LLMResponse(
            tool_calls=[
                ToolCall(id="1", name="lookup_policy", arguments={"policy_id": "POL-1001"})
            ]
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
                        "justification": (
                            "Cosmetic-only removal, with no irritation/bleeding "
                            "documented, is excluded per Section 4."
                        ),
                        "cited_passage_ids": ["POL-1001-chunk1"],
                    },
                )
            ]
        ),
    ],
    "CLM-1002": [
        LLMResponse(
            tool_calls=[
                ToolCall(id="1", name="check_prior_claims", arguments={"patient_id": "PAT-001"})
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="2",
                    name="search_policy_documents",
                    arguments={
                        "policy_id": "POL-1001",
                        "query": "repeat knee arthroscopy same joint within 90 days",
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
                        "status": "flagged_for_review",
                        "justification": (
                            "Same-joint arthroscopy within 90 days of a prior approved "
                            "arthroscopy requires peer-to-peer review per Section 7."
                        ),
                        "cited_passage_ids": ["POL-1001-chunk3"],
                    },
                )
            ]
        ),
    ],
    "CLM-1003": [
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="1",
                    name="search_policy_documents",
                    arguments={
                        "policy_id": "POL-2002",
                        "query": "brain MRI prior authorization emergency department",
                    },
                )
            ]
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    id="2",
                    name="calculate_coverage",
                    arguments={
                        "policy_id": "POL-2002",
                        "procedure_code": "70551",
                        "billed_amount": 2100.0,
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
                        "status": "approved",
                        "justification": (
                            "ED-ordered brain MRI with documented ED context and "
                            "retrospective auth number; prior auth waived per Section 3."
                        ),
                        "cited_passage_ids": ["POL-2002-chunk1"],
                    },
                )
            ]
        ),
    ],
    "CLM-1004": [
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
                        "justification": "CPT 27447 is not on POL-2002's covered-procedures list.",
                        "cited_passage_ids": [],
                    },
                )
            ]
        ),
    ],
}
