#!/usr/bin/env python3
"""Evaluation harness for the claims-triage agent.

Usage:
    python eval/run_eval.py --llm openai          # evaluate a real model (needs OPENAI_API_KEY)
    python eval/run_eval.py --llm fake-reference   # exercise the harness offline, no API key

``fake-reference`` replays a hand-written, known-correct tool-call
trajectory for each case. It is NOT a claim about any model's quality --
it exists so the harness itself (loading cases, running the agent,
scoring, reporting) can be run and inspected with zero network access or
API key, the same way the test suite can. Point ``--llm openai`` at this
harness to get a real accuracy number for a specific model.

Metrics reported, in the spirit of a real ML eval rather than a vibe
check:
  - decision accuracy: exact match of predicted status vs. expected_status
  - citation accuracy: for cases where requires_rag is true, whether the
    expected chunk_id appears in the decision's cited_passage_ids
Both are reported per-case and as an aggregate over the whole suite, and
every run is reproducible from the cases file plus (for fake-reference)
no external state at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claims_triage_agent.agent import ClaimsTriageAgent  # noqa: E402
from claims_triage_agent.llm_client import FakeLLMClient, LLMResponse, ToolCall  # noqa: E402
from claims_triage_agent.retriever import BM25Retriever  # noqa: E402

EVAL_DIR = Path(__file__).parent
CLAIMS_DIR = EVAL_DIR.parent / "src" / "claims_triage_agent" / "data" / "claims"

# Hand-written, known-correct trajectories for --llm fake-reference.
# Each maps a claim_id to the scripted LLMResponse sequence a competent
# adjudicator would produce for that specific claim.
_REFERENCE_SCRIPTS: dict[str, list[LLMResponse]] = {
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


def _build_llm_client(kind: str, claim_id: str):
    if kind == "fake-reference":
        return FakeLLMClient(_REFERENCE_SCRIPTS[claim_id])
    if kind == "openai":
        from claims_triage_agent.llm_client import OpenAIChatCompletionsClient

        return OpenAIChatCompletionsClient()
    raise ValueError(f"Unknown --llm value: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm", choices=["openai", "fake-reference"], default="fake-reference"
    )
    args = parser.parse_args()

    with open(EVAL_DIR / "eval_cases.json", encoding="utf-8") as f:
        cases = json.load(f)

    retriever = BM25Retriever()
    results = []
    for case in cases:
        claim_id = case["claim_id"]
        with open(CLAIMS_DIR / f"{claim_id}.json", encoding="utf-8") as f:
            claim = json.load(f)

        llm_client = _build_llm_client(args.llm, claim_id)
        agent = ClaimsTriageAgent(llm_client=llm_client, retriever=retriever)
        trail = agent.run(claim)

        decision_correct = trail.decision.status.value == case["expected_status"]
        citation_correct = True
        if case["requires_rag"]:
            citation_correct = case["expected_cited_chunk"] in trail.decision.cited_passage_ids

        results.append(
            {
                "claim_id": claim_id,
                "expected_status": case["expected_status"],
                "actual_status": trail.decision.status.value,
                "decision_correct": decision_correct,
                "requires_rag": case["requires_rag"],
                "citation_correct": citation_correct,
                "override_reason": trail.decision.override_reason,
            }
        )

    n = len(results)
    decision_accuracy = sum(r["decision_correct"] for r in results) / n
    rag_cases = [r for r in results if r["requires_rag"]]
    citation_accuracy = (
        sum(r["citation_correct"] for r in rag_cases) / len(rag_cases)
        if rag_cases
        else float("nan")
    )

    print(f"LLM backend: {args.llm}\n")
    for r in results:
        mark = "PASS" if r["decision_correct"] and r["citation_correct"] else "FAIL"
        print(
            f"[{mark}] {r['claim_id']}: expected={r['expected_status']!r} "
            f"actual={r['actual_status']!r} citation_correct={r['citation_correct']} "
            f"override={r['override_reason']}"
        )

    n_correct = sum(r["decision_correct"] for r in results)
    print(f"\nDecision accuracy: {decision_accuracy:.2%} ({n_correct}/{n})")
    if rag_cases:
        print(
            f"Citation accuracy (RAG-required cases): {citation_accuracy:.2%} "
            f"({sum(r['citation_correct'] for r in rag_cases)}/{len(rag_cases)})"
        )


if __name__ == "__main__":
    main()
