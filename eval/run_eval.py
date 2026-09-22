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
from claims_triage_agent.demo_scripts import DEMO_SCRIPTS as _REFERENCE_SCRIPTS  # noqa: E402
from claims_triage_agent.llm_client import FakeLLMClient  # noqa: E402
from claims_triage_agent.retriever import BM25Retriever  # noqa: E402

EVAL_DIR = Path(__file__).parent
CLAIMS_DIR = EVAL_DIR.parent / "src" / "claims_triage_agent" / "data" / "claims"


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
