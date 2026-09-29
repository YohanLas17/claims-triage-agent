#!/usr/bin/env python3
"""Evaluation CLI for the claims-triage agent.

All scoring, failure-taxonomy classification, baselines and report
rendering live in ``claims_triage_agent.evaluation`` (typed and unit
tested independently of this CLI, see ``tests/test_evaluation.py``).
This file only does argument parsing, wiring a backend, driving the
agent loop over the case list, and writing the report to disk.

Usage:
    python eval/run_eval.py --llm baseline-approve
    python eval/run_eval.py --llm baseline-structured
    python eval/run_eval.py --llm fake-reference
    python eval/run_eval.py --llm openai --model gpt-4o-mini
    python eval/run_eval.py --llm openai \\
        --model gemini-2.5-flash \\
        --base-url https://generativelanguage.googleapis.com/v1beta/openai/ \\
        --api-key-env GEMINI_API_KEY --runs 3 --sleep 4

``fake-reference`` replays hand-written, known-correct tool-call
trajectories for the 4 original demo claims only (see
``demo_scripts.DEMO_SCRIPTS``). It validates that the harness itself
(loading cases, running the agent, scoring, reporting) works end to end
with zero network access -- it is NOT a claim about any model's
quality. The two ``baseline-*`` backends don't call a model at all; they
exist so every real accuracy number has something honest to compare
against. Point ``--llm openai`` at any OpenAI-compatible endpoint (a
real OpenAI key, Gemini's free-tier OpenAI-compatible endpoint, or a
local Ollama server) to get a real accuracy number for an actual model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claims_triage_agent.agent import ClaimsTriageAgent  # noqa: E402
from claims_triage_agent.demo_scripts import DEMO_SCRIPTS  # noqa: E402
from claims_triage_agent.evaluation import (  # noqa: E402
    BASELINES,
    CaseResult,
    Summary,
    render_markdown_report,
    result_from_trail,
    summarize,
    summarize_consistency,
)
from claims_triage_agent.llm_client import FakeLLMClient  # noqa: E402
from claims_triage_agent.retriever import BM25Retriever  # noqa: E402
from claims_triage_agent.schema import AuditTrail  # noqa: E402

EVAL_DIR = Path(__file__).parent
CLAIMS_DIR = EVAL_DIR.parent / "src" / "claims_triage_agent" / "data" / "claims"
RESULTS_DIR = EVAL_DIR / "results"

LLM_CHOICES = ["openai", "baseline-approve", "baseline-structured", "fake-reference"]


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        cases: list[dict[str, Any]] = json.load(f)
    return cases


def _load_claim(claim_id: str) -> dict[str, Any]:
    with open(CLAIMS_DIR / f"{claim_id}.json", encoding="utf-8") as f:
        claim: dict[str, Any] = json.load(f)
    return claim


def _run_one(
    args: argparse.Namespace, case: dict[str, Any], retriever: BM25Retriever
) -> AuditTrail:
    claim = _load_claim(case["claim_id"])

    if args.llm in BASELINES:
        return BASELINES[args.llm](claim)

    if args.llm == "fake-reference":
        llm_client = FakeLLMClient(DEMO_SCRIPTS[case["claim_id"]])
        return ClaimsTriageAgent(llm_client, retriever).run(claim)

    if args.llm == "openai":
        from claims_triage_agent.llm_client import OpenAIChatCompletionsClient

        api_key = os.environ.get(args.api_key_env)
        llm_client = OpenAIChatCompletionsClient(
            model=args.model, api_key=api_key, base_url=args.base_url
        )
        return ClaimsTriageAgent(llm_client, retriever).run(claim)

    raise ValueError(f"Unknown --llm value: {args.llm}")


def _run_all_cases(
    args: argparse.Namespace, cases: list[dict[str, Any]], retriever: BM25Retriever
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for i, case in enumerate(cases):
        trail = _run_one(args, case, retriever)
        results.append(result_from_trail(case, trail))
        if args.sleep and i < len(cases) - 1:
            time.sleep(args.sleep)
    return results


def _default_out_path(args: argparse.Namespace) -> Path:
    if args.llm == "openai":
        model_slug = (args.model or "model").replace("/", "-")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return RESULTS_DIR / f"openai-{model_slug}-{stamp}"
    return RESULTS_DIR / args.llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", choices=LLM_CHOICES, default="fake-reference")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for --llm openai.")
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL (Gemini, Ollama, ...)."
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Env var holding the API key for --llm openai.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0, help="Seconds to sleep between cases."
    )
    parser.add_argument("--runs", type=int, default=1, help="Repeat the whole eval N times.")
    parser.add_argument(
        "--cases", default=str(EVAL_DIR / "eval_cases.json"), help="Path to the eval cases JSON."
    )
    parser.add_argument(
        "--out", default=None, help="Output path base (writes <out>.md and <out>.json)."
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    note = None
    if args.llm == "fake-reference":
        cases = [c for c in cases if c["claim_id"] in DEMO_SCRIPTS]
        note = (
            "**fake-reference** replays hand-written, known-correct trajectories for the "
            f"{len(cases)} original demo claims only. This validates that the eval harness "
            "itself works end to end -- it is NOT a measurement of any model's quality."
        )

    retriever = BM25Retriever()

    runs: list[list[CaseResult]] = []
    for _ in range(args.runs):
        runs.append(_run_all_cases(args, cases, retriever))

    results = runs[-1]
    summary = summarize(results, include_citation_metrics=args.llm not in BASELINES)
    consistency = summarize_consistency(runs) if args.runs > 1 else None

    baseline_summaries: dict[str, Summary] = {}
    for name, fn in BASELINES.items():
        if name == args.llm:
            continue  # already the primary backend; don't show it twice
        baseline_results = [
            result_from_trail(case, fn(_load_claim(case["claim_id"]))) for case in cases
        ]
        baseline_summaries[name] = summarize(baseline_results, include_citation_metrics=False)

    report_md = render_markdown_report(
        backend_label=args.llm if args.llm != "openai" else f"openai:{args.model}",
        results=results,
        summary=summary,
        baseline_summaries=baseline_summaries,
        consistency=consistency,
        note=note,
    )

    out_base = Path(args.out) if args.out else _default_out_path(args)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    md_path = out_base.with_suffix(".md")
    json_path = out_base.with_suffix(".json")

    md_path.write_text(report_md, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "llm": args.llm,
                "model": args.model if args.llm == "openai" else None,
                "runs": args.runs,
                "n_cases": len(cases),
                "summary": summary.to_dict(),
                "consistency": consistency.to_dict() if consistency else None,
                "baseline_summaries": {k: v.to_dict() for k, v in baseline_summaries.items()},
                "results": [r.to_dict() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report_md)
    print(f"\nWrote {md_path} and {json_path}")


if __name__ == "__main__":
    main()
