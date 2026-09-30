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
from collections.abc import Callable
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
from claims_triage_agent.llm_client import FakeLLMClient, QuotaExceededError  # noqa: E402
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
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    retriever: BM25Retriever,
    *,
    results: list[CaseResult],
    on_case_done: Callable[[], None],
) -> None:
    """Run every case in ``cases`` not already scored in ``results``.

    ``results`` is mutated in place (appended to) rather than returned,
    so a caller that already has some entries in it (resuming a run
    that was interrupted partway through) only pays for the remaining
    cases. ``on_case_done`` is called after each new result is appended,
    so the caller can checkpoint progress to disk immediately -- if a
    later case raises (e.g. ``QuotaExceededError``), everything up to
    that point is already saved.
    """
    already_done = {r.claim_id for r in results}
    pending = [c for c in cases if c["claim_id"] not in already_done]
    for i, case in enumerate(pending):
        trail = _run_one(args, case, retriever)
        results.append(result_from_trail(case, trail))
        on_case_done()
        if args.sleep and i < len(pending) - 1:
            time.sleep(args.sleep)


def _checkpoint_path(out_base: Path) -> Path:
    return out_base.with_name(out_base.name + ".partial.json")


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        result: dict[str, Any] = json.load(f)
        return result


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    """Write atomically (write-then-rename) so a crash mid-write can't
    leave a truncated, unreadable checkpoint behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run instead of starting over -- e.g. after a "
            "provider's daily quota cut a run short. Requires --out to name the "
            "same base path as the interrupted run; progress lives in "
            "<out>.partial.json and cases already scored there are not re-run."
        ),
    )
    args = parser.parse_args()

    if args.resume and not args.out:
        parser.error("--resume requires --out (the checkpoint lives at <out>.partial.json)")

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

    out_base = Path(args.out) if args.out else _default_out_path(args)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _checkpoint_path(out_base)
    case_ids = [c["claim_id"] for c in cases]
    model = args.model if args.llm == "openai" else None

    if args.resume:
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint is None:
            parser.error(f"--resume given but no checkpoint found at {checkpoint_path}")
        assert checkpoint is not None  # parser.error() above always exits
        if checkpoint["case_ids"] != case_ids:
            parser.error(
                f"--resume checkpoint at {checkpoint_path} was built from a different "
                "--cases file; pass the same --cases used originally."
            )
        if checkpoint["llm"] != args.llm or checkpoint["model"] != model:
            parser.error(
                f"--resume checkpoint at {checkpoint_path} was recorded for "
                f"llm={checkpoint['llm']!r} model={checkpoint['model']!r}, not "
                f"llm={args.llm!r} model={model!r}."
            )
        if checkpoint["runs_target"] != args.runs:
            print(
                f"Note: resuming with the checkpoint's original --runs="
                f"{checkpoint['runs_target']} (ignoring --runs={args.runs})."
            )
            args.runs = checkpoint["runs_target"]
    else:
        checkpoint = {
            "llm": args.llm,
            "model": model,
            "runs_target": args.runs,
            "case_ids": case_ids,
            "completed_runs": [],
        }
        _save_checkpoint(checkpoint_path, checkpoint)

    runs: list[list[CaseResult]] = [
        [CaseResult.from_dict(d) for d in run] for run in checkpoint["completed_runs"]
    ]
    in_progress: list[CaseResult] = (
        runs.pop() if runs and len(runs[-1]) < len(cases) else []
    )

    def _persist() -> None:
        all_runs = [*runs, in_progress] if in_progress else runs
        checkpoint["completed_runs"] = [[r.to_dict() for r in run] for run in all_runs]
        _save_checkpoint(checkpoint_path, checkpoint)

    try:
        while len(runs) < args.runs:
            _run_all_cases(
                args, cases, retriever, results=in_progress, on_case_done=_persist
            )
            runs.append(in_progress)
            in_progress = []
            _persist()
    except QuotaExceededError as exc:
        done = sum(len(r) for r in runs) + len(in_progress)
        target = args.runs * len(cases)
        print(f"\nStopping: provider daily quota exhausted ({exc})")
        print(f"Completed {done}/{target} case-runs; {target - done} remaining.")
        print(f"Progress saved to {checkpoint_path}.")
        print(
            "Resume once quota resets with the same command plus "
            f"--resume --out {out_base}"
        )
        sys.exit(1)

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
