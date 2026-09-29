"""Tests for eval/run_eval.py's checkpoint and resume mechanics.

``eval/run_eval.py`` is a script, not a package module -- it manipulates
``sys.path`` itself at import time (see its own top-of-file comment) --
so it's loaded here directly by file path rather than via a normal
package import. Only the checkpoint/resume plumbing is covered here; the
scoring logic it delegates to (``result_from_trail``, ``summarize``, ...)
is already covered independently in ``tests/test_evaluation.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).parent.parent / "eval"
_spec = importlib.util.spec_from_file_location("run_eval", _EVAL_DIR / "run_eval.py")
assert _spec is not None and _spec.loader is not None
run_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = run_eval
_spec.loader.exec_module(run_eval)


# ---------------------------------------------------------------------------
# checkpoint save/load
# ---------------------------------------------------------------------------


def test_checkpoint_path_is_out_base_name_plus_partial_json(tmp_path: Path) -> None:
    out_base = tmp_path / "gemini-3.8-flash-sanity"
    assert run_eval._checkpoint_path(out_base) == tmp_path / "gemini-3.8-flash-sanity.partial.json"


def test_load_checkpoint_returns_none_when_missing(tmp_path: Path) -> None:
    assert run_eval._load_checkpoint(tmp_path / "missing.partial.json") is None


def test_save_and_load_checkpoint_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.json"
    data: dict[str, Any] = {
        "llm": "openai",
        "model": "gemini-3.8-flash",
        "runs_target": 3,
        "case_ids": ["CLM-1001", "CLM-1002"],
        "completed_runs": [],
    }

    run_eval._save_checkpoint(path, data)

    assert run_eval._load_checkpoint(path) == data


def test_save_checkpoint_leaves_no_leftover_tmp_file(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.json"
    run_eval._save_checkpoint(path, {"a": 1})
    assert not path.with_suffix(".json.tmp").exists()
    assert path.exists()


# ---------------------------------------------------------------------------
# _run_all_cases resume-by-skipping-done-cases
# ---------------------------------------------------------------------------

_CASES = [
    {
        "claim_id": "CLM-1001",
        "category": "cosmetic_exclusion",
        "expected_status": "denied",
        "requires_rag": True,
        "expected_cited_chunk": "POL-1001-chunk1",
    },
    {
        "claim_id": "CLM-1002",
        "category": "repeat_procedure",
        "expected_status": "flagged_for_review",
        "requires_rag": True,
        "expected_cited_chunk": "POL-1001-chunk3",
    },
]


def _args(**overrides: Any) -> argparse.Namespace:
    return argparse.Namespace(llm="baseline-approve", sleep=0.0, **overrides)


def test_run_all_cases_skips_cases_already_in_results() -> None:
    claim = run_eval._load_claim("CLM-1001")
    trail = run_eval.BASELINES["baseline-approve"](claim)
    already_done = run_eval.result_from_trail(_CASES[0], trail)

    results = [already_done]
    on_case_done_calls = 0

    def on_case_done() -> None:
        nonlocal on_case_done_calls
        on_case_done_calls += 1

    run_eval._run_all_cases(
        _args(),
        _CASES,
        run_eval.BM25Retriever(),
        results=results,
        on_case_done=on_case_done,
    )

    # CLM-1001 was already done and must not be re-run/duplicated; only
    # CLM-1002 is new, and on_case_done fires exactly once for it.
    assert [r.claim_id for r in results] == ["CLM-1001", "CLM-1002"]
    assert on_case_done_calls == 1


def test_run_all_cases_runs_everything_when_results_starts_empty() -> None:
    results: list[Any] = []
    run_eval._run_all_cases(
        _args(),
        _CASES,
        run_eval.BM25Retriever(),
        results=results,
        on_case_done=lambda: None,
    )
    assert [r.claim_id for r in results] == ["CLM-1001", "CLM-1002"]


def test_run_all_cases_propagates_quota_exceeded_keeping_prior_results(
    monkeypatch: Any,
) -> None:
    """A case that raises QuotaExceededError must not be caught inside
    _run_all_cases (the caller needs it to stop the whole eval cleanly),
    and results already appended before the failure must survive.
    """

    calls = {"n": 0}

    def _fake_run_one(args: Any, case: dict[str, Any], retriever: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            claim = run_eval._load_claim(case["claim_id"])
            return run_eval.BASELINES["baseline-approve"](claim)
        raise run_eval.QuotaExceededError("daily cap hit")

    monkeypatch.setattr(run_eval, "_run_one", _fake_run_one)

    results: list[Any] = []
    try:
        run_eval._run_all_cases(
            _args(),
            _CASES,
            run_eval.BM25Retriever(),
            results=results,
            on_case_done=lambda: None,
        )
        raise AssertionError("expected QuotaExceededError to propagate")
    except run_eval.QuotaExceededError:
        pass

    assert [r.claim_id for r in results] == ["CLM-1001"]
