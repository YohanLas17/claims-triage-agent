import json
from pathlib import Path

import pytest

from claims_triage_agent.agent import ClaimsTriageAgent
from claims_triage_agent.demo_scripts import DEMO_SCRIPTS
from claims_triage_agent.evaluation import (
    CAUSE_GUARDRAIL_OVERRIDE,
    CAUSE_MISAPPLIED_CLAUSE,
    CAUSE_NEVER_SEARCHED_POLICY,
    CAUSE_REASONING_WITHOUT_CLAUSE,
    CAUSE_RETRIEVAL_MISS,
    OUTCOME_OVER_FLAGGING,
    OUTCOME_UNSAFE_AUTO_DECISION,
    OUTCOME_WRONGFUL_APPROVAL,
    OUTCOME_WRONGFUL_DENIAL,
    CaseResult,
    baseline_structured_only,
    classify_cause,
    classify_outcome,
    find_unstable_cases,
    result_from_trail,
    summarize,
    summarize_consistency,
)
from claims_triage_agent.llm_client import FakeLLMClient
from claims_triage_agent.retriever import BM25Retriever
from claims_triage_agent.schema import AuditTrail, Decision, DecisionStatus, utc_now_iso

CLAIMS_DIR = Path(__file__).parent.parent / "src" / "claims_triage_agent" / "data" / "claims"
EVAL_CASES_PATH = Path(__file__).parent.parent / "eval" / "eval_cases.json"

with open(EVAL_CASES_PATH, encoding="utf-8") as f:
    EVAL_CASES = json.load(f)


def _load_claim(claim_id: str) -> dict:
    with open(CLAIMS_DIR / f"{claim_id}.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# eval_cases.json integrity
# ---------------------------------------------------------------------------


def test_every_case_has_a_claim_file_and_a_note():
    assert len(EVAL_CASES) == 30
    for case in EVAL_CASES:
        claim_path = CLAIMS_DIR / f"{case['claim_id']}.json"
        assert claim_path.exists(), f"missing claim file for {case['claim_id']}"
        assert case["note"].strip(), f"{case['claim_id']} has an empty note"
        assert case["category"], f"{case['claim_id']} has no category"


# ---------------------------------------------------------------------------
# outcome / cause classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "actual", "outcome"),
    [
        ("flagged_for_review", "approved", OUTCOME_UNSAFE_AUTO_DECISION),
        ("flagged_for_review", "denied", OUTCOME_UNSAFE_AUTO_DECISION),
        ("denied", "approved", OUTCOME_WRONGFUL_APPROVAL),
        ("approved", "denied", OUTCOME_WRONGFUL_DENIAL),
        ("approved", "flagged_for_review", OUTCOME_OVER_FLAGGING),
        ("denied", "flagged_for_review", OUTCOME_OVER_FLAGGING),
    ],
)
def test_classify_outcome(expected, actual, outcome):
    assert classify_outcome(expected, actual) == outcome


def test_classify_outcome_raises_on_correct_decision():
    with pytest.raises(ValueError):
        classify_outcome("approved", "approved")


def test_classify_cause_guardrail_override_takes_priority():
    cause = classify_cause(
        requires_rag=True,
        expected_chunk="POL-1001-chunk1",
        searched_policy=True,
        retrieved_chunk_ids=["POL-1001-chunk1"],
        override_reason="ungrounded_decision_after_empty_rag",
    )
    assert cause == CAUSE_GUARDRAIL_OVERRIDE


def test_classify_cause_never_searched_policy():
    cause = classify_cause(
        requires_rag=True,
        expected_chunk="POL-1001-chunk1",
        searched_policy=False,
        retrieved_chunk_ids=[],
        override_reason=None,
    )
    assert cause == CAUSE_NEVER_SEARCHED_POLICY


def test_classify_cause_retrieval_miss():
    cause = classify_cause(
        requires_rag=True,
        expected_chunk="POL-1001-chunk1",
        searched_policy=True,
        retrieved_chunk_ids=["POL-1001-chunk3"],
        override_reason=None,
    )
    assert cause == CAUSE_RETRIEVAL_MISS


def test_classify_cause_misapplied_clause():
    cause = classify_cause(
        requires_rag=True,
        expected_chunk="POL-1001-chunk1",
        searched_policy=True,
        retrieved_chunk_ids=["POL-1001-chunk1"],
        override_reason=None,
    )
    assert cause == CAUSE_MISAPPLIED_CLAUSE


def test_classify_cause_reasoning_without_clause_when_rag_not_required():
    cause = classify_cause(
        requires_rag=False,
        expected_chunk=None,
        searched_policy=False,
        retrieved_chunk_ids=[],
        override_reason=None,
    )
    assert cause == CAUSE_REASONING_WITHOUT_CLAUSE


# ---------------------------------------------------------------------------
# result_from_trail
# ---------------------------------------------------------------------------


def test_result_from_trail_on_the_clm_1001_demo_script():
    case = next(c for c in EVAL_CASES if c["claim_id"] == "CLM-1001")
    claim = _load_claim("CLM-1001")
    agent = ClaimsTriageAgent(FakeLLMClient(DEMO_SCRIPTS["CLM-1001"]), BM25Retriever())
    trail = agent.run(claim)

    result = result_from_trail(case, trail)

    assert result.claim_id == "CLM-1001"
    assert result.expected_status == "denied"
    assert result.actual_status == "denied"
    assert result.decision_correct is True
    assert result.searched_policy is True
    assert "POL-1001-chunk1" in result.retrieved_chunk_ids
    assert result.citation_correct is True
    assert result.outcome is None
    assert result.cause is None
    assert result.right_for_wrong_reason is False


def test_result_from_trail_flags_right_for_wrong_reason():
    case = {
        "claim_id": "CLM-TEST",
        "category": "test",
        "expected_status": "denied",
        "requires_rag": True,
        "expected_cited_chunk": "POL-1001-chunk1",
        "note": "test",
    }
    trail = AuditTrail(run_id="run-test", claim_id="CLM-TEST", started_at=utc_now_iso())
    trail.decision = Decision(
        status=DecisionStatus.DENIED, justification="Denied.", cited_passage_ids=[]
    )
    trail.finished_at = utc_now_iso()

    result = result_from_trail(case, trail)

    assert result.decision_correct is True
    assert result.citation_correct is False
    assert result.right_for_wrong_reason is True


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


def _make_result(
    claim_id: str,
    category: str,
    expected_status: str,
    actual_status: str,
    *,
    requires_rag: bool = False,
    expected_cited_chunk: str | None = None,
    cited_passage_ids: list[str] | None = None,
    override_reason: str | None = None,
    searched_policy: bool = False,
    retrieved_chunk_ids: list[str] | None = None,
) -> CaseResult:
    case = {
        "claim_id": claim_id,
        "category": category,
        "expected_status": expected_status,
        "requires_rag": requires_rag,
        "expected_cited_chunk": expected_cited_chunk,
    }
    trail = AuditTrail(run_id=f"run-{claim_id}", claim_id=claim_id, started_at=utc_now_iso())
    trail.decision = Decision(
        status=DecisionStatus(actual_status),
        justification="test",
        cited_passage_ids=cited_passage_ids or [],
        override_reason=override_reason,
    )
    trail.finished_at = utc_now_iso()
    if searched_policy:
        from claims_triage_agent.schema import RetrievedPassage, ToolCallRecord

        passages = [
            RetrievedPassage(
                policy_id="POL-1001",
                document_id="POL-1001",
                chunk_id=cid,
                text="x",
                score=1.0,
            )
            for cid in (retrieved_chunk_ids or [])
        ]
        trail.tool_calls.append(
            ToolCallRecord(
                call_index=1,
                tool_name="search_policy_documents",
                arguments={},
                result={},
                cited_passages=passages,
            )
        )
    return result_from_trail(case, trail)


def test_summarize_on_a_small_hand_made_list():
    results = [
        _make_result("A", "cat1", "approved", "approved"),
        _make_result("B", "cat1", "denied", "denied"),
        _make_result(
            "C",
            "cat2",
            "flagged_for_review",
            "approved",  # unsafe auto-decision
        ),
        _make_result("D", "cat2", "denied", "approved"),  # wrongful approval
        _make_result("E", "cat2", "approved", "flagged_for_review"),  # over-flagging
    ]

    summary = summarize(results)

    assert summary.n == 5
    assert summary.accuracy == pytest.approx(2 / 5)
    assert summary.unsafe_auto_decision_rate == pytest.approx(1 / 1)  # 1 expected-flagged case
    assert summary.wrongful_approval_rate == pytest.approx(1 / 2)  # 2 expected-denied cases
    assert summary.over_flag_rate == pytest.approx(1 / 4)  # 4 non-flagged-expected cases
    assert summary.confusion_matrix["denied"]["approved"] == 1
    assert summary.per_category_accuracy["cat1"] == pytest.approx(1.0)
    assert summary.per_category_accuracy["cat2"] == pytest.approx(0.0)
    assert summary.outcome_counts[OUTCOME_UNSAFE_AUTO_DECISION] == 1
    assert summary.outcome_counts[OUTCOME_WRONGFUL_APPROVAL] == 1
    assert summary.outcome_counts[OUTCOME_OVER_FLAGGING] == 1


def test_summarize_excludes_citation_metrics_for_baselines():
    results = [_make_result("A", "cat1", "approved", "approved")]
    summary = summarize(results, include_citation_metrics=False)
    assert summary.citation_accuracy is None
    assert summary.right_for_wrong_reason_count is None


def test_summarize_citation_accuracy_only_over_clause_dependent_cases():
    results = [
        _make_result(
            "A",
            "cat1",
            "denied",
            "denied",
            requires_rag=True,
            expected_cited_chunk="POL-1001-chunk1",
            cited_passage_ids=["POL-1001-chunk1"],
        ),
        _make_result(
            "B",
            "cat1",
            "denied",
            "denied",
            requires_rag=True,
            expected_cited_chunk="POL-1001-chunk1",
            cited_passage_ids=[],
        ),
        _make_result("C", "cat1", "approved", "approved"),  # not clause-dependent
    ]
    summary = summarize(results)
    assert summary.citation_accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# baseline_structured_only
# ---------------------------------------------------------------------------


def test_structured_baseline_misses_clm_1001():
    """CLM-1001's correct answer (denied) only comes from the free-text
    exclusion clause; the structured baseline has no access to it and
    will wrongly approve, proving the eval set actually needs RAG.
    """
    claim = _load_claim("CLM-1001")
    trail = baseline_structured_only(claim)
    assert trail.decision.status == DecisionStatus.APPROVED  # wrong: expected 'denied'


def test_structured_baseline_denies_uncovered_code():
    claim = _load_claim("CLM-1004")
    trail = baseline_structured_only(claim)
    assert trail.decision.status == DecisionStatus.DENIED  # correct


# ---------------------------------------------------------------------------
# multi-run consistency
# ---------------------------------------------------------------------------


def test_find_unstable_cases_detects_a_changed_answer():
    run1 = [
        _make_result("A", "cat", "approved", "approved"),
        _make_result("B", "cat", "denied", "denied"),
    ]
    run2 = [
        _make_result("A", "cat", "approved", "approved"),
        _make_result("B", "cat", "denied", "flagged_for_review"),
    ]

    assert find_unstable_cases([run1, run2]) == ["B"]


def test_summarize_consistency_reports_mean_min_max():
    run1 = [
        _make_result("A", "cat", "approved", "approved"),
        _make_result("B", "cat", "denied", "denied"),
    ]
    run2 = [
        _make_result("A", "cat", "approved", "denied"),
        _make_result("B", "cat", "denied", "denied"),
    ]

    report = summarize_consistency([run1, run2])

    assert report.per_run_accuracy == [1.0, 0.5]
    assert report.accuracy_mean == pytest.approx(0.75)
    assert report.accuracy_min == pytest.approx(0.5)
    assert report.accuracy_max == pytest.approx(1.0)
    assert report.unstable_claim_ids == ["A"]
