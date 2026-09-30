"""Scoring, failure-taxonomy classification, baselines and reporting for
the claims-triage eval harness.

``eval/run_eval.py`` is a thin CLI: it loads cases, drives
``ClaimsTriageAgent`` (or one of the deterministic baselines below) over
each one, and hands the results to this module. Everything that decides
what "correct" means, why a wrong decision was wrong, and how a run gets
summarized and rendered lives here instead, so it's typed and directly
unit-testable (``tests/test_evaluation.py``) without going through the
CLI or a real model.

Two things this module is deliberately honest about, because "mostly
right" is not good enough for a claims-adjudication system:

- **Accuracy alone is misleading.** Always approving every claim already
  scores ~50% on this eval set (15/30 cases are legitimately
  ``approved``). ``summarize()`` therefore also reports an *unsafe
  auto-decision rate* (cases that needed a human and didn't get one),
  a *wrongful approval rate*, an *over-flag rate*, and a full confusion
  matrix -- because in a claims context those failure modes have very
  different real-world costs, and a single accuracy number hides which
  one you're looking at.
- **A correct status is not the same as a correct reason.** A model can
  land on the right ``approved``/``denied`` without ever reading the
  clause that actually decides the case (see ``right_for_wrong_reason``
  below) -- that's a model that got lucky, not one that reasoned
  correctly, and it will fail differently next time.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from claims_triage_agent.schema import AuditTrail, Decision, DecisionStatus, utc_now_iso
from claims_triage_agent.tools import ToolError, lookup_policy

STATUSES: list[str] = [s.value for s in DecisionStatus]

# --- Failure taxonomy -------------------------------------------------
#
# Outcome: *what* went wrong, ordered worst-first by real-world cost. A
# case that should have been flagged for a human but wasn't is worse
# than a wrongful approval, which is worse than a wrongful denial, which
# is worse than flagging something that didn't need it.
OUTCOME_UNSAFE_AUTO_DECISION = "unsafe_auto_decision"
OUTCOME_WRONGFUL_APPROVAL = "wrongful_approval"
OUTCOME_WRONGFUL_DENIAL = "wrongful_denial"
OUTCOME_OVER_FLAGGING = "over_flagging"
OUTCOME_ORDER: list[str] = [
    OUTCOME_UNSAFE_AUTO_DECISION,
    OUTCOME_WRONGFUL_APPROVAL,
    OUTCOME_WRONGFUL_DENIAL,
    OUTCOME_OVER_FLAGGING,
]

# Cause: *why* it went wrong.
CAUSE_GUARDRAIL_OVERRIDE = "guardrail_override"
CAUSE_NEVER_SEARCHED_POLICY = "never_searched_policy"
CAUSE_RETRIEVAL_MISS = "retrieval_miss"
CAUSE_MISAPPLIED_CLAUSE = "misapplied_clause"
CAUSE_REASONING_WITHOUT_CLAUSE = "reasoning_without_clause"
CAUSE_ORDER: list[str] = [
    CAUSE_GUARDRAIL_OVERRIDE,
    CAUSE_NEVER_SEARCHED_POLICY,
    CAUSE_RETRIEVAL_MISS,
    CAUSE_MISAPPLIED_CLAUSE,
    CAUSE_REASONING_WITHOUT_CLAUSE,
]


def classify_outcome(expected_status: str, actual_status: str) -> str:
    """Classify *what* went wrong for one incorrect decision.

    Must only be called when ``expected_status != actual_status``; the
    caller (``result_from_trail``) already knows that.
    """
    if expected_status == actual_status:
        raise ValueError("classify_outcome called on a correct decision")
    if expected_status == DecisionStatus.FLAGGED_FOR_REVIEW.value:
        return OUTCOME_UNSAFE_AUTO_DECISION
    if actual_status == DecisionStatus.FLAGGED_FOR_REVIEW.value:
        return OUTCOME_OVER_FLAGGING
    if (
        expected_status == DecisionStatus.DENIED.value
        and actual_status == DecisionStatus.APPROVED.value
    ):
        return OUTCOME_WRONGFUL_APPROVAL
    return OUTCOME_WRONGFUL_DENIAL  # expected == approved, actual == denied


def classify_cause(
    *,
    requires_rag: bool,
    expected_chunk: str | None,
    searched_policy: bool,
    retrieved_chunk_ids: list[str],
    override_reason: str | None,
) -> str:
    """Classify *why* one incorrect decision happened."""
    if override_reason is not None:
        return CAUSE_GUARDRAIL_OVERRIDE
    if requires_rag and not searched_policy:
        return CAUSE_NEVER_SEARCHED_POLICY
    if requires_rag and expected_chunk is not None:
        if expected_chunk not in retrieved_chunk_ids:
            return CAUSE_RETRIEVAL_MISS
        return CAUSE_MISAPPLIED_CLAUSE
    return CAUSE_REASONING_WITHOUT_CLAUSE


@dataclass
class CaseResult:
    """The scored outcome of running one eval case through one backend."""

    claim_id: str
    category: str
    expected_status: str
    actual_status: str
    requires_rag: bool
    expected_cited_chunk: str | None
    cited_passage_ids: list[str]
    override_reason: str | None
    justification: str
    searched_policy: bool
    retrieved_chunk_ids: list[str]
    decision_correct: bool
    citation_correct: bool | None
    outcome: str | None
    cause: str | None
    right_for_wrong_reason: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "category": self.category,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "requires_rag": self.requires_rag,
            "expected_cited_chunk": self.expected_cited_chunk,
            "cited_passage_ids": self.cited_passage_ids,
            "override_reason": self.override_reason,
            "justification": self.justification,
            "searched_policy": self.searched_policy,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "decision_correct": self.decision_correct,
            "citation_correct": self.citation_correct,
            "outcome": self.outcome,
            "cause": self.cause,
            "right_for_wrong_reason": self.right_for_wrong_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseResult:
        """Reconstruct a ``CaseResult`` from ``to_dict()``'s output.

        Used by ``eval/run_eval.py`` to resume an interrupted run from
        its on-disk checkpoint without re-calling the model for cases
        already scored.
        """
        return cls(
            claim_id=data["claim_id"],
            category=data["category"],
            expected_status=data["expected_status"],
            actual_status=data["actual_status"],
            requires_rag=data["requires_rag"],
            expected_cited_chunk=data["expected_cited_chunk"],
            cited_passage_ids=list(data["cited_passage_ids"]),
            override_reason=data["override_reason"],
            justification=data["justification"],
            searched_policy=data["searched_policy"],
            retrieved_chunk_ids=list(data["retrieved_chunk_ids"]),
            decision_correct=data["decision_correct"],
            citation_correct=data["citation_correct"],
            outcome=data["outcome"],
            cause=data["cause"],
            right_for_wrong_reason=data["right_for_wrong_reason"],
        )


def result_from_trail(case: dict[str, Any], trail: AuditTrail) -> CaseResult:
    """Score one ``AuditTrail`` against its eval case's expected answer."""
    decision = trail.decision
    if decision is None:  # pragma: no cover - agent.run() always sets it
        raise ValueError(f"{case['claim_id']}: trail has no decision to score")

    search_calls = [tc for tc in trail.tool_calls if tc.tool_name == "search_policy_documents"]
    searched_policy = bool(search_calls)
    retrieved_chunk_ids = sorted({p.chunk_id for tc in search_calls for p in tc.cited_passages})

    expected_status = case["expected_status"]
    actual_status = decision.status.value
    decision_correct = actual_status == expected_status
    requires_rag = bool(case["requires_rag"])
    expected_chunk = case.get("expected_cited_chunk")

    citation_correct: bool | None = None
    if requires_rag and expected_chunk is not None:
        citation_correct = expected_chunk in decision.cited_passage_ids

    outcome: str | None = None
    cause: str | None = None
    if not decision_correct:
        outcome = classify_outcome(expected_status, actual_status)
        cause = classify_cause(
            requires_rag=requires_rag,
            expected_chunk=expected_chunk,
            searched_policy=searched_policy,
            retrieved_chunk_ids=retrieved_chunk_ids,
            override_reason=decision.override_reason,
        )

    right_for_wrong_reason = (
        decision_correct
        and requires_rag
        and expected_chunk is not None
        and expected_chunk not in decision.cited_passage_ids
    )

    return CaseResult(
        claim_id=case["claim_id"],
        category=case["category"],
        expected_status=expected_status,
        actual_status=actual_status,
        requires_rag=requires_rag,
        expected_cited_chunk=expected_chunk,
        cited_passage_ids=list(decision.cited_passage_ids),
        override_reason=decision.override_reason,
        justification=decision.justification,
        searched_policy=searched_policy,
        retrieved_chunk_ids=retrieved_chunk_ids,
        decision_correct=decision_correct,
        citation_correct=citation_correct,
        outcome=outcome,
        cause=cause,
        right_for_wrong_reason=right_for_wrong_reason,
    )


# --- Deterministic baselines --------------------------------------------
#
# Neither baseline calls a model, a tool, or the retriever -- they exist
# so every real accuracy number has something honest to be compared
# against. "Always approve" is the sanity floor (it scores ~50% on this
# eval set, which is the whole point: accuracy alone doesn't tell you
# whether a system is safe). "Structured-only" is what you get from the
# numeric coverage table with zero clause-reading at all.


def baseline_always_approve(claim: dict[str, Any]) -> AuditTrail:
    trail = AuditTrail(
        run_id="baseline-approve", claim_id=claim["claim_id"], started_at=utc_now_iso()
    )
    trail.decision = Decision(
        status=DecisionStatus.APPROVED,
        justification="Baseline: always approve, regardless of claim content.",
    )
    trail.finished_at = utc_now_iso()
    return trail


def baseline_structured_only(claim: dict[str, Any]) -> AuditTrail:
    """Approve iff the procedure code is on the covered-procedures table
    and does not require prior authorization; deny otherwise. Never reads
    a policy document, never checks prior claims, never calculates cost
    share -- this is exactly (and only) what ``lookup_policy`` knows.
    """
    trail = AuditTrail(
        run_id="baseline-structured", claim_id=claim["claim_id"], started_at=utc_now_iso()
    )
    try:
        policy = lookup_policy(claim["policy_id"])
    except ToolError as exc:
        decision = Decision(
            status=DecisionStatus.DENIED, justification=f"Baseline: {exc}"
        )
    else:
        code = claim["procedure_code"]
        covered = bool(policy["covered_procedures"].get(code, False))
        requires_auth = code in policy["requires_prior_auth"]
        if covered and not requires_auth:
            decision = Decision(
                status=DecisionStatus.APPROVED,
                justification=(
                    f"Baseline: CPT {code} is on {claim['policy_id']}'s covered-procedures "
                    "table and does not require prior authorization."
                ),
            )
        else:
            reason = "requires prior authorization" if requires_auth else "is not covered"
            decision = Decision(
                status=DecisionStatus.DENIED,
                justification=f"Baseline: CPT {code} {reason} per the structured table.",
            )
    trail.decision = decision
    trail.finished_at = utc_now_iso()
    return trail


BASELINES = {
    "baseline-approve": baseline_always_approve,
    "baseline-structured": baseline_structured_only,
}


# --- Summary metrics ------------------------------------------------------


@dataclass
class Summary:
    n: int
    accuracy: float
    unsafe_auto_decision_rate: float | None
    wrongful_approval_rate: float | None
    over_flag_rate: float | None
    confusion_matrix: dict[str, dict[str, int]]
    citation_accuracy: float | None
    right_for_wrong_reason_count: int | None
    per_category_accuracy: dict[str, float]
    outcome_counts: dict[str, int]
    cause_counts: dict[str, int]
    taxonomy: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "unsafe_auto_decision_rate": self.unsafe_auto_decision_rate,
            "wrongful_approval_rate": self.wrongful_approval_rate,
            "over_flag_rate": self.over_flag_rate,
            "confusion_matrix": self.confusion_matrix,
            "citation_accuracy": self.citation_accuracy,
            "right_for_wrong_reason_count": self.right_for_wrong_reason_count,
            "per_category_accuracy": self.per_category_accuracy,
            "outcome_counts": self.outcome_counts,
            "cause_counts": self.cause_counts,
            "taxonomy": self.taxonomy,
        }


def summarize(results: list[CaseResult], *, include_citation_metrics: bool = True) -> Summary:
    """Aggregate a list of per-case results into headline metrics.

    ``include_citation_metrics=False`` is used for the deterministic
    baselines, which never search or cite anything -- reporting a
    "citation accuracy" for them would be meaningless, not just zero.
    """
    n = len(results)
    if n == 0:
        raise ValueError("summarize() called with zero results")

    accuracy = sum(r.decision_correct for r in results) / n

    flagged_status = DecisionStatus.FLAGGED_FOR_REVIEW.value
    expected_flagged = [r for r in results if r.expected_status == flagged_status]
    unsafe = [r for r in expected_flagged if r.outcome == OUTCOME_UNSAFE_AUTO_DECISION]
    unsafe_auto_decision_rate = len(unsafe) / len(expected_flagged) if expected_flagged else None

    expected_denied = [r for r in results if r.expected_status == DecisionStatus.DENIED.value]
    wrongful_approvals = [r for r in expected_denied if r.outcome == OUTCOME_WRONGFUL_APPROVAL]
    wrongful_approval_rate = (
        len(wrongful_approvals) / len(expected_denied) if expected_denied else None
    )

    flaggable = [r for r in results if r.expected_status != flagged_status]
    over_flagged = [r for r in flaggable if r.outcome == OUTCOME_OVER_FLAGGING]
    over_flag_rate = len(over_flagged) / len(flaggable) if flaggable else None

    confusion: dict[str, dict[str, int]] = {e: dict.fromkeys(STATUSES, 0) for e in STATUSES}
    for r in results:
        confusion[r.expected_status][r.actual_status] += 1

    categories = sorted({r.category for r in results})
    per_category_accuracy = {
        cat: sum(r.decision_correct for r in results if r.category == cat)
        / sum(1 for r in results if r.category == cat)
        for cat in categories
    }

    outcome_counts: dict[str, int] = dict.fromkeys(OUTCOME_ORDER, 0)
    cause_counts: dict[str, int] = dict.fromkeys(CAUSE_ORDER, 0)
    taxonomy: dict[str, dict[str, int]] = {o: dict.fromkeys(CAUSE_ORDER, 0) for o in OUTCOME_ORDER}
    for r in results:
        if r.outcome is not None:
            outcome_counts[r.outcome] += 1
        if r.cause is not None:
            cause_counts[r.cause] += 1
        if r.outcome is not None and r.cause is not None:
            taxonomy[r.outcome][r.cause] += 1

    citation_accuracy: float | None = None
    right_for_wrong_reason_count: int | None = None
    if include_citation_metrics:
        citable = [r for r in results if r.citation_correct is not None]
        citation_accuracy = (
            sum(bool(r.citation_correct) for r in citable) / len(citable) if citable else None
        )
        right_for_wrong_reason_count = sum(r.right_for_wrong_reason for r in results)

    return Summary(
        n=n,
        accuracy=accuracy,
        unsafe_auto_decision_rate=unsafe_auto_decision_rate,
        wrongful_approval_rate=wrongful_approval_rate,
        over_flag_rate=over_flag_rate,
        confusion_matrix=confusion,
        citation_accuracy=citation_accuracy,
        right_for_wrong_reason_count=right_for_wrong_reason_count,
        per_category_accuracy=per_category_accuracy,
        outcome_counts=outcome_counts,
        cause_counts=cause_counts,
        taxonomy=taxonomy,
    )


# --- Multi-run consistency -------------------------------------------------


@dataclass
class ConsistencyReport:
    accuracy_mean: float
    accuracy_min: float
    accuracy_max: float
    per_run_accuracy: list[float]
    unstable_claim_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_mean": self.accuracy_mean,
            "accuracy_min": self.accuracy_min,
            "accuracy_max": self.accuracy_max,
            "per_run_accuracy": self.per_run_accuracy,
            "unstable_claim_ids": self.unstable_claim_ids,
        }


def find_unstable_cases(runs: list[list[CaseResult]]) -> list[str]:
    """Claim ids whose ``actual_status`` differed across at least two runs."""
    statuses_by_claim: dict[str, set[str]] = {}
    for run in runs:
        for r in run:
            statuses_by_claim.setdefault(r.claim_id, set()).add(r.actual_status)
    return sorted(claim_id for claim_id, statuses in statuses_by_claim.items() if len(statuses) > 1)


def summarize_consistency(runs: list[list[CaseResult]]) -> ConsistencyReport:
    if not runs:
        raise ValueError("summarize_consistency() called with zero runs")
    per_run_accuracy = [sum(r.decision_correct for r in run) / len(run) for run in runs]
    return ConsistencyReport(
        accuracy_mean=statistics.mean(per_run_accuracy),
        accuracy_min=min(per_run_accuracy),
        accuracy_max=max(per_run_accuracy),
        per_run_accuracy=per_run_accuracy,
        unstable_claim_ids=find_unstable_cases(runs),
    )


# --- Markdown report rendering ---------------------------------------------


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_int_or_na(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def render_markdown_report(
    *,
    backend_label: str,
    results: list[CaseResult],
    summary: Summary,
    baseline_summaries: dict[str, Summary],
    consistency: ConsistencyReport | None = None,
    note: str | None = None,
) -> str:
    lines: list[str] = [f"# Eval report: {backend_label}", ""]
    if note:
        lines += [note, ""]

    lines += ["## Headline metrics", ""]
    columns = [backend_label, *baseline_summaries.keys()]
    summaries = [summary, *baseline_summaries.values()]
    header = "| Metric | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    lines += [header, sep]
    lines.append("| Accuracy | " + " | ".join(_fmt_pct(s.accuracy) for s in summaries) + " |")
    lines.append(
        "| Unsafe auto-decision rate (of expected-flagged cases) | "
        + " | ".join(_fmt_pct(s.unsafe_auto_decision_rate) for s in summaries)
        + " |"
    )
    lines.append(
        "| Wrongful approval rate (of expected-denied cases) | "
        + " | ".join(_fmt_pct(s.wrongful_approval_rate) for s in summaries)
        + " |"
    )
    lines.append(
        "| Over-flag rate (of expected-decidable cases) | "
        + " | ".join(_fmt_pct(s.over_flag_rate) for s in summaries)
        + " |"
    )
    lines.append(
        "| Citation accuracy (clause-dependent cases) | "
        + " | ".join(_fmt_pct(s.citation_accuracy) for s in summaries)
        + " |"
    )
    lines.append(
        "| Right for the wrong reason (count) | "
        + " | ".join(_fmt_int_or_na(s.right_for_wrong_reason_count) for s in summaries)
        + " |"
    )
    lines.append("")

    if consistency is not None:
        unstable = ", ".join(consistency.unstable_claim_ids) or "none"
        lines += [
            "## Consistency across runs",
            "",
            f"- Runs: {len(consistency.per_run_accuracy)}",
            f"- Accuracy: mean {consistency.accuracy_mean:.1%}, "
            f"min {consistency.accuracy_min:.1%}, max {consistency.accuracy_max:.1%}",
            f"- Unstable cases (answer changed between runs): {unstable}",
            "",
        ]

    lines += ["## Confusion matrix (rows = expected, columns = actual)", ""]
    header = "| expected \\ actual | " + " | ".join(STATUSES) + " |"
    lines += [header, "|---" * (len(STATUSES) + 1) + "|"]
    for expected in STATUSES:
        row = summary.confusion_matrix[expected]
        lines.append(f"| {expected} | " + " | ".join(str(row[a]) for a in STATUSES) + " |")
    lines.append("")

    lines += ["## Failure taxonomy (outcome x cause)", ""]
    header = "| outcome \\ cause | " + " | ".join(CAUSE_ORDER) + " | total |"
    lines += [header, "|---" * (len(CAUSE_ORDER) + 2) + "|"]
    for outcome in OUTCOME_ORDER:
        row = summary.taxonomy[outcome]
        total = summary.outcome_counts[outcome]
        lines.append(
            f"| {outcome} | " + " | ".join(str(row[c]) for c in CAUSE_ORDER) + f" | {total} |"
        )
    lines.append("")

    lines += ["## Per-category accuracy", ""]
    lines += ["| Category | Accuracy |", "|---|---|"]
    for cat, acc in sorted(summary.per_category_accuracy.items()):
        lines.append(f"| {cat} | {_fmt_pct(acc)} |")
    lines.append("")

    misses = [r for r in results if not r.decision_correct]
    lines += [f"## Misses ({len(misses)}/{len(results)})", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        lines += [
            f"### {r.claim_id} ({r.category})",
            f"- Expected: `{r.expected_status}` -- Actual: `{r.actual_status}`",
            f"- Outcome: `{r.outcome}` -- Cause: `{r.cause}`",
            f"- Expected cited chunk: `{r.expected_cited_chunk}`",
            f"- Retrieved chunks: {r.retrieved_chunk_ids or 'none'}",
            f"- Cited chunks: {r.cited_passage_ids or 'none'}",
            f"- Override reason: `{r.override_reason}`",
            f"- Justification: {r.justification}",
            "",
        ]

    return "\n".join(lines)
