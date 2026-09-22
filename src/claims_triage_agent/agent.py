"""The claims-triage agent orchestrator: the tool-calling loop itself.

``ClaimsTriageAgent`` is intentionally the only piece of this codebase
that knows how to sequence an LLM, a set of tools and a retriever into a
decision. It depends only on the ``LLMClient`` and ``Retriever``
protocols (see ``llm_client.py`` / ``retriever.py``), never on a concrete
vendor SDK, which is what lets the whole loop -- including its
reliability guardrails -- be unit tested with ``FakeLLMClient`` and no
network access (``tests/test_agent.py``).

Two reliability guardrails are enforced here, deterministically, rather
than left to the model's judgement:

1. **Tool-call budget.** If the model has not called ``submit_decision``
   within ``max_tool_calls`` tool calls, the run is force-terminated as
   ``flagged_for_review`` instead of letting the loop run unbounded.
2. **Ungrounded RAG-dependent answers.** If any ``search_policy_documents``
   call in the run came back empty (no passage cleared the retriever's
   relevance threshold) and the model's final decision does not cite any
   retrieved passage, the decision is overridden to
   ``flagged_for_review``. A model should not be allowed to approve or
   deny a claim "from memory" when the free-text policy lookup it asked
   for came back empty.

Both overrides are recorded on the returned ``Decision`` via
``override_reason``, and both are exercised explicitly in
``tests/test_agent.py``.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from claims_triage_agent.llm_client import LLMClient, LLMResponse
from claims_triage_agent.retriever import Retriever
from claims_triage_agent.schema import (
    AuditTrail,
    Decision,
    DecisionStatus,
    RetrievedPassage,
    ToolCallRecord,
    utc_now_iso,
)
from claims_triage_agent.tools import (
    ToolError,
    calculate_coverage,
    check_prior_claims,
    lookup_policy,
)

DEFAULT_MAX_TOOL_CALLS = 8

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "lookup_policy",
        "description": (
            "Look up the structured coverage rules for a policy: deductible, "
            "coinsurance rate, out-of-pocket max, the flat covered-procedures "
            "table, and which procedure codes require prior authorization. "
            "Does NOT include free-text exclusions or conditional rules -- "
            "use search_policy_documents for those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string", "description": "e.g. 'POL-1001'"},
            },
            "required": ["policy_id"],
        },
    },
    {
        "name": "check_prior_claims",
        "description": (
            "Look up a patient's prior claims history, to check for "
            "duplicates or repeat procedures on the same body part/joint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {"type": "string", "description": "e.g. 'PAT-001'"},
            },
            "required": ["patient_id"],
        },
    },
    {
        "name": "calculate_coverage",
        "description": (
            "Compute the plan/patient cost split (deductible, coinsurance, "
            "out-of-pocket max) for a billed amount under a given policy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "procedure_code": {"type": "string"},
                "billed_amount": {"type": "number"},
            },
            "required": ["policy_id", "procedure_code", "billed_amount"],
        },
    },
    {
        "name": "search_policy_documents",
        "description": (
            "Full-text search over a policy's free-text document for "
            "exclusions, prior-authorization conditions and other clauses "
            "not present in the structured coverage table. Returns the "
            "top-k matching passages with their chunk ids; you MUST cite "
            "a chunk id in your final decision if you rely on one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "query": {
                    "type": "string",
                    "description": (
                        "A natural-language question, e.g. 'is skin tag "
                        "removal covered for cosmetic reasons'"
                    ),
                },
            },
            "required": ["policy_id", "query"],
        },
    },
    {
        "name": "submit_decision",
        "description": (
            "Terminate the run with a final, structured, justified "
            "decision. This must be the last tool call of the run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["approved", "denied", "flagged_for_review"],
                },
                "justification": {
                    "type": "string",
                    "description": "A concise, specific explanation of the decision.",
                },
                "cited_passage_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "chunk_id values returned by search_policy_documents "
                        "that this decision relies on, if any."
                    ),
                },
            },
            "required": ["status", "justification"],
        },
    },
]

SYSTEM_PROMPT = """You are a claims-triage assistant for a US health insurer. \
You are given one health-insurance claim and a set of tools. Gather every \
piece of information you need with tools before deciding -- never guess a \
policy's coverage rules, a patient's history, or a cost calculation. \
Structured tools (lookup_policy, check_prior_claims, calculate_coverage) \
cover the numeric rules; search_policy_documents covers free-text \
exclusions and conditions that are NOT in the structured data, and you \
must check it before approving or denying anything with a cosmetic, \
prior-authorization, network, or repeat-procedure angle. If you rely on a \
passage from search_policy_documents, you must include its chunk_id in \
cited_passage_ids. If you are not confident the claim can be safely \
approved or denied, submit status='flagged_for_review' rather than guessing. \
Finish every run by calling submit_decision exactly once."""


def _build_claim_message(claim: dict[str, Any]) -> str:
    return "Adjudicate this claim:\n" + json.dumps(claim, indent=2)


def apply_reliability_guardrails(
    status: DecisionStatus, cited_ids: list[str], had_empty_required_search: bool
) -> tuple[DecisionStatus, str | None]:
    """Apply the "don't let the model guess" override.

    Shared between ``ClaimsTriageAgent`` (the hand-rolled loop) and
    ``agent_langgraph.LangGraphClaimsTriageAgent`` (the LangGraph-based
    orchestrator) so the reliability rule is defined exactly once and both
    orchestrators are held to the same standard. See the module docstring
    for the full rationale.
    """
    if (
        had_empty_required_search
        and status != DecisionStatus.FLAGGED_FOR_REVIEW
        and not cited_ids
    ):
        return DecisionStatus.FLAGGED_FOR_REVIEW, "ungrounded_decision_after_empty_rag"
    return status, None


class ClaimsTriageAgent:
    """Runs the tool-calling loop for one claim and returns an audit trail."""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Retriever,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        self._llm_client = llm_client
        self._retriever = retriever
        self._max_tool_calls = max_tool_calls

    def _execute_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[Any, list[RetrievedPassage]]:
        """Run one non-terminal tool call. Returns (result, cited_passages)."""
        if name == "lookup_policy":
            return lookup_policy(**arguments), []
        if name == "check_prior_claims":
            return check_prior_claims(**arguments), []
        if name == "calculate_coverage":
            return calculate_coverage(**arguments), []
        if name == "search_policy_documents":
            passages = self._retriever.search(
                policy_id=arguments["policy_id"], query=arguments["query"], k=3
            )
            result: dict[str, Any] = {
                "passages": [
                    {
                        "chunk_id": p.chunk_id,
                        "text": p.text,
                        "score": p.score,
                    }
                    for p in passages
                ]
            }
            if not passages:
                result["note"] = "No relevant passages found for this query."
            return result, passages
        raise ToolError(f"Unknown tool '{name}'.")

    def run(self, claim: dict[str, Any]) -> AuditTrail:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        trail = AuditTrail(
            run_id=run_id, claim_id=claim["claim_id"], started_at=utc_now_iso()
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_claim_message(claim)},
        ]

        had_empty_required_search = False
        call_index = 0

        while True:
            if call_index >= self._max_tool_calls:
                decision = Decision(
                    status=DecisionStatus.FLAGGED_FOR_REVIEW,
                    justification=(
                        f"Automatically flagged: exceeded the maximum of "
                        f"{self._max_tool_calls} tool calls without reaching "
                        f"a decision."
                    ),
                    override_reason="max_tool_calls_exceeded",
                )
                trail.decision = decision
                trail.finished_at = utc_now_iso()
                return trail

            response: LLMResponse = self._llm_client.complete(messages, TOOL_SCHEMAS)

            if not response.tool_calls:
                # The model answered in free text instead of calling
                # submit_decision. We never accept an unstructured
                # decision -- flag for a human instead of parsing prose.
                decision = Decision(
                    status=DecisionStatus.FLAGGED_FOR_REVIEW,
                    justification=(
                        "Automatically flagged: the model produced a final "
                        "answer without calling submit_decision."
                    ),
                    override_reason="missing_submit_decision_call",
                )
                trail.decision = decision
                trail.finished_at = utc_now_iso()
                return trail

            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                }
            )

            for tool_call in response.tool_calls:
                if tool_call.name == "submit_decision":
                    args = tool_call.arguments
                    raw_status = DecisionStatus(args["status"])
                    cited_ids = list(args.get("cited_passage_ids", []) or [])

                    status, override_reason = apply_reliability_guardrails(
                        raw_status, cited_ids, had_empty_required_search
                    )

                    decision = Decision(
                        status=status,
                        justification=args["justification"],
                        cited_passage_ids=cited_ids,
                        override_reason=override_reason,
                    )
                    call_index += 1
                    trail.tool_calls.append(
                        ToolCallRecord(
                            call_index=call_index,
                            tool_name="submit_decision",
                            arguments=args,
                            result=decision.to_dict(),
                        )
                    )
                    trail.decision = decision
                    trail.finished_at = utc_now_iso()
                    return trail

                call_index += 1
                try:
                    result, cited_passages = self._execute_tool(
                        tool_call.name, tool_call.arguments
                    )
                    if tool_call.name == "search_policy_documents" and not cited_passages:
                        had_empty_required_search = True
                except ToolError as exc:
                    result, cited_passages = {"error": str(exc)}, []

                trail.tool_calls.append(
                    ToolCallRecord(
                        call_index=call_index,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        result=result,
                        cited_passages=cited_passages,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(result),
                    }
                )
