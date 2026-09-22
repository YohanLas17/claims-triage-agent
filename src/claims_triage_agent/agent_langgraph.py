"""A second orchestrator for the same agent, built on LangGraph.

``agent.py`` implements the tool-calling loop by hand, on purpose: it
keeps the core logic auditable and testable with zero third-party
dependencies. This module builds the *same* claims-triage agent again,
this time as an explicit LangGraph ``StateGraph`` -- nodes, conditional
edges, a checkpointer, and a real ``interrupt()`` call for
Human-in-the-Loop review -- to show the same system built with the
industry-standard 2026 agent-orchestration primitives, not just wrapped
in a one-line prebuilt agent constructor.

Both orchestrators wrap the exact same domain logic: ``tools.py``,
``retriever.py`` and the reliability guardrail
(``agent.apply_reliability_guardrails``) are reused unchanged here, not
reimplemented. Only the loop mechanics differ:

- ``agent.ClaimsTriageAgent``: a manual ``while`` loop against the
  ``LLMClient`` protocol.
- ``LangGraphClaimsTriageAgent`` (this module): a compiled LangGraph
  ``StateGraph`` against any LangChain ``BaseChatModel``, with the same
  tools bound via LangChain's ``@tool`` decorator.

Graph shape::

    START -> agent -> [tools -> agent (loop) | tools -> finalize]
                    -> no_decision
    finalize -> human_review -> END
    no_decision -> human_review -> END

``human_review`` is where the "don't let the model guess" guardrail
becomes a *real* pause: whenever the decision resolves to
``flagged_for_review`` (whether the model asked for it directly, or the
reliability override forced it), the graph calls ``interrupt()`` and
actually stops there, checkpointed by thread id, until
``resume_human_review(...)`` is called with a human's decision. This is
the same guardrail as in ``agent.py``, but here it is a literal
execution pause rather than just a label on the returned decision.

Written against the documented LangGraph 2026 ``interrupt()`` /
``Command(resume=...)`` pattern (dynamic interrupts inside a node, run
against a checkpointer). I do not have network access in the sandbox
this was written in, so unlike the rest of the codebase, this module and
its test file were reviewed by hand rather than actually executed --
see the "Testing status" section of the README before trusting it.
Everything it wraps (``tools.py``, ``retriever.py``,
``agent.apply_reliability_guardrails``) already has its own passing
tests independent of this file.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from claims_triage_agent.agent import (
    SYSTEM_PROMPT,
    _build_claim_message,
    apply_reliability_guardrails,
)
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

SUBMIT_DECISION_TOOL_NAME = "submit_decision_tool"
SEARCH_TOOL_NAME = "search_policy_documents_tool"

# LangGraph's recursion_limit counts graph super-steps (agent -> tools is
# 2 steps per round), not raw tool calls. This factor is a documented
# approximation of the same tool-call budget used by ClaimsTriageAgent,
# not an exact 1:1 mapping.
_RECURSION_STEPS_PER_TOOL_CALL = 2


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    decision: dict | None


def build_tools(retriever: Retriever) -> list:
    """Wrap the existing pure tool functions as LangChain ``@tool``s.

    The business logic lives in ``tools.py`` and ``retriever.py`` and is
    called as-is here; these wrappers only translate argument/return
    shapes into what a LangGraph ``ToolNode`` expects.
    """

    @tool
    def lookup_policy_tool(policy_id: str) -> dict:
        """Look up the structured coverage rules for a policy: deductible,
        coinsurance rate, out-of-pocket max, the flat covered-procedures
        table, and which procedure codes require prior authorization.
        Does NOT include free-text exclusions -- use
        search_policy_documents_tool for those."""
        try:
            return lookup_policy(policy_id)
        except ToolError as exc:
            return {"error": str(exc)}

    @tool
    def check_prior_claims_tool(patient_id: str) -> dict:
        """Look up a patient's prior claims history, to check for
        duplicates or repeat procedures on the same body part/joint."""
        try:
            return check_prior_claims(patient_id)
        except ToolError as exc:
            return {"error": str(exc)}

    @tool
    def calculate_coverage_tool(
        policy_id: str, procedure_code: str, billed_amount: float
    ) -> dict:
        """Compute the plan/patient cost split (deductible, coinsurance,
        out-of-pocket max) for a billed amount under a given policy."""
        try:
            return calculate_coverage(policy_id, procedure_code, billed_amount)
        except ToolError as exc:
            return {"error": str(exc)}

    @tool
    def search_policy_documents_tool(policy_id: str, query: str) -> dict:
        """Full-text search over a policy's free-text document for
        exclusions, prior-authorization conditions and other clauses not
        present in the structured coverage table. Returns the top-k
        matching passages with their chunk ids; you MUST cite a chunk id
        in your final decision if you rely on one."""
        passages = retriever.search(policy_id=policy_id, query=query, k=3)
        if not passages:
            return {"passages": [], "note": "No relevant passages found for this query."}
        return {
            "passages": [
                {"chunk_id": p.chunk_id, "text": p.text, "score": p.score} for p in passages
            ]
        }

    @tool
    def submit_decision_tool(
        status: str, justification: str, cited_passage_ids: list[str] | None = None
    ) -> str:
        """Terminate the run with a final, structured, justified decision.
        status must be one of 'approved', 'denied', 'flagged_for_review'.
        This must be the last tool call of the run."""
        return "Decision recorded."

    return [
        lookup_policy_tool,
        check_prior_claims_tool,
        calculate_coverage_tool,
        search_policy_documents_tool,
        submit_decision_tool,
    ]


def _parse_tool_content(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _extract_cited_passages(tool_name: str, result: Any) -> list[RetrievedPassage]:
    if tool_name != SEARCH_TOOL_NAME or not isinstance(result, dict):
        return []
    return [
        RetrievedPassage(
            policy_id="",
            document_id=p.get("chunk_id", "").rsplit("-chunk", 1)[0],
            chunk_id=p.get("chunk_id", ""),
            text=p.get("text", ""),
            score=p.get("score", 0.0),
        )
        for p in result.get("passages", [])
    ]


def _build_audit_trail_from_messages(
    run_id: str, claim_id: str, started_at: str, messages: list
) -> AuditTrail:
    """Reconstruct an ``AuditTrail`` from a graph run's accumulated message list.

    Note on timestamps: LangGraph's message state doesn't carry
    per-call wall-clock times, so every ``ToolCallRecord`` here is
    stamped at reconstruction time rather than at the moment each call
    actually happened. A production deployment that needs true per-call
    timestamps would attach a LangChain callback handler
    (``on_tool_start``/``on_tool_end``) instead; this is flagged rather
    than faked.
    """
    trail = AuditTrail(run_id=run_id, claim_id=claim_id, started_at=started_at)
    pending_calls: dict[str, dict[str, Any]] = {}
    call_index = 0

    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                call_id = tc["id"]
                if call_id is not None:
                    pending_calls[call_id] = {"name": tc["name"], "arguments": tc["args"]}
        elif isinstance(message, ToolMessage):
            call_index += 1
            info = pending_calls.pop(
                message.tool_call_id, {"name": message.name, "arguments": {}}
            )
            result = _parse_tool_content(message.content)
            trail.tool_calls.append(
                ToolCallRecord(
                    call_index=call_index,
                    tool_name=info["name"],
                    arguments=info["arguments"],
                    result=result,
                    timestamp=utc_now_iso(),
                    cited_passages=_extract_cited_passages(info["name"], result),
                )
            )

    return trail


def _last_ai_tool_call(messages: list, name: str) -> dict | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                if tc["name"] == name:
                    return tc["args"]
            return None  # the most recent AIMessage didn't call this tool
    return None


def _had_empty_required_search(messages: list) -> bool:
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == SEARCH_TOOL_NAME:
            result = _parse_tool_content(message.content)
            if isinstance(result, dict) and not result.get("passages"):
                return True
    return False


def route_after_agent(state: AgentState) -> Literal["tools", "no_decision"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "no_decision"


def route_after_tools(state: AgentState) -> Literal["agent", "finalize"]:
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            names = [tc["name"] for tc in message.tool_calls]
            return "finalize" if SUBMIT_DECISION_TOOL_NAME in names else "agent"
    return "agent"  # pragma: no cover - defensive, tools node always follows an AIMessage


def finalize(state: AgentState) -> dict:
    """Build the Decision from the submit_decision_tool call, applying the
    same reliability guardrail as ClaimsTriageAgent."""
    submit_args = _last_ai_tool_call(state["messages"], SUBMIT_DECISION_TOOL_NAME)
    had_empty_search = _had_empty_required_search(state["messages"])
    # route_after_tools only sends us here when the last AIMessage's tool
    # calls included submit_decision_tool, so this is always populated.
    assert submit_args is not None, "finalize reached without a submit_decision_tool call"

    raw_status = DecisionStatus(submit_args["status"])
    cited_ids = list(submit_args.get("cited_passage_ids") or [])
    status, override_reason = apply_reliability_guardrails(raw_status, cited_ids, had_empty_search)

    decision = Decision(
        status=status,
        justification=submit_args["justification"],
        cited_passage_ids=cited_ids,
        override_reason=override_reason,
    )
    return {"decision": decision.to_dict()}


def no_decision(state: AgentState) -> dict:
    decision = Decision(
        status=DecisionStatus.FLAGGED_FOR_REVIEW,
        justification=(
            "Automatically flagged: the model produced a final answer "
            "without calling submit_decision_tool."
        ),
        override_reason="missing_submit_decision_call",
    )
    return {"decision": decision.to_dict()}


def human_review(state: AgentState) -> dict:
    """Where the reliability guardrail becomes a real pause.

    If the decision is not flagged, there is nothing for a human to do --
    this node is a no-op and the graph proceeds straight to END. If it
    IS flagged, this calls ``interrupt()``: the graph genuinely stops
    executing here (checkpointed by thread id) until
    ``LangGraphClaimsTriageAgent.resume_human_review`` sends a
    ``Command(resume=...)`` with the human's decision.
    """
    decision = state["decision"]
    # human_review is only reached from finalize or no_decision, both of
    # which always set "decision" before routing here.
    assert decision is not None, "human_review reached without a decision in state"
    if decision["status"] != DecisionStatus.FLAGGED_FOR_REVIEW.value:
        return {}

    human_input = interrupt(
        {
            "reason": decision.get("override_reason") or "model_flagged_for_review",
            "decision": decision,
        }
    )
    if not human_input:
        return {}

    merged = dict(decision)
    merged["status"] = human_input.get("final_status", decision["status"])
    notes = human_input.get("reviewer_notes")
    if notes:
        merged["justification"] = f"{merged['justification']} [Human review: {notes}]"
    return {"decision": merged}


class LangGraphClaimsTriageAgent:
    """Runs the LangGraph StateGraph-based claims-triage agent for one claim.

    ``model`` is any LangChain ``BaseChatModel`` (e.g.
    ``ChatOpenAI(model="gpt-4o-mini")`` in production, or
    ``FakeToolCallingModel`` from ``tests/test_agent_langgraph.py`` for
    deterministic, offline tests -- mirroring how ``ClaimsTriageAgent``
    is tested against ``FakeLLMClient``).

    Every ``run()`` call gets its own thread id (the run id), checkpointed
    in-memory, so a flagged decision can be resumed later with
    ``resume_human_review()`` -- including from a different process, if
    the checkpointer is swapped for a persistent one (e.g.
    ``langgraph.checkpoint.postgres.PostgresSaver``) in production.
    """

    def __init__(
        self,
        model: Any,
        retriever: Retriever,
        max_tool_calls: int = 8,
    ) -> None:
        tools = build_tools(retriever)
        self._model_with_tools = model.bind_tools(tools)
        self._max_tool_calls = max_tool_calls
        self._recursion_limit = max_tool_calls * _RECURSION_STEPS_PER_TOOL_CALL + 4

        builder: StateGraph = StateGraph(AgentState)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("finalize", finalize)
        builder.add_node("no_decision", no_decision)
        builder.add_node("human_review", human_review)

        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", route_after_agent, {"tools": "tools", "no_decision": "no_decision"}
        )
        builder.add_conditional_edges(
            "tools", route_after_tools, {"agent": "agent", "finalize": "finalize"}
        )
        builder.add_edge("finalize", "human_review")
        builder.add_edge("no_decision", "human_review")
        builder.add_edge("human_review", END)

        self._checkpointer = InMemorySaver()
        self._graph = builder.compile(checkpointer=self._checkpointer)

    def _agent_node(self, state: AgentState) -> dict:
        response = self._model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def _decision_from_state(self, state: dict) -> Decision:
        decision_dict = state.get("decision")
        if decision_dict is None:  # pragma: no cover - defensive, a node always sets it
            return Decision(
                status=DecisionStatus.FLAGGED_FOR_REVIEW,
                justification="Automatically flagged: run ended without a recorded decision.",
                override_reason="missing_decision_state",
            )
        return Decision(
            status=DecisionStatus(decision_dict["status"]),
            justification=decision_dict["justification"],
            cited_passage_ids=list(decision_dict.get("cited_passage_ids") or []),
            override_reason=decision_dict.get("override_reason"),
        )

    def run(self, claim: dict[str, Any]) -> AuditTrail:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        started_at = utc_now_iso()
        config: RunnableConfig = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": self._recursion_limit,
        }
        initial_state = {
            "messages": [("system", SYSTEM_PROMPT), ("user", _build_claim_message(claim))],
            "decision": None,
        }

        try:
            final_state = self._graph.invoke(initial_state, config=config)
        except GraphRecursionError:
            decision = Decision(
                status=DecisionStatus.FLAGGED_FOR_REVIEW,
                justification=(
                    f"Automatically flagged: exceeded the tool-call budget "
                    f"(~{self._max_tool_calls} calls) without reaching a decision."
                ),
                override_reason="max_tool_calls_exceeded",
            )
            trail = AuditTrail(run_id=run_id, claim_id=claim["claim_id"], started_at=started_at)
            trail.decision = decision
            trail.finished_at = utc_now_iso()
            return trail

        trail = _build_audit_trail_from_messages(
            run_id, claim["claim_id"], started_at, final_state["messages"]
        )
        trail.decision = self._decision_from_state(final_state)
        trail.finished_at = utc_now_iso()
        # Runs that paused at human_review's interrupt() are still
        # reported with their (flagged) decision so callers get a usable
        # result immediately; resume_human_review() with this same run_id
        # continues the same checkpointed thread if a human acts on it.
        return trail

    def resume_human_review(
        self, run_id: str, final_status: str, reviewer_notes: str = ""
    ) -> Decision:
        """Continue a run paused at the ``human_review`` interrupt.

        ``run_id`` must be the ``run_id`` (thread id) of a prior ``run()``
        call whose decision was ``flagged_for_review``. Returns the
        human-confirmed (or human-overridden) ``Decision``.
        """
        config: RunnableConfig = {"configurable": {"thread_id": run_id}}
        final_state = self._graph.invoke(
            Command(resume={"final_status": final_status, "reviewer_notes": reviewer_notes}),
            config=config,
        )
        return self._decision_from_state(final_state)
