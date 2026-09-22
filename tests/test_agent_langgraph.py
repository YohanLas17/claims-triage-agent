"""Tests for the LangGraph StateGraph-based orchestrator.

These mirror tests/test_agent.py case for case, against
LangGraphClaimsTriageAgent instead of ClaimsTriageAgent, using
FakeToolCallingModel (defined below) in place of FakeLLMClient. Both
fakes exist for the same reason: a scripted, deterministic stand-in for
the LLM that needs no network access or API key. One additional test
(``test_human_review_resume_can_overturn_a_flagged_decision``) exercises
the Human-in-the-Loop interrupt/resume path that has no equivalent in
the hand-rolled agent.

NOTE (see README "Testing status"): this file was written and reviewed
by hand but not executed, since langgraph/langchain-core were not
installable in the sandbox this was authored in (no network access).
Run `pytest tests/test_agent_langgraph.py -v` after
`pip install -r requirements.txt` to confirm before relying on it.
"""

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from claims_triage_agent.agent_langgraph import LangGraphClaimsTriageAgent
from claims_triage_agent.retriever import BM25Retriever
from claims_triage_agent.schema import DecisionStatus

CLAIMS_DIR = Path(__file__).parent.parent / "src" / "claims_triage_agent" / "data" / "claims"


def _load_claim(claim_id: str) -> dict:
    with open(CLAIMS_DIR / f"{claim_id}.json", encoding="utf-8") as f:
        return json.load(f)


class FakeToolCallingModel(BaseChatModel):
    """A scripted BaseChatModel: replays a fixed list of AIMessages in
    order, one per call, regardless of which tools were bound. This is
    the LangChain-side equivalent of llm_client.FakeLLMClient -- it lets
    LangGraphClaimsTriageAgent be tested deterministically, with no
    network access or API key.

    Note: the ``human_review`` node re-invokes each node function on
    resume, but the ``agent`` node (and therefore this model) is never
    re-invoked once a run has reached ``human_review`` -- so the script
    only needs one entry per actual agent turn, same as for
    ClaimsTriageAgent's FakeLLMClient.
    """

    responses: list[AIMessage]
    _index: int = PrivateAttr(default=0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._index >= len(self.responses):
            raise AssertionError(
                f"FakeToolCallingModel script exhausted after {self._index} calls."
            )
        response = self.responses[self._index]
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        # The script already encodes which tools are "called" and with
        # what arguments, so binding is a no-op beyond returning self.
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-model"


def _tool_call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}


def test_agent_denies_cosmetic_removal_using_only_the_free_text_clause():
    claim = _load_claim("CLM-1001")
    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("lookup_policy_tool", {"policy_id": "POL-1001"})],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "search_policy_documents_tool",
                        {"policy_id": "POL-1001", "query": "is cosmetic skin tag removal covered"},
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "submit_decision_tool",
                        {
                            "status": "denied",
                            "justification": "Cosmetic-only removal is excluded per Section 4.",
                            "cited_passage_ids": ["POL-1001-chunk1"],
                        },
                    )
                ],
            ),
        ]
    )
    agent = LangGraphClaimsTriageAgent(model, BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.DENIED
    assert trail.decision.cited_passage_ids == ["POL-1001-chunk1"]
    search_calls = [tc for tc in trail.tool_calls if tc.tool_name == "search_policy_documents_tool"]
    assert len(search_calls) == 1
    assert search_calls[0].cited_passages


def test_agent_forces_flagged_for_review_when_rag_is_empty_and_uncited():
    claim = _load_claim("CLM-1001")
    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "search_policy_documents_tool",
                        {"policy_id": "POL-1001", "query": "zzzznonexistentqueryterm"},
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "submit_decision_tool",
                        {
                            "status": "approved",
                            "justification": "Looks fine.",
                            "cited_passage_ids": [],
                        },
                    )
                ],
            ),
        ]
    )
    agent = LangGraphClaimsTriageAgent(model, BM25Retriever())
    trail = agent.run(claim)

    # The override forces flagged_for_review, which means this run also
    # paused at the human_review interrupt() -- see the resume test below.
    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW
    assert trail.decision.override_reason == "ungrounded_decision_after_empty_rag"


def test_agent_flags_when_model_never_calls_submit_decision():
    claim = _load_claim("CLM-1001")
    model = FakeToolCallingModel(
        responses=[AIMessage(content="I think this claim looks fine overall.", tool_calls=[])]
    )
    agent = LangGraphClaimsTriageAgent(model, BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW
    assert trail.decision.override_reason == "missing_submit_decision_call"


def test_agent_forces_flagged_for_review_when_tool_call_budget_is_exceeded():
    claim = _load_claim("CLM-1001")
    looping_responses = [
        AIMessage(
            content="",
            tool_calls=[_tool_call("lookup_policy_tool", {"policy_id": "POL-1001"})],
        )
        for _ in range(50)
    ]
    model = FakeToolCallingModel(responses=looping_responses)
    agent = LangGraphClaimsTriageAgent(model, BM25Retriever(), max_tool_calls=3)
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW
    assert trail.decision.override_reason == "max_tool_calls_exceeded"


def test_human_review_resume_can_overturn_a_flagged_decision():
    """Exercises the interrupt()/Command(resume=...) round trip: a run
    that gets auto-flagged (empty RAG, uncited) pauses at human_review,
    and a human reviewer can then approve it -- proving the pause is a
    real execution stop, not just a status label, and that resuming
    continues the *same* checkpointed thread (by run_id)."""
    claim = _load_claim("CLM-1001")
    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "search_policy_documents_tool",
                        {"policy_id": "POL-1001", "query": "zzzznonexistentqueryterm"},
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "submit_decision_tool",
                        {
                            "status": "approved",
                            "justification": "Looks fine.",
                            "cited_passage_ids": [],
                        },
                    )
                ],
            ),
        ]
    )
    agent = LangGraphClaimsTriageAgent(model, BM25Retriever())
    trail = agent.run(claim)
    assert trail.decision.status == DecisionStatus.FLAGGED_FOR_REVIEW

    human_decision = agent.resume_human_review(
        trail.run_id,
        final_status="approved",
        reviewer_notes="Reviewed the chart manually, approving after all.",
    )

    assert human_decision.status == DecisionStatus.APPROVED
    assert "Human review" in human_decision.justification


def test_human_review_is_a_no_op_when_decision_is_not_flagged():
    """A clean approve/deny should reach END without ever calling
    interrupt() -- resume_human_review should not be needed at all."""
    claim = _load_claim("CLM-1004")
    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("lookup_policy_tool", {"policy_id": "POL-2002"})],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "submit_decision_tool",
                        {
                            "status": "denied",
                            "justification": (
                                "CPT 27447 is not on this plan's covered-procedures list."
                            ),
                            "cited_passage_ids": [],
                        },
                    )
                ],
            ),
        ]
    )
    agent = LangGraphClaimsTriageAgent(model, BM25Retriever())
    trail = agent.run(claim)

    assert trail.decision.status == DecisionStatus.DENIED
    assert trail.decision.override_reason is None
