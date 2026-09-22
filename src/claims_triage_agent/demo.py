"""No-API-key demo ASGI entrypoint.

    uvicorn claims_triage_agent.demo:app --reload

Then open http://localhost:8000/docs and POST /adjudicate with one of the
4 synthetic claims in ``data/claims/`` (CLM-1001..CLM-1004).

This wires ``ReferenceScriptLLMClient`` -- which replays the hand-written,
known-correct tool-call trajectories in ``demo_scripts.DEMO_SCRIPTS`` --
instead of a real model, so the whole app (tool-calling loop, RAG,
reliability guardrails, audit trail) runs with no API key and no network
access. See the README's "Demo (no API key needed)" section for the
important caveat: this is a demo of the architecture, not a measurement of
any model's adjudication quality. For a real accuracy number against a
real model, use ``eval/run_eval.py --llm openai`` instead.
"""

from __future__ import annotations

from claims_triage_agent.api import create_app
from claims_triage_agent.demo_scripts import DEMO_SCRIPTS
from claims_triage_agent.llm_client import ReferenceScriptLLMClient

app = create_app(llm_client=ReferenceScriptLLMClient(DEMO_SCRIPTS))
