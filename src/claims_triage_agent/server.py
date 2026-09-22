"""Production ASGI entrypoint.

``api.create_app()`` takes the ``LLMClient`` as an explicit constructor
argument on purpose (see its docstring) so tests can inject
``FakeLLMClient`` with zero wiring. This module is the one place that
wires a *real* backend for an actual deployment:

    uvicorn claims_triage_agent.server:app --host 0.0.0.0 --port 8000

Model choice and API key come from the environment (``CLAIMS_AGENT_MODEL``,
``OPENAI_API_KEY``) rather than being hardcoded here.
"""

from __future__ import annotations

import os

from claims_triage_agent.api import create_app
from claims_triage_agent.llm_client import OpenAIChatCompletionsClient

app = create_app(
    llm_client=OpenAIChatCompletionsClient(
        model=os.environ.get("CLAIMS_AGENT_MODEL", "gpt-4o-mini")
    )
)
