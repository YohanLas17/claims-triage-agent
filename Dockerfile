# Minimal, single-stage image: this project has no compiled dependencies,
# so a build stage would add complexity without a real size/security win.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Synthetic demo data ships inside the package (src/claims_triage_agent/data/);
# no real patient data is ever baked into this image.

EXPOSE 8000

# CLAIMS_AGENT_MODEL and OPENAI_API_KEY are read at import time by
# server.py; pass them with `docker run -e OPENAI_API_KEY=... -e ...`.
CMD ["uvicorn", "claims_triage_agent.server:app", "--host", "0.0.0.0", "--port", "8000"]
