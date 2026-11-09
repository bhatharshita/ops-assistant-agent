FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code changes don't invalidate this layer.
COPY pyproject.toml ./
COPY ops_assistant/__init__.py ops_assistant/__init__.py
RUN pip install --upgrade pip && pip install .

COPY ops_assistant ./ops_assistant
COPY evals ./evals
COPY scripts ./scripts
RUN pip install --no-deps .

RUN useradd --create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "ops_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
