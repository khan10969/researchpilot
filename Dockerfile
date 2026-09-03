# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.9 \
    /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
        --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --extra web \
        --no-install-project

COPY src ./src
COPY apps ./apps
COPY .streamlit ./.streamlit

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --extra web \
        --no-editable

RUN mkdir -p \
    /app/data \
    /models/cache/huggingface \
    /models/cache/torch

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "researchpilot.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]