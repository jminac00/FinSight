# syntax=docker/dockerfile:1

# Build stage: resolve and install dependencies into a self-contained virtualenv.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies are installed before the source is copied so this layer stays
# cached across code changes. Torch alone accounts for most of the build time.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# Runtime stage: only the virtualenv and the application source.
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system finsight \
    && useradd --system --gid finsight --create-home finsight

WORKDIR /app

COPY --from=builder --chown=finsight:finsight /app/.venv /app/.venv
COPY --chown=finsight:finsight app ./app

# Mirrors the repository layout: app.services.deep_learning.artifacts resolves
# the model directory as <parent of the app package>/ml_models.
RUN mkdir -p /app/ml_models && chown finsight:finsight /app/ml_models

USER finsight

EXPOSE 8000

# Caddy terminates TLS and forwards the client address in X-Forwarded-For.
# Without proxy headers every request would carry the reverse proxy's address
# and the per-IP rate limiter would throttle all clients as one.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
