# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: install Orion and its dependencies into a virtualenv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# The server extra is included so `orion serve` works out of the box.
RUN pip install --upgrade pip && pip install ".[server]"

# ---------------------------------------------------------------------------
# Runtime stage: minimal image running as an unprivileged user.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    ORION_DATA_DIR=/home/orion/.orion \
    ORION_APPROVAL=ask

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash orion

COPY --from=builder /opt/venv /opt/venv

USER orion
WORKDIR /workspace

# `orion` reads GEMINI_API_KEY from the environment or a mounted .env file.
ENTRYPOINT ["orion"]
CMD ["--help"]
