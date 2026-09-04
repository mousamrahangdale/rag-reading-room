# syntax=docker/dockerfile:1.4
#
# Microsoft's official Playwright image already has Chromium (and its
# many OS-level dependencies) pre-installed, matched to a specific
# Playwright version — this avoids the usual headache of installing
# a headless browser's system dependencies (fonts, codecs, GTK libs,
# etc.) by hand in a plain python:slim image.
#
# IMPORTANT: keep the tag's version (v1.45.0 below) in sync with the
# `playwright` version pinned in requirements.txt. A mismatch between
# the installed Python package and the pre-baked browser binaries is
# the most common cause of Playwright breaking inside Docker.
FROM mcr.microsoft.com/playwright/python:v1.45.0-jammy

WORKDIR /app

# Copy just the manifest first so Docker can cache this layer — the
# (slow) dependency install only reruns when requirements.txt changes,
# not on every code edit.
COPY requirements.txt .

# No local torch / sentence-transformers install anymore — embeddings
# run through Hugging Face's hosted Inference API instead (see
# rag_engine.get_embeddings()), specifically so this image, and the
# running container's memory footprint, stay small enough for low-RAM
# free-tier hosts (e.g. Render's free 512MB web service). This is also
# why bake_embeddings.py is gone: there's no local model to pre-download.
#
# The --mount=type=cache line caches pip's download cache across builds
# (needs Docker BuildKit, which is the default in modern Docker/Docker
# Desktop) — it doesn't shrink the very first build, but every build
# after that re-uses already-downloaded wheels instead of re-fetching
# them, which is where Docker build time usually hurts most day-to-day.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Safety net: if requirements.txt is ever bumped to a newer Playwright
# version than what this base image's tag has baked in, this makes sure
# the matching browser binary still gets installed. Harmless no-op when
# versions already match (the browser is already present).
RUN playwright install chromium --with-deps

COPY . .

EXPOSE 8000

# Render (and most PaaS hosts) assign the port dynamically via $PORT and
# route traffic to it — a hardcoded 8000 means the platform's health
# check can never reach the app, even though it's running fine. Falling
# back to 8000 keeps `docker run`/docker-compose (no $PORT set) working
# unchanged for local use. Shell form (not exec-array form) is required
# here so $PORT actually gets expanded at container start.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}