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

# Install a CPU-only PyTorch build FIRST, from PyTorch's own CPU wheel
# index, before requirements.txt pulls in sentence-transformers (which
# depends on torch). A plain `pip install torch` on Linux grabs the
# CUDA-enabled build by default — several GB of NVIDIA CUDA runtime
# libraries this app never touches, since embeddings run on CPU here.
# Installing the CPU wheel first satisfies sentence-transformers'
# dependency without ever triggering that much bigger download.
#
# The --mount=type=cache line caches pip's download cache across builds
# (needs Docker BuildKit, which is the default in modern Docker/Docker
# Desktop) — it doesn't shrink the very first build, but every build
# after that re-uses already-downloaded wheels instead of re-fetching
# them, which is where Docker build time usually hurts most day-to-day.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Safety net: if requirements.txt is ever bumped to a newer Playwright
# version than what this base image's tag has baked in, this makes sure
# the matching browser binary still gets installed. Harmless no-op when
# versions already match (the browser is already present).
RUN playwright install chromium --with-deps

# Bake the local embedding model into the image at build time instead
# of downloading it from HuggingFace on the container's first real
# request — trades a slightly longer build for a much faster first use
# after the container starts.
#
# This is a best-effort optimization only — see bake_embeddings.py.
# Hugging Face occasionally rate-limits anonymous downloads (HTTP 429),
# especially across repeated builds on the same network. If that happens
# here, this step retries a few times and then simply skips baking the
# model in rather than failing the whole image build — the app still
# works fine either way, since get_embeddings() in rag_engine.py
# downloads the same model lazily on first real use at runtime if it
# isn't already cached in the image.
COPY bake_embeddings.py .
RUN python bake_embeddings.py

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]