"""
Best-effort pre-download of the local embedding model at Docker *build*
time, so the first real request after the container starts doesn't have
to wait for it.

This is purely an optimization, not a requirement: rag_engine.get_embeddings()
already lazily downloads/loads this exact model on first use at runtime.
So if this script can't reach Hugging Face during the build (rate limits,
a flaky network, HF having a bad day) it must NOT fail the whole image
build - it retries a few times with backoff, and if that still doesn't
work, prints a warning and exits 0 anyway. The image is still fully
functional either way; the only difference is whether the model is
already sitting in the image or gets fetched on first use.
"""
import sys
import time

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 20

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        HuggingFaceEmbeddings(model_name=MODEL_NAME)
        print(f"Embedding model '{MODEL_NAME}' baked into image.")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001 - deliberately broad, this step must never break the build
        print(
            f"[bake_embeddings] Attempt {attempt}/{MAX_ATTEMPTS} failed: {e}",
            file=sys.stderr,
        )
        if attempt < MAX_ATTEMPTS:
            print(f"[bake_embeddings] Retrying in {RETRY_DELAY_SECONDS}s...", file=sys.stderr)
            time.sleep(RETRY_DELAY_SECONDS)

print(
    "[bake_embeddings] WARNING: could not pre-download the embedding model "
    "at build time (Hugging Face may be rate-limiting this network right "
    "now). This is NOT a fatal error - the app will download the model "
    "automatically on its first real request instead. Continuing build.",
    file=sys.stderr,
)
sys.exit(0)