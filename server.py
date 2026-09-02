"""
FastAPI backend for the RAG app.

Wraps rag_engine.py (loading, chunking, embedding, retrieval, QA) behind
a small JSON API, and serves the static HTML/CSS/JS frontend.

Run with: uvicorn server:app --reload
Then open: http://localhost:8000
"""

import asyncio
import os
import tempfile
from typing import List, Optional

from dotenv import load_dotenv

# Load GROQ_API_KEY (and anything else) from a local .env file when running
# outside Docker (`uvicorn server:app --reload`). Under docker-compose the
# variable is already injected into the environment, so this is a harmless
# no-op there if no .env file is present in the image.
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import rag_engine as rag

app = FastAPI(title="RAG Q&A API")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Unexpected server error: {exc}"})

# ---------------------------------------------------------------------------
# In-memory app state (single shared session - fine for a local/single-user
# deployment; swap for a per-user store if you need multi-user support).
# ---------------------------------------------------------------------------
STATE = {
    "vectordb": None,
    "qa_chain_cache": {},  # {(api_key, sorted-tuple-of-source-names|None): RetrievalQA}
    "sources": [],  # list of {"name": str, "chunks": int}
    # Raw pandas DataFrames for CSV/Excel uploads, kept alongside the
    # chunked/embedded text so aggregation questions ("total revenue",
    # "average price by category") can be computed exactly instead of
    # guessed from retrieved text chunks. {filename: {sheet_name: DataFrame}}
    "dataframes": {},
}


def resolve_api_key() -> str:
    """
    Every request uses the single shared GROQ_API_KEY configured on the
    server (.env / docker-compose environment) — there is no per-user key
    in this deployment, so nobody has to sign up for or paste in their own.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise HTTPException(
            status_code=400,
            detail=(
                "No Groq API key configured on the server. Set GROQ_API_KEY "
                "in your .env file (copy .env.example) and restart the app."
            ),
        )
    return key


def flatten_dataframes(source_filter: Optional[List[str]] = None) -> dict:
    """Turns STATE['dataframes'] ({filename: {sheet: df}}) into a flat
    {readable_name: df} dict, scoped to source_filter if given (matching
    the same source names used for the semantic-search scope selector)."""
    flat = {}
    for filename, sheets in STATE["dataframes"].items():
        if source_filter and filename not in source_filter:
            continue
        for sheet_name, df in sheets.items():
            key = filename if sheet_name == "__default__" else f"{filename} — {sheet_name}"
            flat[key] = df
    return flat


def get_qa_chain(api_key: str, source_filter: Optional[List[str]] = None):
    """Build (or reuse a cached) QA chain scoped to source_filter (None =
    search every uploaded document)."""
    if STATE["vectordb"] is None:
        raise HTTPException(status_code=400, detail="No documents ingested yet.")

    if source_filter:
        known = {s["name"] for s in STATE["sources"]}
        unknown = [s for s in source_filter if s not in known]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Not filed in this session: {', '.join(unknown)}",
            )

    cache_key = (api_key, tuple(sorted(source_filter)) if source_filter else None)
    chain = STATE["qa_chain_cache"].get(cache_key)
    if chain is None:
        chain = rag.build_qa_chain(STATE["vectordb"], api_key, source_filter=source_filter)
        STATE["qa_chain_cache"][cache_key] = chain
    return chain


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class IngestUrlRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    question: str
    # Optional list of source names (filenames or URLs) to scope the
    # question to. Omit or leave empty to search across every document.
    sources: Optional[List[str]] = None


class SourceOut(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class IngestResponse(BaseModel):
    name: str
    chunks: int
    sources: list
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Thin/blocked-content detection
# ---------------------------------------------------------------------------
# Two real-world failure modes silently produce near-useless documents that
# still "succeed" at the ingest step, so the LLM has nothing to answer from
# later — showing up confusingly as "I don't know" with no explanation:
#   1. URLs: many modern sites (e-commerce category pages, app-shell SPAs,
#      anything behind a bot-check) render their real content with
#      client-side JS after the page loads, or actively block scrapers
#      outright. Our loader only fetches the initial server-rendered HTML.
#   2. PDFs: a scanned/image-based PDF (no embedded text layer, just page
#      images) yields little to no extractable text — this tool doesn't
#      run OCR.
# We detect both here and surface a clear warning instead of staying silent.
#   3. Local HTML files: a locally-uploaded .html file can be just as thin
#      as a scraped URL — e.g. a web app's UI shell (buttons, labels,
#      <template> tags) with almost no real prose once markup is stripped.
#      It "succeeds" at ingest the same way, so it needs the same warning.
# CSV/Excel/DOCX/TXT uploads are excluded from the length check: a small
# spreadsheet or short text file is legitimately thin, not a sign of a
# failed extraction — and truly-empty files already get rejected earlier
# by the "No extractable text found" check in _ingest_docs below.
MIN_USEFUL_CHARS = 400
_LENGTH_CHECKED_TYPES = ("url", "pdf", "html")
_BLOCK_MARKERS = (
    "enable javascript", "please enable cookies", "verify you are human",
    "are you a robot", "captcha", "access denied", "unusual traffic",
    "checking your browser", "just a moment",
)

# A genuine bot-check/CAPTCHA page IS the block message — the entire page
# is just "please verify you're human" or similar, so it's short. A real,
# full-length article that happens to mention one of these words in
# passing (e.g. a Wikipedia article on human computation that discusses
# CAPTCHAs as a topic) is not a block page and must not be flagged just
# because a marker string appears somewhere in 40,000 characters of real
# content. So the marker check only fires below this length — well above
# MIN_USEFUL_CHARS (which flags "too little text" on its own regardless
# of markers) but still far short of any normal article.
_BLOCK_MARKER_MAX_CHARS = 1500


def _detect_thin_or_blocked_content(docs, source_type: str) -> Optional[str]:
    combined = " ".join(d.page_content for d in docs).strip()
    lowered = combined.lower()

    if (
        source_type == "url"
        and len(combined) < _BLOCK_MARKER_MAX_CHARS
        and any(marker in lowered for marker in _BLOCK_MARKERS)
    ):
        return (
            "This page appears to be blocking automated access (a bot-check "
            "or CAPTCHA page was returned instead of the real content), so "
            "there may be little or nothing useful to answer questions from. "
            "Real challenge-based protection like this can't be fetched "
            "around — instead, open the page in your browser, save it "
            "(Ctrl/Cmd+P → Save as PDF, or Save Page As → Webpage HTML), "
            "and upload that file using the File tab instead."
        )

    if source_type in _LENGTH_CHECKED_TYPES and len(combined) < MIN_USEFUL_CHARS:
        if source_type == "pdf":
            return (
                "Very little text could be extracted from this PDF — it's "
                "likely a scanned or image-based PDF (no real text layer, "
                "just page images), which this tool can't OCR. Answers "
                "based on it may be incomplete or unavailable."
            )
        if source_type == "html":
            return (
                "Very little real text could be extracted from this HTML "
                "file — once tags are stripped it looks like mostly UI "
                "markup (buttons, labels, template blocks) rather than "
                "prose, so there may be little or nothing useful to answer "
                "questions from. If this was saved from a web app's page "
                "source, try uploading an article/blog-post HTML export "
                "instead, or use the file's actual content (e.g. a PDF/DOCX "
                "export) if one exists."
            )
        return (
            "Only a small amount of text could be extracted from this page "
            "(it likely loads its real content with JavaScript after the "
            "page loads — common on e-commerce/app-heavy sites like "
            "Flipkart or Amazon, which this tool can't execute). Answers "
            "based on it may be incomplete, inaccurate, or unavailable. "
            "For best results, use articles, blog posts, or documentation "
            "pages that render their content as plain HTML."
        )

    return None


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------

def _ingest_docs(docs, name: str, source_type: str = "file") -> IngestResponse:
    warning = _detect_thin_or_blocked_content(docs, source_type)

    chunks = rag.split_documents(docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in that document.")

    try:
        if STATE["vectordb"] is None:
            rag.reset_vector_store()
            STATE["vectordb"] = rag.build_vector_store(chunks)
        else:
            rag.add_to_vector_store(STATE["vectordb"], chunks)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to embed document (embedding model unreachable?): {e}",
        )

    # Invalidate cached chains so they pick up the newly added documents.
    STATE["qa_chain_cache"] = {}

    STATE["sources"].append({"name": name, "chunks": len(chunks)})
    return IngestResponse(name=name, chunks=len(chunks), sources=STATE["sources"], warning=warning)


@app.post("/api/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    source_type = rag.EXTENSION_TO_TYPE.get(ext)
    if source_type is None:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    tmp_path = None
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        docs = await asyncio.to_thread(rag.load_source, tmp_path, source_type)

        if source_type in ("csv", "xlsx", "xls"):
            try:
                STATE["dataframes"][file.filename] = await asyncio.to_thread(
                    rag.load_raw_dataframes, tmp_path, source_type
                )
            except Exception:
                # Aggregation support is a bonus on top of normal ingestion —
                # if raw parsing fails for some reason, still proceed with
                # the semantic-search path below rather than failing the
                # whole upload.
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Show the real filename in citations, not the temp path.
    for d in docs:
        d.metadata["source"] = file.filename

    # Chunking + embedding (rag.split_documents / build_vector_store /
    # add_to_vector_store inside _ingest_docs) is CPU-bound and can take a
    # while for large files. Running it directly in this async endpoint
    # would block FastAPI's single event loop — freezing every other
    # request (health checks, other users, other tabs) until it finishes.
    # asyncio.to_thread runs it in a worker thread instead, so the server
    # stays responsive while a big upload embeds in the background.
    return await asyncio.to_thread(_ingest_docs, docs, file.filename, source_type)


@app.post("/api/ingest/url", response_model=IngestResponse)
async def ingest_url(payload: IngestUrlRequest):
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")
    try:
        docs = await asyncio.to_thread(rag.load_source, url, "url")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    return await asyncio.to_thread(_ingest_docs, docs, url, "url")


# ---------------------------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------------------------

@app.post("/api/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    api_key = resolve_api_key()

    # Try the aggregation/computation path first (sum, average, count,
    # group-by, top-N, filters...) if there's any tabular data in scope.
    # This never raises — any failure inside just means "couldn't compute
    # it this way", and we fall through to the normal semantic-search
    # answer below instead of erroring out.
    dataframes = flatten_dataframes(payload.sources)
    if dataframes:
        try:
            tabular_result = await asyncio.to_thread(
                rag.answer_tabular_question, payload.question, dataframes, api_key
            )
        except Exception:
            tabular_result = None
        if tabular_result:
            plan = tabular_result["detail"]
            plan_bits = [f"operation={plan.get('operation')}"]
            if plan.get("column"):
                plan_bits.append(f"column={plan['column']}")
            if plan.get("group_by"):
                plan_bits.append(f"group_by={plan['group_by']}")
            if plan.get("filters"):
                plan_bits.append(f"filters={plan['filters']}")
            return AskResponse(
                answer=tabular_result["answer"],
                sources=[
                    SourceOut(
                        source=tabular_result["table"],
                        page=None,
                        snippet="Computed directly from the data (" + ", ".join(plan_bits) + ").",
                    )
                ],
            )

    qa_chain = get_qa_chain(api_key, source_filter=payload.sources)

    try:
        result = await asyncio.to_thread(rag.answer_question, qa_chain, payload.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error from LLM: {e}")

    sources = []
    for doc in result["sources"]:
        page = doc.metadata.get("page")
        sources.append(
            SourceOut(
                source=doc.metadata.get("source", "unknown"),
                page=(page + 1) if isinstance(page, int) else None,
                snippet=doc.page_content[:400],
            )
        )

    return AskResponse(answer=result["answer"], sources=sources)


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def config():
    """Lets the frontend know whether a shared server-side key is set, so
    it can skip asking the user for one."""
    return {"has_server_key": bool(os.environ.get("GROQ_API_KEY", "").strip())}


@app.get("/api/documents")
async def list_documents():
    return {"sources": STATE["sources"]}


@app.post("/api/clear")
async def clear_session():
    try:
        rag.reset_vector_store()
    finally:
        # Always clear in-memory state, even if the on-disk reset above hit
        # an unexpected error — otherwise a partial failure could leave old
        # documents/answers silently mixed in with whatever gets uploaded
        # next.
        STATE["vectordb"] = None
        STATE["qa_chain_cache"] = {}
        STATE["sources"] = []
        STATE["dataframes"] = {}
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(STATIC_DIR):
    # No static/ subfolder present — assume index.html / app.js / style.css
    # sit right next to server.py instead, so the app doesn't crash at
    # startup (StaticFiles raises if the directory doesn't exist).
    STATIC_DIR = BASE_DIR


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")