"""
Core RAG engine.

Handles:
- Loading documents from PDF, TXT, CSV, Excel, DOCX, HTML, and URLs
- Splitting into chunks
- Embedding (local, free - HuggingFace sentence-transformers)
- Storing/retrieving from a Chroma vector store
- Answering questions with an open-source LLM served via Groq's API
"""

import asyncio
import gc
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import uuid
from typing import List, Optional

# Identify as an ordinary desktop browser rather than a labeled scraper.
# This clears *soft* bot checks (some sites gate content behind a plain
# "please enable JavaScript"/user-agent sniff), but it can't and won't
# defeat real challenge-based protection (Cloudflare/CAPTCHA) — those
# require executing JavaScript, which this loader deliberately doesn't do.
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)

# Chroma's anonymous telemetry (posthog) throws a harmless but noisy
# "capture() takes 1 positional argument but 3 were given" error on some
# posthog/chromadb version combos. It doesn't affect functionality, but
# disabling it keeps the logs clean.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    BSHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Local, free embedding model (runs on CPU, no API key needed)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Open-source model served by Groq. This is Groq's flagship open-weight
# production model as of writing — OpenAI's gpt-oss-120b (117B-parameter
# MoE, strong reasoning/agentic quality, 131K context). Swap for any other
# model Groq hosts: https://console.groq.com/docs/models
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

CHROMA_ROOT = os.environ.get(
    "CHROMA_DIR", os.path.join(tempfile.gettempdir(), "rag_chroma_db")
)
# The actual directory Chroma is currently writing to. Starts as
# CHROMA_ROOT itself; reset_vector_store() below may switch it to a fresh
# sub-directory (see the note there about why).
_current_chroma_dir = CHROMA_ROOT

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# How many spreadsheet rows get batched into a single Document before
# embedding. One-row-per-document was the main reason large CSV/Excel
# files were slow to ingest (thousands of tiny embedding calls); batching
# them cuts the number of chunks embedded by roughly this factor. 20 rows
# of typical "col: value" text usually stays comfortably under CHUNK_SIZE,
# and split_documents() will still break up any batch that doesn't.
ROWS_PER_CHUNK = 20

# Hard cap on how many rows of any single table get run through embedding
# for semantic search. Batching (above) cuts the number of *chunks*, but a
# million-row file still means a million rows of text passing through the
# local embedding model — that's the actual source of multi-minute (or
# longer) ingest times, independent of chunk size. Beyond this cap we
# embed an evenly-spaced sample instead of every row.
#
# This only affects semantic/text search over row content. It does NOT
# affect correctness of sums/averages/counts/group-bys — those run in
# answer_tabular_question() against the full, unsampled DataFrame (see
# load_raw_dataframes / server.py's STATE["dataframes"]), so "what's the
# total revenue" is exact regardless of this cap.
MAX_EMBEDDED_ROWS = 5000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> List[Document]:
    """
    Reads a plain-text file, tolerating files that aren't UTF-8. Real-world
    .txt files are often Windows-1252/cp1252 (Notepad exports), or UTF-16
    with a BOM — the previous UTF-8-only loader would crash with
    UnicodeDecodeError on those. Tries a short list of common encodings in
    order, then falls back to lossy-but-non-crashing decoding as a last
    resort so an upload never fails outright over encoding alone.
    """
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return TextLoader(file_path, encoding=encoding).load()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(file_path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": file_path})]


def _ensure_windows_proactor_policy() -> None:
    """
    Playwright's sync API launches the browser as a subprocess. On
    Windows, only `ProactorEventLoop` supports asyncio subprocess
    creation — `SelectorEventLoop` raises `NotImplementedError` on
    `subprocess_exec`, because Windows never implemented that path for
    it. Uvicorn's own worker threads (which is where `asyncio.to_thread`
    runs this, see server.py's ingest endpoints) can end up on a
    Selector-based loop for uvicorn's own reasons, which breaks
    Playwright even though uvicorn's own networking is unaffected either
    way. Setting the process-wide policy here only changes what event
    loop class gets created for loops made AFTER this call — i.e. the
    fresh one Playwright is about to create for this worker thread — so
    it does not disturb uvicorn's already-running main loop or anything
    else already in flight.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def _load_url_rendered(url: str, timeout_ms: int = 20000) -> List[Document]:
    """
    Fetches a URL with a real headless browser (Playwright + Chromium)
    instead of a bare HTTP request, so pages that render their actual
    content with JavaScript after the initial load — most modern SPAs,
    e-commerce category pages, app-shell sites — have something real to
    extract. A plain `requests`-based fetch only ever sees the pre-JS
    HTML shell for these, which is why they used to come back empty.

    This does NOT and will not attempt to defeat genuine anti-bot
    challenges (Cloudflare Turnstile, hCaptcha, real CAPTCHAs) — those
    are specifically designed to detect and block automated browsers,
    headless or not, and this function makes no attempt to evade that
    detection. Pages protected that way will still come back as a
    challenge page, which _detect_thin_or_blocked_content (server.py)
    catches and reports rather than silently failing.
    """
    from playwright.sync_api import sync_playwright

    _ensure_windows_proactor_policy()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=os.environ.get("USER_AGENT"))
            # Images/fonts/media never contribute extractable text, so
            # skip loading them — this meaningfully speeds up the render
            # without changing what ends up in the page's text content.
            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in ("image", "media", "font")
                else route.continue_(),
            )
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            html = page.content()
        finally:
            browser.close()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return [Document(page_content=text, metadata={"source": url})]


def load_url(url: str) -> List[Document]:
    """
    Tries the headless-browser fetch above first. Falls back to the
    plain HTTP `WebBaseLoader` if Playwright/Chromium isn't installed
    (ImportError) or if the render fails for some other reason (timeout,
    navigation error, browser crash) — so ingestion never hard-fails just
    because the richer path hit a snag; it degrades to what this app did
    before instead.
    """
    try:
        return _load_url_rendered(url)
    except ImportError:
        return WebBaseLoader(url).load()
    except Exception:
        try:
            return WebBaseLoader(url).load()
        except Exception:
            raise


def _sample_row_positions(total_rows: int, max_rows: int) -> List[int]:
    """Evenly-spaced 0-based row positions covering the full table, e.g.
    for 1,000,000 rows capped at 5,000 this picks every ~200th row rather
    than just the first 5,000 — so the sample still reflects the whole
    file (early, middle, and late rows) instead of only its head."""
    if total_rows <= max_rows:
        return list(range(total_rows))
    step = total_rows / max_rows
    return sorted({int(i * step) for i in range(max_rows)})


def _dataframe_to_documents(
    df: "pd.DataFrame",
    source_path: str,
    sheet_name: Optional[str] = None,
    rows_per_chunk: int = ROWS_PER_CHUNK,
    max_embedded_rows: int = MAX_EMBEDDED_ROWS,
) -> List[Document]:
    """
    Shared by the CSV and Excel loaders below. Produces:
      - one summary Document per table (column names + row count), so
        aggregate questions ("how many rows?", "what columns does this
        have?") have something relevant to retrieve — a per-row chunk
        alone can't answer those.
      - one Document per BATCH of `rows_per_chunk` rows (not one per row),
        formatted as readable "column: value" lines separated by blank
        lines. One-row-per-document used to mean a 5,000-row CSV produced
        5,000 separate chunks to embed, which is what made large
        spreadsheets so slow to ingest — batching cuts that by ~20x while
        RecursiveCharacterTextSplitter (see split_documents) still splits
        any individual batch further if it comes out longer than
        CHUNK_SIZE, so no row's text gets truncated or lost.

    Tables larger than `max_embedded_rows` are down-sampled to an
    evenly-spaced subset before batching/embedding — see MAX_EMBEDDED_ROWS
    for why. Row labels in the resulting chunk metadata always reflect the
    row's real 1-based position in the original file, even when sampled,
    so citations still point somewhere meaningful.
    """
    if df.empty:
        return []

    df = df.fillna("")  # avoid literal "nan" showing up in the text
    label = f"sheet '{sheet_name}'" if sheet_name else "this CSV file"
    columns = ", ".join(str(c) for c in df.columns)
    total_rows = len(df)
    summary_text = f"Summary of {label}: {total_rows} rows, {len(df.columns)} columns. Columns: {columns}."

    sample_positions = _sample_row_positions(total_rows, max_embedded_rows)
    sampled = len(sample_positions) < total_rows
    if sampled:
        summary_text += (
            f" Note: this table is large, so only an evenly-spaced sample "
            f"of {len(sample_positions)} of its {total_rows} rows is "
            "indexed for text search below. Exact totals, averages, "
            "counts, and other calculations still use every row, not "
            "just the sample."
        )

    base_meta = {"source": source_path}
    if sheet_name:
        base_meta["sheet"] = sheet_name

    docs = [Document(page_content=summary_text, metadata={**base_meta, "row": "summary"})]

    prefix = f"[Sheet: {sheet_name}]\n" if sheet_name else ""
    columns_list = list(df.columns)
    # itertuples(name=None) yields plain tuples and is dramatically faster
    # than iterrows() (which builds a full pandas Series per row) — this
    # was the remaining bottleneck for large files even after batching
    # rows into fewer chunks and capping row count, since it's O(sample
    # size) rather than O(n) once the sample above has already been taken.
    work_df = df.iloc[sample_positions] if sampled else df
    all_rows = list(work_df.itertuples(index=False, name=None))
    n = len(all_rows)
    for start in range(0, n, rows_per_chunk):
        batch = all_rows[start : start + rows_per_chunk]
        batch_positions = sample_positions[start : start + rows_per_chunk]
        row_texts = [
            "\n".join(f"{col}: {val}" for col, val in zip(columns_list, row_values))
            for row_values in batch
        ]
        chunk_text = prefix + "\n\n".join(row_texts)
        # +1 to show 1-based row numbers as a user would see them in a
        # spreadsheet; these are the row's real position in the original
        # file even when `sampled` skipped rows in between.
        first_row, last_row = batch_positions[0] + 1, batch_positions[-1] + 1
        row_label = f"{first_row}" if first_row == last_row else f"{first_row}-{last_row}"
        docs.append(Document(page_content=chunk_text, metadata={**base_meta, "row": row_label}))
    return docs




def _read_csv_df(file_path: str) -> "pd.DataFrame":
    """
    Uses pandas (rather than a single fixed CSV parser) so more real-world
    CSV quirks are handled without failing the whole upload: wrong-guessed
    delimiter (semicolon/tab-separated files saved with a .csv extension),
    non-UTF-8 encoding, and a few malformed rows.
    """
    attempts = [
        {},
        {"encoding": "latin-1"},
        {"sep": None, "engine": "python"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
        {"engine": "python", "on_bad_lines": "skip"},
    ]
    best_df = None
    last_error = None
    for kwargs in attempts:
        try:
            candidate = pd.read_csv(file_path, **kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad, we try the next strategy
            last_error = e
            continue
        # A result with only 1 column is often a sign the delimiter guess
        # was wrong (e.g. a semicolon- or tab-separated file read as plain
        # comma-separated) — keep it as a fallback, but keep trying other
        # strategies in case one finds the real column structure.
        if best_df is None or candidate.shape[1] > best_df.shape[1]:
            best_df = candidate
        if best_df.shape[1] > 1:
            break
    if best_df is None:
        raise ValueError(f"Could not parse CSV file: {last_error}")
    return best_df


def load_csv(file_path: str) -> List[Document]:
    return _dataframe_to_documents(_read_csv_df(file_path), file_path)


def _read_excel_sheets(file_path: str) -> "dict[str, pd.DataFrame]":
    """
    Reads every sheet of an Excel file as {sheet_name: DataFrame}.

    Tries the `calamine` engine first (from the optional `python-calamine`
    package) — it's a Rust-based reader that parses .xlsx/.xls files
    dramatically faster than the default `openpyxl` engine, which walks
    the workbook's XML row-by-row in pure Python. For a million-row sheet
    this is the difference between minutes and single-digit seconds.
    Falls back to the default engine automatically if python-calamine
    isn't installed, so this never breaks a setup that hasn't added the
    optional dependency (`pip install python-calamine`) yet.
    """
    try:
        return pd.read_excel(file_path, sheet_name=None, engine="calamine")
    except ImportError:
        return pd.read_excel(file_path, sheet_name=None)


def load_excel(file_path: str) -> List[Document]:
    """
    Reads every sheet with pandas for consistent extraction, producing a
    summary Document plus batched-row Documents for each sheet (see
    _dataframe_to_documents above).
    """
    sheets = _read_excel_sheets(file_path)  # {sheet_name: DataFrame}
    docs: List[Document] = []
    for sheet_name, df in sheets.items():
        docs.extend(_dataframe_to_documents(df, file_path, sheet_name=str(sheet_name)))
    return docs


def load_raw_dataframes(file_path: str, source_type: str) -> "dict[str, pd.DataFrame]":
    """
    Returns the actual parsed DataFrame(s) for a csv/xlsx/xls file, keyed
    by sheet name ("__default__" for CSV, which has no sheets). This is
    what the aggregation engine (see the "Structured / aggregation
    queries" section below) computes sums, averages, group-bys etc. over
    — the chunked-and-embedded Documents from load_csv/load_excel are
    text for semantic search, not real numbers, so they can't answer
    "what's the total revenue" reliably; this gives the LLM's query plan
    something to actually run pandas math against.
    """
    if source_type == "csv":
        return {"__default__": _read_csv_df(file_path)}
    if source_type in ("xlsx", "xls"):
        sheets = _read_excel_sheets(file_path)
        return {str(name): df for name, df in sheets.items()}
    return {}



def load_docx(file_path: str) -> List[Document]:
    loader = Docx2txtLoader(file_path)
    return loader.load()


def load_html(file_path: str) -> List[Document]:
    try:
        loader = UnstructuredHTMLLoader(file_path)
        return loader.load()
    except Exception:
        # Fallback if the unstructured HTML parser isn't available.
        loader = BSHTMLLoader(file_path)
        return loader.load()


# Maps a source_type string to its loader function.
_LOADERS = {
    "pdf": load_pdf,
    "txt": load_text,
    "csv": load_csv,
    "xlsx": load_excel,
    "xls": load_excel,
    "docx": load_docx,
    "html": load_html,
}

# File extensions that map to each source_type (used by the UI/uploader).
EXTENSION_TO_TYPE = {
    "pdf": "pdf",
    "txt": "txt",
    "csv": "csv",
    "xlsx": "xlsx",
    "xls": "xls",
    "docx": "docx",
    "html": "html",
    "htm": "html",
}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
# Every loader above pulls text out of a very different container (a PDF
# text layer, a CSV parser, an HTML DOM, raw bytes...) and each leaves its
# own kind of debris behind — stray HTML tags, control characters, repeated
# blank lines, mid-word hyphenation, "NaN" cells, and so on. Cleaning this
# consistently before chunking matters a lot for retrieval quality: dirty
# chunks embed poorly and waste context window on noise instead of content.

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")  # PDF word wraps: "exam-\nple"
_HTML_ENTITIES = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&apos;": "'",
}


def clean_text(text: str, *, is_html_source: bool = False) -> str:
    """Generic text cleanup applied to every extracted document, regardless
    of source type."""
    if not text:
        return ""

    # Normalize unicode (curly quotes, ligatures, etc. -> consistent form)
    # and drop control characters that sometimes sneak in from PDFs/Excel.
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)

    if is_html_source:
        for entity, replacement in _HTML_ENTITIES.items():
            text = text.replace(entity, replacement)
        # Belt-and-suspenders: strip any leftover markup the HTML loader
        # didn't already extract as plain text.
        text = _HTML_TAG_RE.sub(" ", text)

    # Rejoin words that were hyphenated across a line break (common in
    # PDF text layers), e.g. "docu-\nment" -> "document".
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Normalize whitespace: trim trailing spaces on each line, collapse
    # runs of spaces/tabs, collapse 3+ blank lines down to one.
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text.strip()


def clean_tabular_text(text: str) -> str:
    """Extra cleanup for CSV/Excel rows, which loaders usually render as
    'column: value' lines — strips genuinely empty cells and common
    placeholder junk so a mostly-blank row doesn't become a wasted chunk."""
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop cells that are empty or just pandas/openpyxl null placeholders.
        if re.fullmatch(r"[A-Za-z0-9_ ]+:\s*(nan|none|null|n/a)?", line, re.IGNORECASE):
            _, _, value = line.partition(":")
            if not value.strip() or value.strip().lower() in ("nan", "none", "null", "n/a"):
                continue
        lines.append(line)
    return "\n".join(lines)


def _sanitize_metadata(docs: List[Document]) -> List[Document]:
    """
    Chroma's embedding upsert only accepts scalar metadata values (str,
    int, float, bool) — no lists, dicts, or None. Several loaders attach
    richer metadata than that, most notably `unstructured`-based ones
    (used for HTML here, and in earlier versions for Excel) which add
    fields like {"languages": ["eng"]} or nested coordinate dicts. Left
    as-is, that crashes the embed step with:
        "Expected metadata value to be a str, int, float or bool, got
        ['eng'] which is a list in upsert."
    This runs on every document from every loader, so that failure can
    never happen regardless of which file type or loader produced it.
    """
    for doc in docs:
        clean_meta = {}
        for key, value in doc.metadata.items():
            if value is None:
                continue  # Chroma rejects None too
            if isinstance(value, (str, int, float, bool)):
                clean_meta[key] = value
            elif isinstance(value, (list, tuple, set)):
                # e.g. languages: ["eng"] -> languages: "eng"
                joined = ", ".join(str(v) for v in value)
                if joined:
                    clean_meta[key] = joined
            else:
                # Nested dicts or other complex objects — stringify rather
                # than drop, so the information isn't silently lost, but
                # never in a shape Chroma would reject.
                clean_meta[key] = str(value)
        doc.metadata = clean_meta
    return docs


def clean_documents(docs: List[Document], source_type: str) -> List[Document]:
    """Applies type-appropriate cleaning to every Document's page_content
    and metadata, and drops any document that ends up with no real text
    left."""
    is_html = source_type in ("html", "url")
    is_tabular = source_type in ("csv", "xlsx", "xls")

    cleaned: List[Document] = []
    for doc in docs:
        text = clean_text(doc.page_content, is_html_source=is_html)
        if is_tabular:
            text = clean_tabular_text(text)
        if not text or len(text.strip()) < 2:
            continue  # skip empty/near-empty fragments (blank rows, stray tags)
        doc.page_content = text
        cleaned.append(doc)

    return _sanitize_metadata(cleaned)


def load_source(source: str, source_type: str) -> List[Document]:
    """
    source: file path (for pdf/txt/csv/xlsx/xls/docx/html) or a URL string (for url)
    source_type: one of "pdf", "txt", "csv", "xlsx", "xls", "docx", "html", "url"
    """
    if source_type == "url":
        docs = load_url(source)
    else:
        loader_fn = _LOADERS.get(source_type)
        if loader_fn is None:
            raise ValueError(f"Unsupported source_type: {source_type}")
        docs = loader_fn(source)

    return clean_documents(docs, source_type)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Embeddings + Vector store
# ---------------------------------------------------------------------------

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def reset_vector_store():
    """
    Clear out any previous Chroma data so a new session (or "clear all")
    starts from a clean slate.

    Rather than deleting the old on-disk directory in place, this always
    switches to a brand-new session sub-directory first. That sidesteps a
    real-world issue on Windows: Chroma's sqlite file can stay briefly
    locked by the OS even after Python drops all references to it, which
    made `shutil.rmtree()` raise `PermissionError` and abort mid-reset —
    leaving the server's in-memory state (and the old, supposedly-cleared
    documents) still pointing at the old data. Switching directories can
    never fail that way; the old directory is then deleted on a
    best-effort basis and simply left behind (harmless, just a few KB) if
    it's still locked.
    """
    global _current_chroma_dir
    old_dir = _current_chroma_dir
    _current_chroma_dir = os.path.join(CHROMA_ROOT, f"session_{uuid.uuid4().hex[:8]}")

    if os.path.exists(old_dir):
        gc.collect()  # release any lingering Chroma/sqlite file handles
        try:
            shutil.rmtree(old_dir)
        except OSError:
            pass  # locked (common on Windows right after use) — harmless


def build_vector_store(chunks: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    os.makedirs(_current_chroma_dir, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=_sanitize_metadata(chunks),  # belt-and-suspenders, see clean_documents()
        embedding=embeddings,
        persist_directory=_current_chroma_dir,
    )
    return vectordb


def add_to_vector_store(vectordb: Chroma, chunks: List[Document]) -> Chroma:
    vectordb.add_documents(_sanitize_metadata(chunks))
    return vectordb


# ---------------------------------------------------------------------------
# QA chain
# ---------------------------------------------------------------------------

# The UI renders answers as plain text (no markdown/HTML rendering), so the
# model is explicitly told to answer in clean prose — no **bold**, no
# markdown tables/pipes, no <br> or other HTML tags, even if the source
# document itself contains that kind of formatting/markup.
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant answering questions about documents "
        "the user has uploaded. Use ONLY the context below to answer. If "
        "the answer isn't in the context, say you don't know.\n\n"
        "Formatting rules (important):\n"
        "- Write in plain, natural prose, like a person explaining something "
        "in chat — short paragraphs or a simple hyphen list if needed.\n"
        "- Do NOT use markdown syntax: no **bold**, no # headings, no | "
        "tables, no markdown bullet symbols.\n"
        "- Do NOT include any HTML tags (like <br>) even if they appear in "
        "the source text — replace them with a normal line break or comma "
        "instead.\n"
        "- Answer in the same language the question was asked in.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


def get_llm(groq_api_key: str = None):
    """
    groq_api_key: explicit key to use. If omitted/empty, falls back to the
    GROQ_API_KEY environment variable (set via .env / docker-compose, and
    shared by every user of this deployment — see server.py).
    """
    key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "No Groq API key available. Set GROQ_API_KEY in your .env file, "
            "or pass one from the client."
        )
    return ChatGroq(
        api_key=key,
        model=GROQ_MODEL_NAME,
        temperature=0,
    )


def build_qa_chain(
    vectordb: Chroma,
    groq_api_key: str,
    k: int = 6,
    source_filter: List[str] = None,
) -> RetrievalQA:
    """
    source_filter: optional list of source names (matching the "source"
    metadata field set on each chunk — the original filename or URL) to
    scope retrieval to. None/empty means search across every uploaded
    document, which is the default.
    """
    llm = get_llm(groq_api_key)
    search_kwargs = {"k": k}
    if source_filter:
        search_kwargs["filter"] = (
            {"source": source_filter[0]}
            if len(source_filter) == 1
            else {"source": {"$in": source_filter}}
        )
    retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return qa_chain


def _clean_answer_text(text: str) -> str:
    """Belt-and-suspenders cleanup in case the model still slips in markdown
    or HTML despite the prompt instructions."""
    import re

    # Strip common HTML line-break / tag artifacts.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    # Strip markdown bold/italic markers, keep the wrapped text.
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)", r"\1", text)
    # Collapse stray markdown table pipes into a simple separator.
    text = text.replace(" | ", "\n").replace("|", "")
    # Tidy up excess blank lines left behind by the above substitutions.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def answer_question(qa_chain: RetrievalQA, question: str) -> dict:
    """Returns {"answer": str, "sources": List[Document]}"""
    result = qa_chain.invoke({"query": question})
    return {
        "answer": _clean_answer_text(result["result"]),
        "sources": result.get("source_documents", []),
    }


# ---------------------------------------------------------------------------
# Structured / aggregation queries over CSV & Excel data
# ---------------------------------------------------------------------------
# The chunk-and-embed pipeline above is semantic search: it's great for
# "what does this say about X" but structurally can't answer "what's the
# total revenue" or "top 5 by quantity" — no single retrieved chunk
# contains the whole column, so the LLM would either say "I don't know"
# or hallucinate a number.
#
# For that class of question this module instead:
#   1. Detects (by keyword) that the question is asking for a computation.
#   2. Asks the LLM to describe the computation as a small, strict JSON
#      "plan" (which table/column/operation/filters/group-by) — never as
#      Python code. The plan is the ONLY thing the LLM produces here.
#   3. Executes that plan with a small whitelist of pandas operations
#      chosen by name from the plan (sum/mean/count/min/max/median/
#      nunique, ==/!=/>/</>=/<=/contains). There is no eval()/exec() of
#      anything the model writes, so a malformed or adversarial plan can
#      at worst fail to match a column/operation name — it can't run
#      arbitrary code.

import json

_AGG_KEYWORDS = (
    "total", "sum", "average", "avg", "mean", "median", "count", "how many",
    "number of", "maximum", "max", "minimum", "min", "highest", "lowest",
    "top ", "bottom ", "group by", "grouped by", "per ", "each ",
    "greater than", "less than", "more than", "at least", "at most",
    "filter", "percentage", "percent", "%", "distinct", "unique",
)


def looks_like_aggregation_question(question: str) -> bool:
    """Cheap keyword gate so we only pay for an extra LLM call (the query
    planner below) when the question plausibly needs real computation over
    a table, not for every ordinary question."""
    q = question.lower()
    return any(kw in q for kw in _AGG_KEYWORDS)


_PLAN_PROMPT = PromptTemplate(
    input_variables=["tables", "question"],
    template=(
        "You turn a user's question about tabular data into a strict JSON "
        "query plan. Do NOT write Python or pandas code — only the JSON "
        "object described below.\n\n"
        "Available tables (name: columns with dtypes):\n{tables}\n\n"
        "Question: {question}\n\n"
        "Respond with ONLY a single JSON object (no markdown fences, no "
        "prose) with these fields:\n"
        '  "table": the table name from the list above that best matches '
        "the question (or the only table name if there's just one),\n"
        '  "operation": one of "sum", "mean", "count", "min", "max", '
        '"median", "nunique",\n'
        '  "column": the exact column name the operation applies to '
        '(omit or null for "count" of rows),\n'
        '  "group_by": exact column name to group by, or null if no '
        "grouping is needed,\n"
        '  "filters": a list of {{"column": ..., "op": "==" | "!=" | ">" | '
        '"<" | ">=" | "<=" | "contains", "value": ...}} objects (empty '
        "list if no filtering is needed),\n"
        '  "sort": "asc" or "desc" or null (only meaningful with '
        "group_by),\n"
        '  "limit": integer row/group limit (e.g. for \"top 5\"), or null.\n\n'
        "If the question genuinely isn't a computation over this data "
        '(it\'s a narrative/lookup question instead), respond with exactly '
        '{{"not_applicable": true}} and nothing else.\n\n'
        "JSON:"
    ),
)

_ALLOWED_OPS = {"sum", "mean", "count", "min", "max", "median", "nunique"}
_ALLOWED_FILTER_OPS = {"==", "!=", ">", "<", ">=", "<=", "contains"}


def _describe_tables(dataframes: "dict[str, pd.DataFrame]") -> str:
    lines = []
    for name, df in dataframes.items():
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        lines.append(f"- {name}: {cols} ({len(df)} rows)")
    return "\n".join(lines)


def _resolve_column(df: "pd.DataFrame", name: Optional[str]) -> Optional[str]:
    """Case-insensitive column lookup — the LLM won't always match casing
    exactly. Returns None (rather than raising) if there's no match, so
    the caller can fail gracefully back to the normal RAG answer."""
    if not name:
        return None
    if name in df.columns:
        return name
    lowered = {str(c).lower(): c for c in df.columns}
    return lowered.get(str(name).lower())


def plan_query(question: str, dataframes: "dict[str, pd.DataFrame]", groq_api_key: str) -> Optional[dict]:
    """Asks the LLM for a JSON query plan. Returns None if the model says
    the question isn't a computation, or if it returns something that
    isn't valid/parseable JSON — either way, the caller falls back to the
    normal semantic-search answer."""
    if not dataframes:
        return None
    llm = get_llm(groq_api_key)
    prompt = _PLAN_PROMPT.format(tables=_describe_tables(dataframes), question=question)
    try:
        raw = llm.invoke(prompt).content.strip()
    except Exception:
        return None

    # Models sometimes wrap JSON in ```json fences despite instructions.
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        plan = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(plan, dict) or plan.get("not_applicable"):
        return None
    if plan.get("operation") not in _ALLOWED_OPS:
        return None
    return plan


def execute_query_plan(plan: dict, dataframes: "dict[str, pd.DataFrame]") -> Optional[dict]:
    """Runs the plan using only whitelisted pandas operations selected by
    name — no exec()/eval() of anything model-generated. Returns None on
    any mismatch (unknown table/column/op) so the caller can fall back
    cleanly instead of surfacing an internal error to the user."""
    table_name = plan.get("table")
    df = dataframes.get(table_name)
    if df is None:
        # Fuzzy match: substring / single-table fallback.
        if len(dataframes) == 1:
            table_name, df = next(iter(dataframes.items()))
        else:
            matches = [k for k in dataframes if table_name and table_name.lower() in k.lower()]
            if len(matches) == 1:
                table_name, df = matches[0], dataframes[matches[0]]
    if df is None:
        return None

    working = df

    # --- filters -----------------------------------------------------
    for f in plan.get("filters") or []:
        col = _resolve_column(working, f.get("column"))
        op = f.get("op")
        value = f.get("value")
        if col is None or op not in _ALLOWED_FILTER_OPS:
            continue
        series = working[col]
        try:
            if op == "contains":
                working = working[series.astype(str).str.contains(str(value), case=False, na=False)]
                continue
            # Try numeric comparison first (most filters are on numbers),
            # fall back to string comparison if that fails.
            try:
                cmp_value = pd.to_numeric(pd.Series([value])).iloc[0]
                cmp_series = pd.to_numeric(series, errors="coerce")
            except (ValueError, TypeError):
                cmp_value = value
                cmp_series = series.astype(str)
            if op == "==":
                mask = cmp_series == cmp_value
            elif op == "!=":
                mask = cmp_series != cmp_value
            elif op == ">":
                mask = cmp_series > cmp_value
            elif op == "<":
                mask = cmp_series < cmp_value
            elif op == ">=":
                mask = cmp_series >= cmp_value
            else:  # "<="
                mask = cmp_series <= cmp_value
            working = working[mask]
        except Exception:
            continue  # skip a filter we couldn't apply rather than failing the whole query

    operation = plan.get("operation")
    column = _resolve_column(working, plan.get("column"))
    group_by = _resolve_column(working, plan.get("group_by"))

    if operation != "count" and column is None and group_by is None:
        return None  # can't compute sum/mean/etc. with no column to operate on

    try:
        if group_by:
            if operation == "count":
                grouped = working.groupby(group_by).size()
            else:
                grouped = working.groupby(group_by)[column].agg(operation)
            if plan.get("sort") in ("asc", "desc"):
                grouped = grouped.sort_values(ascending=(plan["sort"] == "asc"))
            limit = plan.get("limit")
            if isinstance(limit, int) and limit > 0:
                grouped = grouped.head(limit)
            else:
                grouped = grouped.head(50)  # sane cap so the answer stays readable
            return {
                "kind": "grouped",
                "table": table_name,
                "operation": operation,
                "column": column,
                "group_by": group_by,
                "result": grouped,
                "matched_rows": len(working),
            }
        else:
            if operation == "count":
                value = len(working)
            else:
                value = getattr(working[column], operation)()
            return {
                "kind": "scalar",
                "table": table_name,
                "operation": operation,
                "column": column,
                "result": value,
                "matched_rows": len(working),
            }
    except Exception:
        return None


def _format_number(value) -> str:
    if hasattr(value, "item"):  # numpy scalar (int64, float64, ...) -> native Python type
        value = value.item()
    if isinstance(value, float):
        # Keep it readable: whole numbers without a trailing ".0", others
        # rounded to 2 decimals with thousands separators.
        if value == int(value):
            return f"{int(value):,}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def format_query_result(plan: dict, computed: dict) -> str:
    op_label = {
        "sum": "total", "mean": "average", "count": "count", "min": "minimum",
        "max": "maximum", "median": "median", "nunique": "distinct count",
    }.get(computed["operation"], computed["operation"])
    table = computed["table"]
    matched = computed["matched_rows"]
    filtered_note = "" if matched == 0 or plan.get("filters") else ""

    if computed["kind"] == "scalar":
        subject = f"'{computed['column']}'" if computed["column"] else "rows"
        value = _format_number(computed["result"])
        note = f" (matching {matched} row{'s' if matched != 1 else ''})" if plan.get("filters") else ""
        return f"The {op_label} of {subject} in {table} is {value}{note}."

    # grouped
    series = computed["result"]
    subject = f"'{computed['column']}'" if computed["column"] else "count"
    parts = [f"{idx}: {_format_number(val)}" for idx, val in series.items()]
    body = "; ".join(parts) if parts else "no matching groups"
    note = f" (matching {matched} row{'s' if matched != 1 else ''})" if plan.get("filters") else ""
    return f"{op_label.capitalize()} of {subject} in {table}, grouped by '{computed['group_by']}'{note}: {body}."


def answer_tabular_question(question: str, dataframes: "dict[str, pd.DataFrame]", groq_api_key: str) -> Optional[dict]:
    """
    Top-level entry point for the aggregation path. Returns None if the
    question doesn't look like a computation, if the LLM can't produce a
    valid plan, or if the plan can't be executed — in every such case the
    caller should fall back to the normal semantic-search answer instead.
    On success, returns {"answer": str, "table": str, "detail": dict}.
    """
    if not dataframes or not looks_like_aggregation_question(question):
        return None
    plan = plan_query(question, dataframes, groq_api_key)
    if plan is None:
        return None
    computed = execute_query_plan(plan, dataframes)
    if computed is None:
        return None
    return {
        "answer": format_query_result(plan, computed),
        "table": computed["table"],
        "detail": plan,
    }