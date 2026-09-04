# The Reading Room — Document Q&A

Ask questions of your own documents: PDF, DOCX, TXT, CSV, Excel, HTML, or any web page.
FastAPI backend + a static HTML/CSS/JS frontend, free embeddings via
Hugging Face's hosted Inference API (no local model/torch — keeps the
app's memory footprint small enough for free-tier hosting), and answers
from an open-weight model served via Groq's API.

## Files

```
server.py         FastAPI app: API routes + serves the frontend
rag_engine.py     Loading, cleaning, chunking, embedding, retrieval, QA
index.html        Frontend markup
app.js            Frontend logic (talks to server.py's /api/* routes)
style.css         Frontend styling
requirements.txt  Python dependencies
.env.example      Template for your GROQ_API_KEY and HF_TOKEN
```

## Setup

1. **Python 3.10+ recommended.** Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   `python-calamine` is in there for fast Excel reading — if it fails to
   install on your platform, delete that line and reinstall; Excel
   ingestion will still work, just slower on very large files (see
   `_read_excel_sheets` in `rag_engine.py`, which falls back automatically
   either way).

   **After installing, also run once:**

   ```bash
   playwright install chromium
   ```

   This downloads the headless Chromium browser that `load_url()` uses to
   render pages (see "URL fetching" below). If you skip this step, URL
   ingestion still works — it just silently falls back to the older
   plain-HTTP fetch, which can't execute JavaScript.

3. **Set your API keys.** Copy the example env file and fill it in:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and paste in:
   - `GROQ_API_KEY` from https://console.groq.com/keys (free). Everyone
     who uses this running server shares this one key — there's no
     per-user login or key entry in the frontend.
   - `HF_TOKEN` from https://huggingface.co/settings/tokens (free). When
     creating it, pick the **"Inference"** preset (not "Read-Only" — a
     plain read token can download models but can't call the Inference
     API, which is what this app actually uses). See "Why hosted
     embeddings instead of local?" below for why it's needed at all.

## Run

```bash
uvicorn server:app --reload
```

Then open **http://localhost:8000** in your browser.

- `--reload` restarts the server automatically when you edit the Python
  files — handy during development, drop it for a production run.
- To listen on all interfaces (e.g. to reach it from another device on
  your network): `uvicorn server:app --host 0.0.0.0 --port 8000`

## Run with Docker (alternative to the venv setup above)

No local Python/venv/Playwright setup needed — the image bundles Python,
all dependencies, and Chromium (via Microsoft's official Playwright
base image) together.

```bash
cp .env.example .env      # fill in your GROQ_API_KEY first
docker compose up --build
```

Then open **http://localhost:8000**.

- The vector store persists across restarts in a named Docker volume
  (`chroma_data`) — `docker compose down` (without `-v`) keeps your
  filed documents; `docker compose down -v` wipes them.
- To run without docker-compose:
  ```bash
  docker build -t reading-room .
  docker run -p 8000:8000 --env-file .env reading-room
  ```
- If you bump the `playwright` version in `requirements.txt`, also bump
  the base image tag in the `Dockerfile` (`FROM
  mcr.microsoft.com/playwright/python:vX.Y.Z-jammy`) to match — a
  mismatched Python package vs. pre-baked browser version is the most
  common way Playwright breaks inside Docker.

### Deploying to a free host (Render, etc.)

- The `Dockerfile`'s `CMD` reads the platform's `$PORT` env var
  automatically (falling back to 8000 for local/docker-compose use where
  `$PORT` isn't set) — most PaaS hosts assign a port dynamically and
  route traffic to it, so a hardcoded port would make health checks fail.
- Set `GROQ_API_KEY` and `HF_TOKEN` as environment variables in your
  host's dashboard (e.g. Render → your service → **Environment**) — the
  `.env` file itself is git-ignored and never reaches the server on its
  own.
- Free tiers commonly cap RAM at ~512MB. This app deliberately avoids
  loading any local embedding model (see below), so it should comfortably
  fit — the one thing that still uses meaningful memory transiently is
  the headless Chromium browser Playwright launches during URL ingestion.
- Render's free web services also don't persist disk across restarts, so
  `chroma_data` (and anything filed) resets whenever the service
  restarts/redeploys/spins down from inactivity — expected on free tier,
  not a bug.

### Why hosted embeddings instead of local?

Earlier versions of this app loaded the embedding model locally via
`sentence-transformers`, which pulls in `torch`. Torch's own baseline
memory footprint plus the loaded model comfortably exceeds a 512MB
limit, which was silently OOM-killing the process mid-request on Render
free tier (no Python traceback — just the process restarting in the
logs). `rag_engine.get_embeddings()` now calls the same model
(`sentence-transformers/all-MiniLM-L6-v2`) over Hugging Face's hosted
Inference API instead — this process never holds torch or model weights
in memory, only the small embedding vectors that come back over the
network, at the cost of one extra network round-trip per embedding
call. Needs `HF_TOKEN` set (see "Setup" above).

## Using it

1. Drop a file (PDF/TXT/CSV/XLSX/XLS/DOCX/HTML) onto the left panel, or
   switch to the **URL** tab and paste a link to a page.
2. Wait for the "Filed" status — filed documents show up in the catalog
   drawer below.
3. Ask a question in the box at the bottom. Answers cite the specific
   passages/rows they came from.
4. Use the "Asking: all documents" selector above the question box to
   scope a question to specific files instead of everything you've filed.
5. **Clear all** wipes the current session's vector store and dataframes
   so you can start fresh.

## Known limits (by design, not bugs)

- **Scanned/image-only PDFs** have no extractable text layer — this app
  doesn't run OCR, and will warn you rather than fail silently.
- **URL fetching**: `load_url()` renders pages with a real headless
  Chromium browser (Playwright), so most JavaScript-heavy modern sites
  and SPAs now work correctly — a plain HTTP fetch can't see content
  that only appears after JS runs, but a rendered browser can.
  **Sites with real bot-checks (Cloudflare Turnstile, hCaptcha, actual
  CAPTCHAs) still won't work** — those are specifically built to detect
  and block automated browsers, headless or not, and this app makes no
  attempt to evade that detection. The fallback for those pages: save
  the page from your own browser (Print → PDF, or Save Page As → HTML)
  and upload that file instead of the URL.
- **Very large CSV/Excel files** (100k+ rows): only an evenly-spaced
  sample of rows (`MAX_EMBEDDED_ROWS` in `rag_engine.py`, default 5,000)
  is embedded for semantic search, to keep ingestion fast. Sums,
  averages, counts, and group-bys still run against every row, not just
  the sample — see `answer_tabular_question` in `rag_engine.py`.
- **Single shared session** (`STATE` in `server.py`) — this is built for
  local/single-user use. Multiple simultaneous users would share the
  same catalog of documents; there's no per-user isolation.

## Troubleshooting

- **"No GROQ_API_KEY set on server"** in the status footer: your `.env`
  file is missing, empty, or not in the same directory as `server.py`.
- **"No HF_TOKEN set" error when filing a document**: same idea — set
  `HF_TOKEN` in `.env` (local) or your host's environment variables
  (deployed).
- **"Backend unreachable"**: the FastAPI server isn't running, or you're
  opening `index.html` directly as a file instead of via
  `http://localhost:8000`.
- **Excel ingestion feels slow**: confirm `python-calamine` actually
  installed (`pip show python-calamine`); without it, Excel falls back to
  the slower `openpyxl` engine silently.
- **App crashes / restarts with no error, on a free-RAM host**: check
  your host's memory usage graph around the time of the crash — an
  out-of-memory kill looks exactly like this (silent process restart, no
  traceback). This app is already built to avoid that (see "Why hosted
  embeddings instead of local?" above) — if it's still happening, the
  most likely remaining cause is Playwright's Chromium browser during URL
  ingestion; file/CSV/PDF uploads don't launch it.