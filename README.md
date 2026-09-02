# The Reading Room — Document Q&A

Ask questions of your own documents: PDF, DOCX, TXT, CSV, Excel, HTML, or any web page.
FastAPI backend + a static HTML/CSS/JS frontend, local free embeddings
(HuggingFace `sentence-transformers`), and answers from an open-weight
model served via Groq's API.

## Files

```
server.py         FastAPI app: API routes + serves the frontend
rag_engine.py     Loading, cleaning, chunking, embedding, retrieval, QA
index.html        Frontend markup
app.js            Frontend logic (talks to server.py's /api/* routes)
style.css         Frontend styling
requirements.txt  Python dependencies
.env.example      Template for your GROQ_API_KEY
bake_embeddings.py  Best-effort pre-download of the embedding model at
                    Docker build time (see "Run with Docker" below)
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

3. **Set your Groq API key.** Copy the example env file and fill it in:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and paste your key from https://console.groq.com/keys.
   Everyone who uses this running server shares this one key — there's no
   per-user login or key entry in the frontend.

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
- During the build, `bake_embeddings.py` tries to pre-download the local
  embedding model so the first real request after startup is fast. This
  is a pure optimization — if Hugging Face rate-limits the download
  (`HTTP 429`, common when a build is retried several times in a row),
  the script retries a few times and then just skips baking the model in
  rather than failing the whole build. In that case the model downloads
  automatically on the container's first real request instead, so the
  app still works fine either way — the only difference is a slightly
  slower first document upload.

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
- **"Backend unreachable"**: the FastAPI server isn't running, or you're
  opening `index.html` directly as a file instead of via
  `http://localhost:8000`.
- **Excel ingestion feels slow**: confirm `python-calamine` actually
  installed (`pip show python-calamine`); without it, Excel falls back to
  the slower `openpyxl` engine silently.