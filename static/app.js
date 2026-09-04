// ============================================================================
// The Reading Room — frontend logic
// Talks to the FastAPI backend (server.py) over a small JSON/multipart API.
// No API key ever needs to be entered here — the server uses a single
// shared GROQ_API_KEY configured in its own .env file (see server.py /
// rag_engine.py), so every visitor can ask questions immediately.
// ============================================================================

const API_BASE = "/api";

// ---- Element refs ----------------------------------------------------------
const catalogPanel    = document.getElementById("catalog");
const catalogBackdrop = document.getElementById("catalogBackdrop");
const mobileMenuBtn   = document.getElementById("mobileMenuBtn");
const catalogCloseBtn = document.getElementById("catalogCloseBtn");

const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("fileInput");
const urlInput       = document.getElementById("urlInput");
const ingestUrlBtn   = document.getElementById("ingestUrlBtn");
const intakeStatus   = document.getElementById("intakeStatus");
const drawerList     = document.getElementById("drawerList");
const drawerEmpty    = document.getElementById("drawerEmpty");
const clearBtn       = document.getElementById("clearBtn");
const intakeTabs     = document.querySelectorAll(".intake-tab");
const panelFile      = document.getElementById("panel-file");
const panelUrl       = document.getElementById("panel-url");
const statusDot      = document.getElementById("statusDot");
const statusText     = document.getElementById("statusText");

const deskEmpty      = document.getElementById("deskEmpty");
const thread         = document.getElementById("thread");
const askForm        = document.getElementById("askForm");
const questionInput  = document.getElementById("questionInput");
const askBtn         = document.getElementById("askBtn");

const scopeBar        = document.getElementById("scopeBar");
const scopeSelect      = document.getElementById("scopeSelect");
const scopeLabel       = document.getElementById("scopeLabel");
const scopeAll         = document.getElementById("scopeAll");
const scopeFileList    = document.getElementById("scopeFileList");

// Templates
const tplUserMsg      = document.getElementById("tpl-user-msg");
const tplAssistantMsg = document.getElementById("tpl-assistant-msg");
const tplSourceCard   = document.getElementById("tpl-source-card");
const tplDrawerItem   = document.getElementById("tpl-drawer-item");
const tplScopeItem    = document.getElementById("tpl-scope-item");

let hasDocuments = false;
let allSourceNames = [];        // every currently-filed document name (deduped)
let selectedSourceNames = null; // null = "all documents"; otherwise a Set of names

// ---- Mobile catalog drawer ---------------------------------------------------
// On screens <=860px the catalog becomes an off-canvas drawer (see
// style.css) instead of a permanently-squeezed sidebar. These helpers
// open/close it and keep the hamburger button + backdrop + body scroll
// lock in sync. On wider screens these are no-ops (the elements involved
// are simply hidden by CSS, and matchMedia checks below make sure
// leftover "open" state can't get stuck if the window is resized).
const mobileBreakpoint = window.matchMedia("(max-width: 860px)");

function openCatalog() {
  catalogPanel.classList.add("open");
  catalogBackdrop.classList.add("visible");
  mobileMenuBtn?.setAttribute("aria-expanded", "true");
  document.body.classList.add("no-scroll");
}

function closeCatalog() {
  catalogPanel.classList.remove("open");
  catalogBackdrop.classList.remove("visible");
  mobileMenuBtn?.setAttribute("aria-expanded", "false");
  document.body.classList.remove("no-scroll");
}

mobileMenuBtn?.addEventListener("click", () => {
  catalogPanel.classList.contains("open") ? closeCatalog() : openCatalog();
});
catalogCloseBtn?.addEventListener("click", closeCatalog);
catalogBackdrop?.addEventListener("click", closeCatalog);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && catalogPanel.classList.contains("open")) closeCatalog();
});
// Resizing past the breakpoint (e.g. rotating a tablet, or a desktop
// devtools resize) could otherwise leave the drawer "open" underneath a
// now-desktop layout, with body scroll still locked — reset it.
mobileBreakpoint.addEventListener("change", (e) => {
  if (!e.matches) closeCatalog();
});

// ---- Backend connectivity indicator ------------------------------------------
async function checkConnection() {
  try {
    const res = await fetch(`${API_BASE}/config`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (data.has_server_key) {
      statusDot.className = "status-dot ok";
      statusText.textContent = "Ready — connected to Groq";
    } else {
      statusDot.className = "status-dot warn";
      statusText.textContent = "No GROQ_API_KEY set on server";
    }
  } catch (err) {
    statusDot.className = "status-dot error";
    statusText.textContent = "Backend unreachable";
  }
}
checkConnection();

// ---- Intake tab switching ---------------------------------------------------
intakeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    intakeTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const mode = tab.dataset.mode;
    panelFile.classList.toggle("hidden", mode !== "file");
    panelUrl.classList.toggle("hidden", mode !== "url");
    setStatus("");
  });
});

// ---- Status helper -----------------------------------------------------------
function setStatus(text, kind) {
  intakeStatus.textContent = text;
  intakeStatus.className = "intake-status" + (kind ? " " + kind : "");
}

// ---- File intake: click + drag-and-drop ---------------------------------------
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    ingestFiles(Array.from(fileInput.files));
    fileInput.value = "";
  }
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const files = Array.from(e.dataTransfer.files || []);
  if (files.length) ingestFiles(files);
});

// ---- URL intake ---------------------------------------------------------------
ingestUrlBtn.addEventListener("click", () => {
  const url = urlInput.value.trim();
  if (!url) {
    setStatus("Paste a URL first.", "error");
    return;
  }
  ingestUrl(url);
});

urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    ingestUrlBtn.click();
  }
});

// ---- Ingest calls ---------------------------------------------------------------
async function ingestFiles(files) {
  // Process one at a time (not in parallel) — each ingest call adds its
  // chunks to the same shared vector store on the backend, so sequencing
  // them avoids races and lets the status line show clear progress when
  // several different file types are dropped together (e.g. a PDF + a
  // CSV + a DOCX all at once).
  let filed = 0;
  for (const file of files) {
    setStatus(`Filing "${file.name}" (${filed + 1}/${files.length})…`);
    const ok = await ingestFile(file, { silent: true });
    if (ok) filed++;
  }
  if (filed > 0) {
    setStatus(
      files.length > 1 ? `Filed ${filed}/${files.length} document(s).` : "Filed.",
      "success"
    );
  }
}

async function ingestFile(file, { silent = false } = {}) {
  if (!silent) setStatus(`Filing "${file.name}"…`);
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/ingest/file`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to ingest file.");
    onIngestSuccess(data, { announce: !silent });
    return true;
  } catch (err) {
    setStatus(`"${file.name}": ${err.message}`, "error");
    return false;
  }
}

async function ingestUrl(url) {
  setStatus(`Fetching page…`);
  try {
    const res = await fetch(`${API_BASE}/ingest/url`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to fetch URL.");
    urlInput.value = "";
    onIngestSuccess(data);
  } catch (err) {
    setStatus(err.message, "error");
  }
}

function onIngestSuccess(data, { announce = true } = {}) {
  if (data.warning) {
    // Thin/blocked content detected (e.g. a JS-heavy or bot-blocked page) —
    // still file it (so the user can inspect what little was captured),
    // but make the limitation obvious instead of a confusing "I don't
    // know" answer showing up later with no explanation.
    setStatus(`Filed, but: ${data.warning}`, "warn");
  } else if (announce) {
    setStatus(`Filed — ${data.chunks} passages indexed and cleaned.`, "success");
  }
  renderDrawer(data.sources);
  enableAsking();
  // On mobile the catalog is a drawer sitting on top of the desk — once
  // something's filed, get it out of the way so the person can see the
  // "ready to ask" state and start typing immediately.
  if (mobileBreakpoint.matches) closeCatalog();
}

// ---- Drawer rendering -------------------------------------------------------
function renderDrawer(sources) {
  hasDocuments = sources.length > 0;
  drawerEmpty.classList.toggle("hidden", hasDocuments);
  drawerList.querySelectorAll(".drawer-item").forEach((el) => el.remove());

  sources.forEach((s) => {
    const node = tplDrawerItem.content.cloneNode(true);
    node.querySelector(".drawer-item-name").textContent = s.name;
    node.querySelector(".drawer-item-chunks").textContent = `${s.chunks}`;
    drawerList.appendChild(node);
  });

  renderScopeSelector(sources);
}

// ---- "Ask about" scope selector ---------------------------------------------
// Lets the user narrow a question to one or more specific filed documents
// instead of always searching across everything. Re-uploading the same
// name just adds more chunks to the same source, so we dedupe by name here.
function renderScopeSelector(sources) {
  const names = [...new Set(sources.map((s) => s.name))];
  allSourceNames = names;

  // Drop any previously-selected names that no longer exist (e.g. after
  // "clear all" + fresh uploads), and fall back to "all" if nothing
  // selectable remains.
  if (selectedSourceNames) {
    selectedSourceNames = new Set(
      [...selectedSourceNames].filter((n) => names.includes(n))
    );
    if (selectedSourceNames.size === 0) selectedSourceNames = null;
  }

  scopeBar.classList.toggle("hidden", names.length === 0);
  if (names.length === 0) return;

  scopeFileList.innerHTML = "";
  names.forEach((name) => {
    const node = tplScopeItem.content.cloneNode(true);
    const checkbox = node.querySelector(".scope-file-checkbox");
    checkbox.checked = !!selectedSourceNames && selectedSourceNames.has(name);
    node.querySelector(".scope-file-name").textContent = name;
    checkbox.addEventListener("change", () => onScopeFileToggle(name, checkbox.checked));
    scopeFileList.appendChild(node);
  });

  scopeAll.checked = !selectedSourceNames;
  updateScopeLabel();
}

function onScopeFileToggle(name, checked) {
  if (!selectedSourceNames) selectedSourceNames = new Set();
  if (checked) {
    selectedSourceNames.add(name);
  } else {
    selectedSourceNames.delete(name);
  }
  if (selectedSourceNames.size === 0) selectedSourceNames = null; // nothing picked -> back to "all"
  scopeAll.checked = !selectedSourceNames;
  updateScopeLabel();
}

scopeAll.addEventListener("change", () => {
  if (scopeAll.checked) {
    selectedSourceNames = null;
    scopeFileList.querySelectorAll(".scope-file-checkbox").forEach((cb) => (cb.checked = false));
  } else if (allSourceNames.length) {
    // Don't allow leaving the selector with nothing scoped at all.
    scopeAll.checked = true;
  }
  updateScopeLabel();
});

function updateScopeLabel() {
  if (!selectedSourceNames || selectedSourceNames.size === 0) {
    scopeLabel.textContent = "Asking: all documents";
  } else if (selectedSourceNames.size === 1) {
    scopeLabel.textContent = `Asking: ${[...selectedSourceNames][0]}`;
  } else {
    scopeLabel.textContent = `Asking: ${selectedSourceNames.size} documents`;
  }
}

clearBtn.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/clear`, { method: "POST" });
  } catch (err) {
    // Still clear locally even if the request fails.
  }
  selectedSourceNames = null;
  renderDrawer([]);
  thread.innerHTML = "";
  thread.classList.add("hidden");
  deskEmpty.classList.remove("hidden");
  disableAsking();
  setStatus("Catalog cleared.");
});

// ---- Asking -------------------------------------------------------------------
function enableAsking() {
  questionInput.disabled = false;
  askBtn.disabled = false;
  deskEmpty.classList.add("hidden");
  thread.classList.remove("hidden");
  questionInput.focus();
}

function disableAsking() {
  questionInput.disabled = true;
  askBtn.disabled = true;
}

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  scopeSelect.removeAttribute("open");

  addUserMessage(question);
  questionInput.value = "";
  askBtn.disabled = true;
  const pendingBubble = addAssistantPending();

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        sources: selectedSourceNames ? [...selectedSourceNames] : null,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Something went wrong.");
    fillAssistantMessage(pendingBubble, data);
  } catch (err) {
    fillAssistantError(pendingBubble, err.message);
  } finally {
    askBtn.disabled = false;
    questionInput.focus();
  }
});

// ---- Thread rendering -----------------------------------------------------------
function addUserMessage(text) {
  const node = tplUserMsg.content.cloneNode(true);
  node.querySelector(".msg-bubble").textContent = text;
  thread.appendChild(node);
  scrollThreadToBottom();
}

function addAssistantPending() {
  const node = tplAssistantMsg.content.cloneNode(true);
  const bubble = node.querySelector(".msg-bubble");
  bubble.innerHTML = '<span class="thinking-dots"><span></span><span></span><span></span></span>';
  bubble.classList.add("pending");
  thread.appendChild(node);
  scrollThreadToBottom();
  return thread.lastElementChild;
}

function fillAssistantMessage(msgEl, data) {
  const bubble = msgEl.querySelector(".msg-bubble");
  bubble.classList.remove("pending");
  bubble.textContent = data.answer;

  const sourcesEl = msgEl.querySelector(".sources");
  (data.sources || []).forEach((s) => {
    const card = tplSourceCard.content.cloneNode(true);
    const label = s.page ? `${s.source} · p.${s.page}` : s.source;
    card.querySelector(".source-card-name").textContent = label;
    card.querySelector(".source-card-snippet").textContent = s.snippet;
    sourcesEl.appendChild(card);
  });
  scrollThreadToBottom();
}

function fillAssistantError(msgEl, message) {
  const bubble = msgEl.querySelector(".msg-bubble");
  bubble.classList.remove("pending");
  bubble.classList.add("error");
  bubble.textContent = `Couldn't get an answer: ${message}`;
}

function scrollThreadToBottom() {
  thread.scrollTop = thread.scrollHeight;
}

// ---- Restore existing session on load (in case the backend already has docs) --
(async function restoreSession() {
  try {
    const res = await fetch(`${API_BASE}/documents`);
    const data = await res.json();
    if (data.sources && data.sources.length) {
      renderDrawer(data.sources);
      enableAsking();
    }
  } catch (err) {
    // Backend not reachable yet on first paint — ignore, user can retry via UI.
  }
})();