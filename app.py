import os
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Body

from watchfiles import awatch

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter


# ----------------------------
# Docs root manager
# ----------------------------

def default_docs_root() -> Path:
    root = os.environ.get("DOCS_ROOT", "").strip()
    if root:
        return Path(root).expanduser().resolve()
    return (Path.cwd() / "docs").resolve()


class DocsRoot:
    def __init__(self, initial: Path) -> None:
        self._root = initial
        self._lock = asyncio.Lock()

    async def get(self) -> Path:
        async with self._lock:
            return self._root

    async def set(self, new_root: Path) -> None:
        async with self._lock:
            self._root = new_root


docs_root = DocsRoot(default_docs_root())


# ----------------------------
# Markdown + code highlighting
# ----------------------------

formatter = HtmlFormatter(nowrap=False)
PYGMENTS_CSS = formatter.get_style_defs(".codehilite")

md = (
    MarkdownIt("commonmark", {"html": True, "linkify": True})
    .enable("table")  # <- enable pipe tables while staying on commonmark
    .use(footnote_plugin)
    .use(tasklists_plugin)
)

def _highlight_code(code: str, lang: str) -> str:
    lang = (lang or "").strip().lower()
    try:
        if not lang:
            lexer = TextLexer()
        else:
            lexer = get_lexer_by_name(lang)
    except Exception:
        lexer = TextLexer()
    return highlight(code, lexer, formatter)

# Override fence renderer for server-side pygments highlighting
def fence_renderer(tokens: List[Token], idx: int, options, env) -> str:
    token = tokens[idx]
    info = (token.info or "").strip()
    lang = info.split()[0] if info else ""
    code = token.content or ""
    highlighted = _highlight_code(code, lang)
    # Wrap with a class that Pygments CSS targets
    return f'<div class="codehilite">{highlighted}</div>'

md.renderer.rules["fence"] = fence_renderer


def render_markdown(text: str, title: str) -> str:
    html = md.render(text)
    return f"<h1>{title}</h1>{html}"


# ----------------------------
# Tree + file utils
# ----------------------------

MD_SUFFIXES = {".md", ".markdown", ".mdx"}

def is_markdown_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in MD_SUFFIXES

def safe_join(root: Path, rel_path: str) -> Optional[Path]:
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.exists() or not is_markdown_file(candidate):
        return None
    return candidate

def safe_relpath(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()

def build_tree(root: Path) -> Dict[str, Any]:
    def walk(dir_path: Path) -> Dict[str, Any]:
        children: List[Dict[str, Any]] = []
        try:
            entries = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except FileNotFoundError:
            entries = []

        for e in entries:
            if e.name.startswith("."):
                continue
            if e.is_dir():
                subtree = walk(e)
                if subtree.get("children"):
                    children.append(subtree)
            else:
                if is_markdown_file(e):
                    children.append({
                        "name": e.stem,
                        "type": "file",
                        "path": safe_relpath(root, e),
                        "filename": e.name,
                    })
        return {"name": dir_path.name, "type": "dir", "children": children}

    return walk(root)

def iter_markdown_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.name.startswith("."):
            continue
        if is_markdown_file(p):
            out.append(p)
    return out


# ----------------------------
# Full-text search (simple + fast enough for docs)
# ----------------------------

def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def make_snippet(text: str, match_span: Tuple[int, int], radius: int = 110) -> str:
    start, end = match_span
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right]
    # Mark match with <mark>
    rel_start = start - left
    rel_end = end - left
    snippet = (
        snippet[:rel_start]
        + "<mark>"
        + snippet[rel_start:rel_end]
        + "</mark>"
        + snippet[rel_end:]
    )
    # Escape minimal HTML (we'll display in a safe container)
    snippet = (snippet
               .replace("&", "&amp;")
               .replace("<mark>", "<<<MARK>>>")
               .replace("</mark>", "<<<ENDMARK>>>")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
               .replace("<<<MARK>>>", "<mark>")
               .replace("<<<ENDMARK>>>", "</mark>"))
    # Add ellipses if trimmed
    if left > 0:
        snippet = "… " + snippet
    if right < len(text):
        snippet = snippet + " …"
    return snippet

def search_in_text(text: str, query: str) -> List[Tuple[int, int]]:
    """
    Case-insensitive substring search. Returns spans (start,end).
    Keep it simple; upgrade to ripgrep / Whoosh later if needed.
    """
    text_l = text.lower()
    q = query.lower()
    spans: List[Tuple[int, int]] = []
    i = 0
    while True:
        j = text_l.find(q, i)
        if j == -1:
            break
        spans.append((j, j + len(q)))
        i = j + max(1, len(q))
        if len(spans) >= 20:  # cap per file
            break
    return spans


# ----------------------------
# Websocket manager
# ----------------------------

class WSManager:
    def __init__(self) -> None:
        self._clients: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._clients:
                self._clients.remove(ws)

    async def broadcast(self, message: str) -> None:
        async with self._lock:
            clients = list(self._clients)
        dead: List[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

ws_manager = WSManager()


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Simple Markdown Docs Viewer")

STATIC_DIR = Path.cwd() / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_watcher_task: Optional[asyncio.Task] = None

async def _start_watcher() -> None:
    """
    Watches the current docs root; if docs root changes, we restart watcher.
    """
    global _watcher_task

    if _watcher_task and not _watcher_task.done():
        _watcher_task.cancel()
        try:
            await _watcher_task
        except asyncio.CancelledError:
            # Expected when restarting the watcher for a new docs root.
            pass
        except Exception:
            pass

    async def watcher_loop(root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        async for _changes in awatch(str(root)):
            await ws_manager.broadcast("reload")

    root = await docs_root.get()
    _watcher_task = asyncio.create_task(watcher_loop(root))


@app.on_event("startup")
async def startup() -> None:
    await _start_watcher()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Docs Viewer</title>
  <style>
    {PYGMENTS_CSS}

    :root {{
      --bg: #ffffff;
      --fg: #111827;
      --muted: #6b7280;
      --border: #e5e7eb;
      --panel: #f9fafb;
      --link: #2563eb;
      --codebg: #f3f4f6;
      --selected: rgba(37, 99, 235, 0.12);
      --danger: #dc2626;
      --ok: #16a34a;
    }}

    [data-theme="dark"] {{
      --bg: #0b1020;
      --fg: #e5e7eb;
      --muted: #94a3b8;
      --border: #1f2937;
      --panel: #0f172a;
      --link: #60a5fa;
      --codebg: #111827;
      --selected: rgba(96, 165, 250, 0.18);
      --danger: #f87171;
      --ok: #4ade80;
    }}

    [data-theme="sepia"] {{
      --bg: #fbf3e6;
      --fg: #2b2b2b;
      --muted: #6b5b4b;
      --border: #e7d7c6;
      --panel: #f6eadc;
      --link: #b45309;
      --codebg: #f1e3d3;
      --selected: rgba(180, 83, 9, 0.14);
      --danger: #b91c1c;
      --ok: #15803d;
    }}

    [data-theme="nord"] {{
      --bg: #2e3440;
      --fg: #eceff4;
      --muted: #d8dee9;
      --border: #3b4252;
      --panel: #323a49;
      --link: #88c0d0;
      --codebg: #3b4252;
      --selected: rgba(136, 192, 208, 0.20);
      --danger: #ff7a7a;
      --ok: #7dffa0;
    }}

    html, body {{
      height: 100%;
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }}

    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    .app {{
      display: grid;
      grid-template-columns: 360px 1fr;
      height: 100vh;
    }}

    .sidebar{{
      border-right: 1px solid var(--border);
      background: var(--panel);

      /* key change */
      display: grid;
      grid-template-rows: auto 1fr; /* topbar fixed, tree takes remaining height */
      min-width: 280px;

      /* prevent the whole page from scrolling because of sidebar content */
      min-height: 0;
    }}

    .topbar{{
      /* stays fixed at top; not scrollable */
      padding: 12px;
      border-bottom: 1px solid var(--border);
    }}

    .row {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}

    .input {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--fg);
      outline: none;
    }}

    .theme {{
      padding: 10px 10px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--fg);
    }}

    .btn {{
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--fg);
      cursor: pointer;
      font-weight: 600;
      white-space: nowrap;
    }}
    .btn:hover {{ filter: brightness(0.98); }}
    [data-theme="dark"] .btn:hover,
    [data-theme="nord"] .btn:hover {{ filter: brightness(1.08); }}

    .tree{{
      /* key change: tree is the scroll container */
      overflow-y: auto;
      overflow-x: hidden;
      padding: 8px 8px 14px 8px;

      /* important for grid children so overflow works */
      min-height: 0;
    }}

    .node {{
      padding: 8px 10px;
      border-radius: 10px;
      cursor: pointer;
      user-select: none;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }}
    .node:hover {{ background: rgba(0,0,0,0.04); }}
    [data-theme="dark"] .node:hover,
    [data-theme="nord"] .node:hover {{ background: rgba(255,255,255,0.06); }}

    .node.selected {{ background: var(--selected); }}

    .path {{
      font-size: 12px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}

    .main {{
      /* key change: main is the scroll container */
      overflow-y: auto;
      overflow-x: hidden;
      min-height: 0;
    }}

    .content {{
      max-width: 980px;
      margin: 0 auto;
      padding: 20px 26px 80px 26px;
    }}

    .hint {{
      color: var(--muted);
      font-size: 13px;
    }}

    .dropzone {{
      margin-top: 10px;
      border: 1px dashed var(--border);
      border-radius: 14px;
      padding: 10px 12px;
      background: var(--bg);
    }}
    .dropzone.dragover {{
      outline: 2px solid var(--link);
      outline-offset: 2px;
    }}

    /* Markdown styling */
    h1,h2,h3,h4 {{ margin-top: 1.4em; }}
    p, li {{ line-height: 1.65; }}
    hr {{ border: 0; border-top: 1px solid var(--border); margin: 24px 0; }}
    blockquote {{
      margin: 16px 0;
      padding: 10px 14px;
      border-left: 4px solid var(--border);
      background: rgba(0,0,0,0.03);
      border-radius: 10px;
    }}
    [data-theme="dark"] blockquote,
    [data-theme="nord"] blockquote {{ background: rgba(255,255,255,0.06); }}

    code {{
      background: var(--codebg);
      padding: 2px 6px;
      border-radius: 8px;
      border: 1px solid var(--border);
      font-size: 0.95em;
    }}

    .codehilite {{
      margin: 14px 0;
      padding: 14px 16px;
      overflow: auto;
      background: var(--codebg);
      border: 1px solid var(--border);
      border-radius: 14px;
    }}
    .codehilite pre {{
      margin: 0;
      background: transparent;
      border: none;
      padding: 0;
    }}

    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 18px 0;
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
    }}
    th {{ background: rgba(0,0,0,0.03); }}
    [data-theme="dark"] th,
    [data-theme="nord"] th {{ background: rgba(255,255,255,0.06); }}

    .searchResults {{
      margin-top: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .searchGroups {{
      padding: 0 10px 14px 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .searchGroup {{
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--bg);
      overflow: hidden;
    }}

    .searchSummary {{
      cursor: pointer;
      list-style: none;
      padding: 10px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-weight: 700;
    }}

    .searchSummary::-webkit-details-marker {{ display: none; }}
    .searchSummary::before {{
      content: "▸";
      color: var(--muted);
      margin-right: 6px;
      transition: transform 120ms ease;
    }}
    .searchGroup[open] .searchSummary::before {{
      transform: rotate(90deg);
    }}

    .searchGroupBody {{
      padding: 0 10px 10px 10px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .result {{
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--bg);
      cursor: pointer;
    }}
    .result:hover {{
      filter: brightness(0.98);
    }}
    [data-theme="dark"] .result:hover,
    [data-theme="nord"] .result:hover {{
      filter: brightness(1.08);
    }}

    mark {{
      padding: 0 3px;
      border-radius: 6px;
      background: rgba(250, 204, 21, 0.35);
    }}

    .status {{
      font-size: 12px;
      margin-top: 8px;
    }}
    .status.ok {{ color: var(--ok); }}
    .status.bad {{ color: var(--danger); }}

    @media (max-width: 900px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ height: 50vh; }}
    }}
  </style>
</head>

<body data-theme="light">
  <div class="app">
    <div class="sidebar">
      <div class="topbar">
        <div>
          <div class="row">
            <input id="search" class="input" placeholder="Search filenames or full text…" />
          </div>
          <div class="row" style="margin-top:8px;">
            <input id="currentFile" class="input" readonly placeholder="Current file: none selected" />
          </div>
          <div class="dropzone" id="dropzone" title="Drop a PATH string here (e.g. /abs/path/to/docs)">
            <div style="font-weight:700;">Drop a folder path here</div>
            <div class="hint">Drag a path as text (from terminal / address bar), or paste below.</div>
            <div class="row" style="margin-top:8px;">
              <input id="rootPath" class="input" placeholder="e.g. /absolute/path/to/somepath/docs" />
              <button id="setRoot" class="btn">Set</button>
            </div>
            <div id="rootStatus" class="status hint"></div>
          </div>

          <div class="row" style="margin-top:10px;">
            <select id="theme" class="theme" title="Theme">
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="sepia">Sepia</option>
              <option value="nord">Nord</option>
            </select>
            <button id="clearSearch" class="btn" title="Clear search">Clear</button>
          </div>
        </div>
      </div>

      <div class="tree" id="tree"></div>
    </div>

    <div class="main">
      <div class="content" id="content">
        <h1>Docs Viewer</h1>
        <p class="hint">
          Set your docs folder (server-side), then click a file — or search full text.
        </p>
      </div>
    </div>
  </div>

<script>
  const treeEl = document.getElementById('tree');
  const contentEl = document.getElementById('content');
  const searchEl = document.getElementById('search');
  const clearBtn = document.getElementById('clearSearch');
  const currentFileEl = document.getElementById('currentFile');

  const themeEl = document.getElementById('theme');
  const dropzone = document.getElementById('dropzone');
  const rootPathEl = document.getElementById('rootPath');
  const setRootBtn = document.getElementById('setRoot');
  const rootStatusEl = document.getElementById('rootStatus');

  // Persist theme
  const savedTheme = localStorage.getItem('docs_theme') || 'light';
  document.documentElement.dataset.theme = savedTheme;

  themeEl.value = savedTheme;
  themeEl.addEventListener('change', () => {{
    document.documentElement.dataset.theme = themeEl.value;
    localStorage.setItem('docs_theme', themeEl.value);
  }});

  let TREE = null;
  let CURRENT_PATH = null;

  function isSearchActive() {{
    return !!searchEl.value.trim();
  }}

  function updateCurrentFileField(path) {{
    const p = (path || "").trim();
    currentFileEl.value = p ? p : "None selected";
  }}

  function escapeHtml(s) {{
    return s.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  }}

  function flattenFiles(node) {{
    let out = [];
    if (!node) return out;
    if (node.type === 'file') {{
      out.push({{
        name: node.name,
        path: node.path,
        display: node.path
      }});
      return out;
    }}
    for (const ch of (node.children || [])) {{
      out = out.concat(flattenFiles(ch));
    }}
    return out;
  }}

  function renderFileList(files) {{
    treeEl.innerHTML = "";
    if (!files || files.length === 0) {{
      treeEl.innerHTML = '<div class="hint" style="padding:12px">No files.</div>';
      return;
    }}
    for (const f of files) {{
      const div = document.createElement('div');
      div.className = 'node' + (f.path === CURRENT_PATH ? ' selected' : '');
      div.innerHTML = `
        <div style="min-width:0">
          <div style="font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${{escapeHtml(f.name)}}</div>
          <div class="path">${{escapeHtml(f.display)}}</div>
        </div>
        <div class="hint">md</div>
      `;
      div.onclick = () => loadFile(f.path);
      treeEl.appendChild(div);
    }}
  }}

  async function loadTree() {{
    const res = await fetch('/api/tree');
    TREE = await res.json();
    if (isSearchActive()) {{
      await doSearch(searchEl.value);
      return;
    }}
    renderFileList(flattenFiles(TREE));
  }}

  async function loadFile(path) {{
    CURRENT_PATH = path;
    updateCurrentFileField(path);
    if (!isSearchActive()) {{
      renderFileList(flattenFiles(TREE));
    }}

    const res = await fetch('/api/file?path=' + encodeURIComponent(path));
    if (!res.ok) {{
      contentEl.innerHTML = `<p class="hint">Failed to load file: <code>${{escapeHtml(path)}}</code></p>`;
      return;
    }}
    const data = await res.json();
    document.title = data.title ? (data.title + " — Docs Viewer") : "Docs Viewer";
    contentEl.innerHTML = data.html;
    localStorage.setItem('docs_last_path', path);
  }}

  function resolveMarkdownPath(basePath, href) {{
    if (!href) return null;
    const raw = href.trim();
    if (!raw) return null;
    if (raw.startsWith('#')) return null;
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(raw)) return null; // external scheme

    const hashIdx = raw.indexOf('#');
    const noHash = hashIdx >= 0 ? raw.slice(0, hashIdx) : raw;
    const queryIdx = noHash.indexOf('?');
    const clean = queryIdx >= 0 ? noHash.slice(0, queryIdx) : noHash;
    if (!clean) return null;

    const lower = clean.toLowerCase();
    if (!(lower.endsWith('.md') || lower.endsWith('.markdown') || lower.endsWith('.mdx'))) {{
      return null;
    }}

    let baseParts = [];
    if (basePath) {{
      baseParts = basePath.split('/').slice(0, -1);
    }}

    const inputParts = clean.split('/');
    const out = clean.startsWith('/') ? [] : baseParts;
    for (const part of inputParts) {{
      if (!part || part === '.') continue;
      if (part === '..') {{
        if (out.length === 0) return null;
        out.pop();
        continue;
      }}
      out.push(part);
    }}
    if (out.length === 0) return null;
    return out.join('/');
  }}

  contentEl.addEventListener('click', (e) => {{
    const link = e.target.closest('a[href]');
    if (!link) return;
    const resolved = resolveMarkdownPath(CURRENT_PATH, link.getAttribute('href') || "");
    if (!resolved) return;
    e.preventDefault();
    loadFile(resolved);
  }});

  function buildSearchGroup(title, results, openByDefault = false) {{
    const details = document.createElement('details');
    details.className = "searchGroup";
    details.open = openByDefault;

    const summary = document.createElement('summary');
    summary.className = "searchSummary";
    summary.innerHTML = `
      <span>${{escapeHtml(title)}}</span>
      <span class="hint">${{results.length}} match(es)</span>
    `;
    details.appendChild(summary);

    const body = document.createElement('div');
    body.className = "searchGroupBody";
    if (!results.length) {{
      body.innerHTML = '<div class="hint">No matches.</div>';
      details.appendChild(body);
      return details;
    }}

    for (const r of results) {{
      const div = document.createElement('div');
      div.className = "result";
      div.innerHTML = `
        <div style="font-weight:800;">${{escapeHtml(r.title)}}</div>
        <div class="path">${{escapeHtml(r.path)}}</div>
        ${{r.snippet ? `<div class="hint" style="margin-top:6px;">${{r.snippet}}</div>` : ''}}
      `;
      div.onclick = () => loadFile(r.path);
      body.appendChild(div);
    }}

    details.appendChild(body);
    return details;
  }}

  function renderSearchResults(fileResults, textResults, query) {{
    treeEl.innerHTML = "";
    const total = fileResults.length + textResults.length;
    const header = document.createElement('div');
    header.style.padding = "12px";
    header.innerHTML = `<div style="font-weight:800;">Results for "${{escapeHtml(query)}}"</div>
                        <div class="hint">${{total}} total match(es)</div>`;
    treeEl.appendChild(header);

    const wrap = document.createElement('div');
    wrap.className = "searchGroups";
    wrap.appendChild(buildSearchGroup("Filename matches", fileResults, true));
    wrap.appendChild(buildSearchGroup("Full-text matches", textResults, total === 0 ? true : false));
    treeEl.appendChild(wrap);
  }}

  let searchTimer = null;
  async function doSearch(q) {{
    q = q.trim();
    if (!q) {{
      await loadTree();
      return;
    }}
    const res = await fetch('/api/search?q=' + encodeURIComponent(q));
    if (!res.ok) return;
    const data = await res.json();
    const fileResults = data.file_results || [];
    const textResults = data.text_results || data.results || [];
    renderSearchResults(fileResults, textResults, q);
  }}

  searchEl.addEventListener('input', () => {{
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => doSearch(searchEl.value), 180);
  }});

  clearBtn.addEventListener('click', async () => {{
    searchEl.value = "";
    await loadTree();
  }});

  // Live reload via websocket
  function connectWS() {{
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${{proto}}://${{location.host}}/ws`);
    ws.onmessage = (ev) => {{
      if (ev.data === 'reload') {{
        // Refresh tree + current file (if any)
        loadTree().then(() => {{
          if (CURRENT_PATH) loadFile(CURRENT_PATH);
        }});
      }}
      if (ev.data === 'root_changed') {{
        loadTree().then(() => {{
          CURRENT_PATH = null;
          updateCurrentFileField("");
          contentEl.innerHTML = `<h1>Docs Viewer</h1><p class="hint">Docs folder changed. Select a file.</p>`;
        }});
      }}
    }};
    ws.onclose = () => setTimeout(connectWS, 700);
  }}

  // Set root (server-side)
  async function setRootPath(path) {{
    path = (path || "").trim();
    if (!path) return;

    rootStatusEl.className = "status hint";
    rootStatusEl.textContent = "Setting root…";

    const res = await fetch('/api/set_root', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ path }})
    }});

    const data = await res.json().catch(() => ({{}}));
    if (res.ok) {{
      rootStatusEl.className = "status ok";
      rootStatusEl.textContent = data.message || "Root updated.";
      localStorage.setItem('docs_root', path);
      searchEl.value = "";
      await loadTree();
    }} else {{
      rootStatusEl.className = "status bad";
      rootStatusEl.textContent = data.error || "Failed to set root.";
    }}
  }}

  setRootBtn.addEventListener('click', () => setRootPath(rootPathEl.value));
  rootPathEl.addEventListener('keydown', (e) => {{
    if (e.key === 'Enter') setRootPath(rootPathEl.value);
  }});

  // Drag & drop: accept TEXT paths
  dropzone.addEventListener('dragover', (e) => {{
    e.preventDefault();
    dropzone.classList.add('dragover');
  }});
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', async (e) => {{
    e.preventDefault();
    dropzone.classList.remove('dragover');

    const text = (e.dataTransfer && e.dataTransfer.getData('text/plain')) || "";
    if (text.trim()) {{
      rootPathEl.value = text.trim();
      await setRootPath(text.trim());
      return;
    }}

    // If user dropped actual files/folder, browsers typically won't give an absolute path.
    rootStatusEl.className = "status bad";
    rootStatusEl.textContent = "Drop a PATH as text (e.g. from terminal). Dropping folders directly usually hides the full path.";
  }});

  // Initial boot
  (async () => {{
    updateCurrentFileField("");
    const savedRoot = localStorage.getItem('docs_root');
    if (savedRoot) {{
      rootPathEl.value = savedRoot;
    }}
    await loadTree();

    const last = localStorage.getItem('docs_last_path');
    if (last) {{
      // Try to load it; ignore if missing
      loadFile(last).catch(() => {{}});
    }}
  }})();

  connectWS();
</script>
</body>
</html>
"""


@app.get("/api/tree")
async def api_tree() -> JSONResponse:
    root = await docs_root.get()
    root.mkdir(parents=True, exist_ok=True)
    return JSONResponse(content=build_tree(root))


@app.get("/api/file")
async def api_file(path: str) -> JSONResponse:
    root = await docs_root.get()
    p = safe_join(root, path)
    if not p:
        return JSONResponse(status_code=404, content={"error": "Not found"})

    text = p.read_text(encoding="utf-8", errors="replace")
    title = p.stem
    html = render_markdown(text, title)
    return JSONResponse(content={"title": title, "html": html})


@app.get("/api/search")
async def api_search(q: str) -> JSONResponse:
    q = (q or "").strip()
    if not q:
        return JSONResponse(content={"results": [], "file_results": [], "text_results": []})

    root = await docs_root.get()
    files = iter_markdown_files(root)

    q_lower = q.lower()
    file_results: List[Dict[str, Any]] = []
    text_results: List[Dict[str, Any]] = []
    for f in files:
        rel_path = safe_relpath(root, f)
        filename_haystack = f"{f.stem} {f.name} {rel_path}".lower()
        if q_lower in filename_haystack:
            file_results.append({
                "title": f.stem,
                "path": rel_path,
            })

        try:
            text = normalize(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue

        spans = search_in_text(text, q)
        if not spans:
            if len(file_results) >= 80:
                continue

        # Use first match for snippet
        if spans:
            snippet = make_snippet(text, spans[0])
            text_results.append({
                "title": f.stem,
                "path": rel_path,
                "snippet": snippet
            })

        if len(file_results) >= 80 and len(text_results) >= 80:
            break

    # Simple ranking: shorter path first (often more "root" docs), then title
    file_results.sort(key=lambda r: (len(r["path"]), r["path"].lower()))
    text_results.sort(key=lambda r: (len(r["path"]), r["path"].lower()))
    return JSONResponse(content={
        "results": text_results,
        "file_results": file_results,
        "text_results": text_results
    })


@app.post("/api/set_root")
async def api_set_root(payload: Dict[str, str] = Body(...)) -> JSONResponse:
    raw = (payload.get("path") or "").strip()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "Missing path"})

    new_root = Path(raw).expanduser().resolve()
    if not new_root.exists() or not new_root.is_dir():
        return JSONResponse(status_code=400, content={"error": "Path does not exist or is not a directory"})

    # Optional: enforce it ends with /docs (you asked for {somepath}/docs/)
    # Comment out if you want any folder allowed.
    if new_root.name.lower() != "docs":
        return JSONResponse(status_code=400, content={"error": "Folder must be named 'docs' (…/docs)"})


    await docs_root.set(new_root)
    await _start_watcher()
    await ws_manager.broadcast("root_changed")
    return JSONResponse(content={"message": f"Docs root set to: {str(new_root)}"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws_manager.connect(ws)
    try:
        while True:
            # Keepalive (client doesn't send, so we wait; some proxies may need ping later)
            await ws.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(ws)
    except Exception:
        await ws_manager.disconnect(ws)