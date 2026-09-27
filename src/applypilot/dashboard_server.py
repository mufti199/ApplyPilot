"""Live ApplyPilot dashboard served on localhost.

`applypilot dashboard --serve` runs this. Unlike the static dashboard, it reads
the database on every request, so jobs and scores appear while a pipeline run
is still going, and you can record your own status for each job.

Stdlib only (http.server). Binds to 127.0.0.1 and rejects requests whose Host
header is not local, so other sites can't reach it via DNS rebinding. State
changes need a custom header, which cross-site forms and simple requests can't
send.
"""

from __future__ import annotations

import json
import logging
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from applypilot.config import COVER_LETTER_DIR, get_resume_tracks, load_search_config
from applypilot.database import TRACKING_STATUSES, get_connection, init_db, set_tracking_status

log = logging.getLogger(__name__)

DEFAULT_PORT = 8765
REFRESH_SECONDS = 30
_WRITE_HEADER = "X-ApplyPilot"


# ---------------------------------------------------------------------------
# Data (kept separate from HTTP so it can be tested directly)
# ---------------------------------------------------------------------------

def dashboard_data(conn=None) -> dict:
    """Stats plus every scored job, without full descriptions (fetched on demand)."""
    if conn is None:
        conn = get_connection()

    def count(where: str) -> int:
        return conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {where}").fetchone()[0]

    stats = {
        "total": count("1=1"),
        "with_description": count("full_description IS NOT NULL"),
        "scored": count("fit_score IS NOT NULL"),
        "unscored": count("full_description IS NOT NULL AND fit_score IS NULL"),
        "strong": count("fit_score >= 7"),
        "cover_letters": count("cover_letter_path IS NOT NULL AND cover_letter_path != ''"),
        "tracked": count("tracking_status IS NOT NULL"),
        "duplicates": count("duplicate_of IS NOT NULL"),
        "excluded": count("excluded_reason IS NOT NULL AND duplicate_of IS NULL"),
    }
    rows = conn.execute("""
        SELECT rowid AS id, url, application_url, title, company, site, location, salary,
               fit_score, score_reasoning, resume_track, tracking_status, tracking_updated_at,
               cover_letter_path, discovered_at, scored_at, excluded_reason
        FROM jobs WHERE duplicate_of IS NULL AND (fit_score IS NOT NULL OR excluded_reason IS NOT NULL)
        ORDER BY fit_score DESC NULLS LAST, scored_at DESC
    """).fetchall()
    copies: dict[int, list] = {}
    for d in conn.execute("SELECT duplicate_of, site, url FROM jobs WHERE duplicate_of IS NOT NULL"):
        copies.setdefault(d["duplicate_of"], []).append({"site": d["site"], "url": d["url"]})

    jobs = []
    for r in rows:
        keywords, _, reasoning = (r["score_reasoning"] or "").partition("\n")
        jobs.append({
            "id": r["id"],
            "url": r["url"],
            "apply_url": r["application_url"] or r["url"],
            "title": r["title"] or "Untitled",
            "site": r["site"] or "",
            "company": r["company"] or "",
            "location": r["location"] or "",
            "salary": r["salary"] or "",
            "score": r["fit_score"] or 0,
            "excluded": r["excluded_reason"],
            "keywords": keywords.strip(),
            "reasoning": reasoning.strip(),
            "track": r["resume_track"],
            "status": r["tracking_status"],
            "status_at": r["tracking_updated_at"],
            "has_cover_letter": bool(_cover_pdf_path(r["cover_letter_path"])),
            "scored_at": r["scored_at"],
            "also_on": copies.get(r["id"], []),
        })
    return {"stats": stats, "jobs": jobs, "statuses": list(TRACKING_STATUSES)}


def job_description(job_id: int, conn=None) -> str | None:
    if conn is None:
        conn = get_connection()
    row = conn.execute("SELECT full_description FROM jobs WHERE rowid = ?", (job_id,)).fetchone()
    return row[0] if row else None


def resume_pdf(track: str) -> Path | None:
    """The configured PDF for a resume track, or None."""
    tracks = get_resume_tracks(load_search_config())
    entry = tracks.get(track)
    return entry["pdf"] if entry and entry["pdf"] else None


def cover_letter_pdf(job_id: int, conn=None) -> Path | None:
    """The job's cover letter PDF, only if it lives inside the cover letters folder."""
    if conn is None:
        conn = get_connection()
    row = conn.execute("SELECT cover_letter_path FROM jobs WHERE rowid = ?", (job_id,)).fetchone()
    return _cover_pdf_path(row[0]) if row else None


def _cover_pdf_path(txt_path: str | None) -> Path | None:
    if not txt_path:
        return None
    pdf = Path(txt_path).with_suffix(".pdf")
    try:
        pdf.resolve().relative_to(COVER_LETTER_DIR.resolve())
    except ValueError:
        return None
    return pdf if pdf.is_file() else None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ApplyPilotDashboard"

    def log_message(self, fmt, *args):  # route http.server logs through logging
        log.debug("dashboard: " + fmt, *args)

    # -- helpers --

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in ("127.0.0.1", "localhost")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "file not found")
            return
        self._send(HTTPStatus.OK, path.read_bytes(), "application/pdf")

    # -- routes --

    def do_GET(self) -> None:
        if not self._host_ok():
            self._error(HTTPStatus.FORBIDDEN, "local access only")
            return
        path = self.path.split("?")[0]
        try:
            if path == "/":
                self._send(HTTPStatus.OK, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/data":
                self._json(dashboard_data())
            elif m := re.fullmatch(r"/api/job/(\d+)/description", path):
                desc = job_description(int(m.group(1)))
                if desc is None:
                    self._error(HTTPStatus.NOT_FOUND, "job not found")
                else:
                    self._json({"description": desc})
            elif m := re.fullmatch(r"/files/resume/([a-z0-9_-]+)", path):
                self._file(resume_pdf(m.group(1)))
            elif m := re.fullmatch(r"/files/cover/(\d+)", path):
                self._file(cover_letter_pdf(int(m.group(1))))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as e:  # keep the server up; report the failure
            log.exception("Dashboard request failed: %s", path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def do_POST(self) -> None:
        if not self._host_ok() or self.headers.get(_WRITE_HEADER) != "1":
            self._error(HTTPStatus.FORBIDDEN, "forbidden")
            return
        m = re.fullmatch(r"/api/job/(\d+)/status", self.path)
        if not m:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            status = body.get("status")
            if status is not None and not isinstance(status, str):
                raise ValueError("status must be a string or null")
            job = set_tracking_status(m.group(1), status)
        except (ValueError, json.JSONDecodeError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
            return
        except LookupError as e:
            self._error(HTTPStatus.NOT_FOUND, str(e))
            return
        log.info("Dashboard set status %s for job %s", status or "cleared", job["id"])
        self._json({"id": job["id"], "status": status})


def serve(port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Run the live dashboard until Ctrl+C."""
    init_db()  # adds any new columns (e.g. tracking_status) to older databases
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://127.0.0.1:{port}/"
    log.info("Live dashboard at %s", url)
    print(f"Live dashboard: {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ---------------------------------------------------------------------------
# Page (static shell; all data comes from /api/data)
# ---------------------------------------------------------------------------

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ApplyPilot Live</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 2rem; }
  h1 { font-size: 1.6rem; margin-bottom: .25rem; }
  .sub { color: #94a3b8; font-size: .85rem; margin-bottom: 1.5rem; }
  .live { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%;
          background: #10b981; margin-right: .35rem; }
  .live.off { background: #64748b; } .live.err { background: #ef4444; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .75rem; margin-bottom: 1.5rem; }
  .stat { background: #1e293b; border-radius: 10px; padding: 1rem; }
  .stat b { font-size: 1.6rem; display: block; } .stat span { color: #94a3b8; font-size: .8rem; }
  .filters { background: #1e293b; border-radius: 10px; padding: 1rem; margin-bottom: 1.25rem;
             display: flex; flex-wrap: wrap; gap: .5rem 1rem; align-items: center; }
  .group { display: flex; gap: .35rem; align-items: center; flex-wrap: wrap; }
  .label { color: #94a3b8; font-size: .8rem; font-weight: 600; }
  button, select, input { font: inherit; font-size: .8rem; }
  .fbtn { background: #334155; border: 0; color: #cbd5e1; padding: .35rem .7rem; border-radius: 6px; cursor: pointer; }
  .fbtn.on { background: #60a5fa; color: #0f172a; font-weight: 600; }
  input[type=text] { background: #334155; border: 1px solid #475569; color: #e2e8f0; padding: .35rem .7rem; border-radius: 6px; width: 200px; }
  label.chk { color: #cbd5e1; font-size: .8rem; display: flex; gap: .3rem; align-items: center; cursor: pointer; }
  .count { color: #94a3b8; font-size: .85rem; margin-bottom: .75rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 1rem; }
  .card { background: #1e293b; border-radius: 10px; padding: 1rem; border-left: 3px solid #334155; }
  .card.s9, .card.s10 { border-left-color: #10b981; } .card.s8 { border-left-color: #34d399; }
  .card.s7 { border-left-color: #60a5fa; } .card.s6, .card.s5 { border-left-color: #f59e0b; }
  .card.done { opacity: .6; }
  .head { display: flex; gap: .5rem; align-items: flex-start; margin-bottom: .45rem; }
  .pill { min-width: 1.6rem; height: 1.6rem; border-radius: 6px; color: #0f172a; font-weight: 700;
          display: inline-flex; align-items: center; justify-content: center; font-size: .8rem; flex-shrink: 0; }
  .title { color: #e2e8f0; font-weight: 600; text-decoration: none; }
  .title:hover { color: #60a5fa; }
  .num { color: #64748b; font-size: .75rem; margin-left: auto; flex-shrink: 0; }
  .tags { display: flex; flex-wrap: wrap; gap: .35rem; margin-bottom: .45rem; }
  .tag { font-size: .72rem; padding: .12rem .45rem; border-radius: 4px; background: #334155; color: #94a3b8; }
  .tag.software { background: #312e81; color: #c7d2fe; } .tag.devops { background: #064e3b; color: #6ee7b7; }
  .tag.loc { background: #1e3a5f; color: #93c5fd; } .tag.sal { background: #3f3f46; color: #fde68a; }
  .tag.status { background: #7c2d12; color: #fed7aa; }
  .tag.co { background: #475569; color: #f1f5f9; font-weight: 600; }
  .excl { font-size: .75rem; color: #fca5a5; margin-bottom: .3rem; }
  .also { font-size: .75rem; color: #94a3b8; margin-top: .5rem; } .also a { color: #93c5fd; }
  .kw { font-size: .75rem; color: #10b981; margin-bottom: .3rem; }
  .why { font-size: .75rem; color: #94a3b8; font-style: italic; margin-bottom: .6rem; line-height: 1.4; }
  .actions { display: flex; flex-wrap: wrap; gap: .4rem; align-items: center; }
  .link { font-size: .78rem; color: #60a5fa; text-decoration: none; padding: .25rem .6rem;
          border: 1px solid #60a5fa44; border-radius: 6px; }
  .link:hover { background: #60a5fa22; } .link.off { color: #64748b; border-color: #33415588; pointer-events: none; }
  select { background: #334155; color: #e2e8f0; border: 1px solid #475569; border-radius: 6px; padding: .25rem .4rem; }
  details { margin-top: .6rem; } summary { color: #60a5fa; font-size: .78rem; cursor: pointer; }
  .desc { font-size: .78rem; color: #cbd5e1; line-height: 1.55; margin-top: .4rem; padding: .7rem;
          background: #0f172a; border-radius: 8px; max-height: 380px; overflow-y: auto; white-space: pre-wrap; }
  .toast { position: fixed; bottom: 1rem; right: 1rem; background: #ef4444; color: white; padding: .6rem 1rem;
           border-radius: 8px; font-size: .85rem; display: none; }
  @media (max-width: 700px) { body { padding: 1rem; } .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>ApplyPilot Live</h1>
<p class="sub"><span id="dot" class="live off"></span><span id="updated">Loading…</span>
  &middot; refreshes every __REFRESH__s &middot; <label class="chk" style="display:inline-flex"><input type="checkbox" id="auto" checked> auto-refresh</label></p>

<div class="stats" id="stats"></div>

<div class="filters">
  <div class="group"><span class="label">Score</span>
    <button class="fbtn" data-min="1">All</button><button class="fbtn on" data-min="5">5+</button>
    <button class="fbtn" data-min="7">7+</button><button class="fbtn" data-min="8">8+</button></div>
  <div class="group"><span class="label">Track</span>
    <button class="fbtn on" data-track="">All</button><button class="fbtn" data-track="software">Software</button>
    <button class="fbtn" data-track="devops">DevOps</button></div>
  <div class="group"><span class="label">Status</span>
    <select id="statusFilter"><option value="open">Not applied / skipped</option><option value="">Any</option>
      <option value="none">Untracked</option></select></div>
  <div class="group"><label class="chk"><input type="checkbox" id="showExcluded"> Show excluded</label></div>
  <div class="group"><input type="text" id="search" placeholder="Search title, company, location…"></div>
</div>

<div class="count" id="count"></div>
<div class="grid" id="grid"></div>
<div class="toast" id="toast"></div>

<script>
const state = { min: 5, track: "", status: "open", q: "", data: null, open: new Set(), showExcluded: false };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const color = (s) => s >= 7 ? "#10b981" : s >= 5 ? "#f59e0b" : "#ef4444";

function toast(msg) { const t = $("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(toast.h); toast.h = setTimeout(() => t.style.display = "none", 4000); }

async function load() {
  try {
    const r = await fetch("/api/data", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    state.data = await r.json();
    $("dot").className = "live"; $("updated").textContent = "Updated " + new Date().toLocaleTimeString();
    render();
  } catch (e) { $("dot").className = "live err"; $("updated").textContent = "Update failed: " + e.message; }
}

function renderStats(s) {
  const items = [["Total jobs", s.total], ["With description", s.with_description], ["Scored", s.scored],
    ["Waiting to score", s.unscored], ["Strong fit (7+)", s.strong], ["Cover letters", s.cover_letters],
    ["Tracked by you", s.tracked], ["Duplicates grouped", s.duplicates], ["Excluded (not permanent FT)", s.excluded]];
  $("stats").innerHTML = items.map(([l, v]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
}

function visible(j) {
  if (j.excluded) return state.showExcluded;
  if (j.score < state.min) return false;
  if (state.track && j.track !== state.track) return false;
  if (state.status === "open" && (j.status === "applied" || j.status === "skipped")) return false;
  if (state.status === "none" && j.status) return false;
  if (state.status && !["open", "none"].includes(state.status) && j.status !== state.status) return false;
  if (state.q) { const hay = (j.title + " " + j.company + " " + j.site + " " + j.location + " " + j.keywords).toLowerCase();
    if (!hay.includes(state.q)) return false; }
  return true;
}

function card(j, statuses) {
  const opts = ['<option value="">— status —</option>'].concat(statuses.map(s =>
    `<option value="${s}" ${j.status === s ? "selected" : ""}>${s}</option>`)).join("");
  const done = j.status === "applied" || j.status === "skipped" || j.status === "rejected";
  const resume = j.track ? `<a class="link" target="_blank" href="/files/resume/${esc(j.track)}">Resume (${esc(j.track)})</a>` : "";
  const cover = `<a class="link ${j.has_cover_letter ? "" : "off"}" target="_blank" href="/files/cover/${j.id}">${j.has_cover_letter ? "Cover letter" : "No cover letter yet"}</a>`;
  return `<div class="card s${j.score} ${done ? "done" : ""}" data-id="${j.id}">
    <div class="head"><span class="pill" style="background:${j.excluded ? "#64748b" : color(j.score)}">${j.score || "–"}</span>
      <a class="title" target="_blank" href="${esc(j.url)}">${esc(j.title)}</a><span class="num">#${j.id}</span></div>
    <div class="tags">${j.track ? `<span class="tag ${esc(j.track)}">${esc(j.track)}</span>` : ""}
      ${j.company ? `<span class="tag co">${esc(j.company)}</span>` : ""}<span class="tag">${esc(j.site)}</span>${j.location ? `<span class="tag loc">${esc(j.location)}</span>` : ""}
      ${j.salary ? `<span class="tag sal">${esc(j.salary)}</span>` : ""}
      ${j.status ? `<span class="tag status">${esc(j.status)}</span>` : ""}</div>
    ${j.excluded ? `<div class="excl">Excluded: ${esc(j.excluded)}</div>` : ""}
    ${j.keywords ? `<div class="kw">${esc(j.keywords)}</div>` : ""}
    ${j.reasoning ? `<div class="why">${esc(j.reasoning)}</div>` : ""}
    <div class="actions"><a class="link" target="_blank" href="${esc(j.apply_url)}">Open posting</a>${resume}${cover}
      <select class="status" data-id="${j.id}">${opts}</select></div>
    ${j.also_on.length ? `<div class="also">Also posted on: ${j.also_on.map(a => `<a target="_blank" href="${esc(a.url)}">${esc(a.site)}</a>`).join(", ")}</div>` : ""}
    <details data-id="${j.id}" ${state.open.has(j.id) ? "open" : ""}><summary>Full description</summary><div class="desc">…</div></details>
  </div>`;
}

function render() {
  const d = state.data; if (!d) return;
  renderStats(d.stats);
  const shown = d.jobs.filter(visible);
  $("count").textContent = `Showing ${shown.length} of ${d.jobs.length} jobs`;
  $("grid").innerHTML = shown.map(j => card(j, d.statuses)).join("");
  document.querySelectorAll("details[open]").forEach(loadDesc);
}

async function loadDesc(el) {
  const box = el.querySelector(".desc"); if (box.dataset.loaded) return;
  try { const r = await fetch(`/api/job/${el.dataset.id}/description`);
    const d = await r.json(); box.textContent = d.description || "(no description)"; box.dataset.loaded = "1";
  } catch (e) { box.textContent = "Could not load description"; }
}

async function setStatus(id, status) {
  try {
    const r = await fetch(`/api/job/${id}/status`, { method: "POST",
      headers: { "Content-Type": "application/json", "X-ApplyPilot": "1" },
      body: JSON.stringify({ status: status || null }) });
    if (!r.ok) throw new Error((await r.json()).error || ("HTTP " + r.status));
    const job = state.data.jobs.find(j => j.id === id); if (job) job.status = status || null;
    render();
  } catch (e) { toast("Could not save status: " + e.message); load(); }
}

document.addEventListener("click", (e) => {
  const b = e.target.closest(".fbtn"); if (!b) return;
  const key = "min" in b.dataset ? "min" : "track";
  b.parentElement.querySelectorAll(".fbtn").forEach(x => x.classList.toggle("on", x === b));
  state[key] = key === "min" ? Number(b.dataset.min) : b.dataset.track; render();
});
document.addEventListener("change", (e) => {
  if (e.target.matches("select.status")) setStatus(Number(e.target.dataset.id), e.target.value);
  if (e.target.id === "statusFilter") { state.status = e.target.value; render(); }
  if (e.target.id === "showExcluded") { state.showExcluded = e.target.checked; render(); }
});
document.addEventListener("toggle", (e) => {
  if (e.target.tagName !== "DETAILS") return;
  const id = Number(e.target.dataset.id);
  if (e.target.open) { state.open.add(id); loadDesc(e.target); } else state.open.delete(id);
}, true);
$("search").addEventListener("input", (e) => { state.q = e.target.value.toLowerCase(); render(); });

load();
setInterval(() => { if ($("auto").checked && !document.hidden && !document.activeElement.matches("select.status")) load(); },
            __REFRESH__ * 1000);
</script>
</body>
</html>
""".replace("__REFRESH__", str(REFRESH_SECONDS))
