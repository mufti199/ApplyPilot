"""ApplyPilot database layer: schema, migrations, stats, and connection helpers.

Single source of truth for the jobs table schema. All columns from every
pipeline stage are created up front so any stage can run independently
without migration ordering issues.
"""

import hashlib
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from applypilot.config import DB_PATH

# Thread-local connection storage — each thread gets its own connection
# (required for SQLite thread safety with parallel workers)
_local = threading.local()


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Get a thread-local cached SQLite connection with WAL mode enabled.

    Each thread gets its own connection (required for SQLite thread safety).
    Connections are cached and reused within the same thread.

    Args:
        db_path: Override the default DB_PATH. Useful for testing.

    Returns:
        sqlite3.Connection configured with WAL mode and row factory.
    """
    path = str(db_path or DB_PATH)

    if not hasattr(_local, 'connections'):
        _local.connections = {}

    conn = _local.connections.get(path)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.ProgrammingError:
            pass

    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    _local.connections[path] = conn
    return conn


def close_connection(db_path: Path | str | None = None) -> None:
    """Close the cached connection for the current thread."""
    path = str(db_path or DB_PATH)
    if hasattr(_local, 'connections'):
        conn = _local.connections.pop(path, None)
        if conn is not None:
            conn.close()


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create the full jobs table with all columns from every pipeline stage.

    This is idempotent -- safe to call on every startup. Uses CREATE TABLE IF NOT EXISTS
    so it won't destroy existing data.

    Schema columns by stage:
      - Discovery:  url, title, salary, description, location, site, strategy, discovered_at
      - Enrichment: full_description, application_url, detail_scraped_at, detail_error
      - Scoring:    fit_score, score_reasoning, scored_at
      - Tailoring:  tailored_resume_path, tailored_at, tailor_attempts
      - Cover:      cover_letter_path, cover_letter_at, cover_attempts
      - Apply:      applied_at, apply_status, apply_error, apply_attempts,
                   agent_id, last_attempted_at, apply_duration_ms, apply_task_id,
                   verification_confidence

    Args:
        db_path: Override the default DB_PATH.

    Returns:
        sqlite3.Connection with the schema initialized.
    """
    path = db_path or DB_PATH

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            -- Discovery stage (smart_extract / job_search)
            url                   TEXT PRIMARY KEY,
            title                 TEXT,
            salary                TEXT,
            description           TEXT,
            location              TEXT,
            site                  TEXT,
            strategy              TEXT,
            discovered_at         TEXT,

            -- Enrichment stage (detail_scraper)
            full_description      TEXT,
            application_url       TEXT,
            detail_scraped_at     TEXT,
            detail_error          TEXT,

            -- Scoring stage (job_scorer)
            fit_score             INTEGER,
            score_reasoning       TEXT,
            scored_at             TEXT,

            -- Tailoring stage (resume tailor)
            tailored_resume_path  TEXT,
            tailored_at           TEXT,
            tailor_attempts       INTEGER DEFAULT 0,

            -- Cover letter stage
            cover_letter_path     TEXT,
            cover_letter_at       TEXT,
            cover_attempts        INTEGER DEFAULT 0,

            -- Application stage
            applied_at            TEXT,
            apply_status          TEXT,
            apply_error           TEXT,
            apply_attempts        INTEGER DEFAULT 0,
            agent_id              TEXT,
            last_attempted_at     TEXT,
            apply_duration_ms     INTEGER,
            apply_task_id         TEXT,
            verification_confidence TEXT
        )
    """)
    conn.commit()

    # Run migrations for any columns added after initial schema
    ensure_columns(conn)

    return conn


# Complete column registry: column_name -> SQL type with optional default.
# This is the single source of truth. Adding a column here is all that's needed
# for it to appear in both new databases and migrated ones.
_ALL_COLUMNS: dict[str, str] = {
    # Discovery
    "url": "TEXT PRIMARY KEY",
    "title": "TEXT",
    "salary": "TEXT",
    "description": "TEXT",
    "location": "TEXT",
    "site": "TEXT",
    "company": "TEXT",
    "strategy": "TEXT",
    "discovered_at": "TEXT",
    # Enrichment
    "full_description": "TEXT",
    "application_url": "TEXT",
    "detail_scraped_at": "TEXT",
    "detail_error": "TEXT",
    # Scoring
    "fit_score": "INTEGER",
    "score_reasoning": "TEXT",
    "scored_at": "TEXT",
    "resume_track": "TEXT",
    # Tailoring
    "tailored_resume_path": "TEXT",
    "tailored_at": "TEXT",
    "tailor_attempts": "INTEGER DEFAULT 0",
    # Cover letter
    "cover_letter_path": "TEXT",
    "cover_letter_at": "TEXT",
    "cover_attempts": "INTEGER DEFAULT 0",
    # Application
    "applied_at": "TEXT",
    "apply_status": "TEXT",
    "apply_error": "TEXT",
    "apply_attempts": "INTEGER DEFAULT 0",
    "agent_id": "TEXT",
    "last_attempted_at": "TEXT",
    "apply_duration_ms": "INTEGER",
    "apply_task_id": "TEXT",
    "verification_confidence": "TEXT",
    # Manual tracking (set by the user, not the auto-apply agent)
    "tracking_status": "TEXT",
    "tracking_updated_at": "TEXT",
    # Same job posted more than once (other board or repost): rowid of the main copy
    "dedupe_key": "TEXT",
    "duplicate_of": "INTEGER",
}

# Jobs the pipeline should work on (duplicates are handled through their main copy).
ACTIVE_JOB_SQL = "duplicate_of IS NULL"

# Statuses a user can record for a job they handle themselves.
TRACKING_STATUSES = ("applied", "skipped", "interviewing", "rejected", "offer", "cold-call")


def ensure_columns(conn: sqlite3.Connection | None = None) -> list[str]:
    """Add any missing columns to the jobs table (forward migration).

    Reads the current table schema via PRAGMA table_info and compares against
    the full column registry. Any missing columns are added with ALTER TABLE.

    This makes it safe to upgrade the database from any previous version --
    columns are only added, never removed or renamed.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        List of column names that were added (empty if schema was already current).
    """
    if conn is None:
        conn = get_connection()

    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    added = []

    for col, dtype in _ALL_COLUMNS.items():
        if col not in existing:
            # PRIMARY KEY columns can't be added via ALTER TABLE, but url
            # is always created with the table itself so this is safe
            if "PRIMARY KEY" in dtype:
                continue
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {dtype}")
            added.append(col)

    if added:
        conn.commit()

    return added


def get_stats(conn: sqlite3.Connection | None = None) -> dict:
    """Return job counts by pipeline stage.

    Provides a snapshot of how many jobs are at each stage, useful for
    dashboard display and pipeline progress tracking.

    Args:
        conn: Database connection. Uses get_connection() if None.

    Returns:
        Dictionary with keys:
            total, by_site, pending_detail, with_description,
            scored, unscored, tailored, untailored_eligible,
            with_cover_letter, applied, score_distribution
    """
    if conn is None:
        conn = get_connection()

    stats: dict = {}

    # Total jobs
    stats["total"] = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    # By site breakdown
    rows = conn.execute(
        "SELECT site, COUNT(*) as cnt FROM jobs GROUP BY site ORDER BY cnt DESC"
    ).fetchall()
    stats["by_site"] = [(row[0], row[1]) for row in rows]

    # Enrichment stage
    stats["pending_detail"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_scraped_at IS NULL"
    ).fetchone()[0]

    stats["with_description"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL"
    ).fetchone()[0]

    stats["detail_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE detail_error IS NOT NULL"
    ).fetchone()[0]

    # Scoring stage
    stats["scored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL"
    ).fetchone()[0]

    stats["unscored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE full_description IS NOT NULL AND fit_score IS NULL"
    ).fetchone()[0]

    # Score distribution
    dist_rows = conn.execute(
        "SELECT fit_score, COUNT(*) as cnt FROM jobs "
        "WHERE fit_score IS NOT NULL "
        "GROUP BY fit_score ORDER BY fit_score DESC"
    ).fetchall()
    stats["score_distribution"] = [(row[0], row[1]) for row in dist_rows]

    # Tailoring stage
    stats["tailored"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL"
    ).fetchone()[0]

    stats["untailored_eligible"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE fit_score >= 7 AND full_description IS NOT NULL "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    stats["tailor_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(tailor_attempts, 0) >= 5 "
        "AND tailored_resume_path IS NULL"
    ).fetchone()[0]

    # Cover letter stage
    stats["with_cover_letter"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE cover_letter_path IS NOT NULL"
    ).fetchone()[0]

    stats["cover_exhausted"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE COALESCE(cover_attempts, 0) >= 5 "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '')"
    ).fetchone()[0]

    # Application stage
    stats["applied"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE applied_at IS NOT NULL"
    ).fetchone()[0]

    stats["apply_errors"] = conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE apply_error IS NOT NULL"
    ).fetchone()[0]

    stats["ready_to_apply"] = conn.execute(
        "SELECT COUNT(*) FROM jobs "
        "WHERE tailored_resume_path IS NOT NULL "
        "AND applied_at IS NULL "
        "AND application_url IS NOT NULL"
    ).fetchone()[0]

    return stats


def store_jobs(conn: sqlite3.Connection, jobs: list[dict],
               site: str, strategy: str) -> tuple[int, int]:
    """Store discovered jobs, skipping duplicates by URL.

    Args:
        conn: Database connection.
        jobs: List of job dicts with keys: url, title, salary, description, location.
        site: Source site name (e.g. "RemoteOK", "Dice").
        strategy: Extraction strategy used (e.g. "json_ld", "api_response", "css_selectors").

    Returns:
        Tuple of (new_count, duplicate_count).
    """
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    existing = 0

    for job in jobs:
        url = job.get("url")
        if not url:
            continue
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, salary, description, location, site, strategy, discovered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (url, job.get("title"), job.get("salary"), job.get("description"),
                 job.get("location"), site, strategy, now),
            )
            new += 1
        except sqlite3.IntegrityError:
            existing += 1

    conn.commit()
    return new, existing


_DEDUPE_MIN_DESCRIPTION = 200  # shorter descriptions are too generic to match on
_DEDUPE_PREFIX_CHARS = 400


def dedupe_key(title: str | None, description: str | None) -> str | None:
    """Fingerprint for spotting the same job posted twice: title + description start.

    Returns None when the description is too short to match safely.
    """
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    desc = norm(description)
    if len(desc) < _DEDUPE_MIN_DESCRIPTION or not norm(title):
        return None
    digest = hashlib.sha1(desc[:_DEDUPE_PREFIX_CHARS].encode("utf-8")).hexdigest()[:16]
    return f"{norm(title)}|{digest}"


def mark_duplicates(conn: sqlite3.Connection | None = None) -> int:
    """Group jobs that are the same posting (other board or repost).

    The main copy is one already scored if any, else the earliest stored; the
    others get duplicate_of = its rowid. Idempotent. Returns how many jobs are
    currently marked as duplicates.
    """
    if conn is None:
        conn = get_connection()
    rows = conn.execute(
        "SELECT rowid, title, full_description, fit_score, dedupe_key, duplicate_of FROM jobs"
    ).fetchall()

    groups: dict[str, list] = {}
    for r in rows:
        key = r["dedupe_key"] or dedupe_key(r["title"], r["full_description"])
        if key is None:
            continue
        if r["dedupe_key"] is None:
            conn.execute("UPDATE jobs SET dedupe_key = ? WHERE rowid = ?", (key, r["rowid"]))
        groups.setdefault(key, []).append(r)

    marked = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        main = min(members, key=lambda r: (r["fit_score"] is None, r["rowid"]))
        for m in members:
            target = None if m is main else main["rowid"]
            if m["duplicate_of"] != target:
                conn.execute("UPDATE jobs SET duplicate_of = ? WHERE rowid = ?", (target, m["rowid"]))
            marked += target is not None
    conn.commit()
    return marked


def find_job(ref: str, conn: sqlite3.Connection | None = None) -> dict | None:
    """Look up a job by its number (SQLite rowid) or by its URL / application URL."""
    if conn is None:
        conn = get_connection()
    ref = ref.strip()
    if ref.isdigit():
        row = conn.execute("SELECT rowid AS id, * FROM jobs WHERE rowid = ?", (int(ref),)).fetchone()
    else:
        row = conn.execute(
            "SELECT rowid AS id, * FROM jobs WHERE url = ? OR application_url = ? LIMIT 1", (ref, ref),
        ).fetchone()
    return dict(row) if row else None


def set_tracking_status(ref: str, status: str | None,
                        conn: sqlite3.Connection | None = None) -> dict:
    """Record the user's own status for a job (None clears it).

    Marking 'applied' also sets applied_at (if unset) so pipeline stats count it.

    Returns:
        The job row as it was found (with its id).

    Raises:
        ValueError: unknown status.
        LookupError: no job matches ref.
    """
    if status is not None and status not in TRACKING_STATUSES:
        raise ValueError(f"Unknown status {status!r}. Valid: {', '.join(TRACKING_STATUSES)}")
    if conn is None:
        conn = get_connection()
    job = find_job(ref, conn)
    if job is None:
        raise LookupError(f"No job found for {ref!r} (use the job number or its exact URL)")

    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE jobs SET tracking_status = ?, tracking_updated_at = ? WHERE rowid = ?",
        (status, now if status else None, job["id"]),
    )
    if status == "applied":
        conn.execute("UPDATE jobs SET applied_at = COALESCE(applied_at, ?) WHERE rowid = ?", (now, job["id"]))
    conn.commit()
    return job


def cover_letter_pending_where(min_score: int, max_attempts: int, tailoring: bool,
                               tracks: list[str]) -> tuple[str, list]:
    """WHERE clause (and params) for jobs that still need a cover letter.

    With tailoring on, a job also needs its tailored resume first. With more
    than one resume track, the job's resume_track must be one of them (jobs
    scored before tracks existed are left out until re-scored).
    """
    where = (
        "fit_score >= ? AND full_description IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        f"AND COALESCE(cover_attempts, 0) < ? AND {ACTIVE_JOB_SQL}"
    )
    params: list = [min_score, max_attempts]
    if tailoring:
        where += " AND tailored_resume_path IS NOT NULL"
    if len(tracks) > 1:
        where += f" AND resume_track IN ({','.join('?' * len(tracks))})"
        params.extend(tracks)
    return where, params


def preferred_location_order(patterns: list[str]) -> tuple[str, list[str]]:
    """Build an ORDER BY term that puts jobs in preferred locations first.

    Returns (sql, params). The SQL evaluates to 0 for a job whose location
    contains any pattern (case-insensitive) and 1 otherwise. With no
    patterns it is NULL, so ordering is unchanged (a bare integer would be
    read by SQLite as a column position).
    """
    if not patterns:
        return "NULL", []
    escaped = [p.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") for p in patterns]
    clauses = " OR ".join("LOWER(COALESCE(location, '')) LIKE ? ESCAPE '\\'" for _ in escaped)
    return f"(CASE WHEN {clauses} THEN 0 ELSE 1 END)", [f"%{p}%" for p in escaped]


def get_jobs_by_stage(conn: sqlite3.Connection | None = None,
                      stage: str = "discovered",
                      min_score: int | None = None,
                      limit: int = 100,
                      preferred_locations: list[str] | None = None) -> list[dict]:
    """Fetch jobs filtered by pipeline stage.

    Args:
        conn: Database connection. Uses get_connection() if None.
        stage: One of "discovered", "enriched", "scored", "tailored", "applied".
        min_score: Minimum fit_score filter (only relevant for scored+ stages).
        limit: Maximum number of rows to return.
        preferred_locations: Location patterns that win ties on fit_score.

    Returns:
        List of job dicts.
    """
    if conn is None:
        conn = get_connection()

    conditions = {
        "discovered": "1=1",
        "pending_detail": "detail_scraped_at IS NULL",
        "enriched": "full_description IS NOT NULL",
        "pending_score": f"full_description IS NOT NULL AND fit_score IS NULL AND {ACTIVE_JOB_SQL}",
        "scored": "fit_score IS NOT NULL",
        "pending_tailor": (
            "fit_score >= ? AND full_description IS NOT NULL "
            "AND tailored_resume_path IS NULL AND COALESCE(tailor_attempts, 0) < 5 "
            f"AND {ACTIVE_JOB_SQL}"
        ),
        "tailored": "tailored_resume_path IS NOT NULL",
        "pending_apply": (
            "tailored_resume_path IS NOT NULL AND applied_at IS NULL "
            "AND application_url IS NOT NULL"
        ),
        "applied": "applied_at IS NOT NULL",
    }

    where = conditions.get(stage, "1=1")
    params: list = []

    if "?" in where and min_score is not None:
        params.append(min_score)
    elif "?" in where:
        params.append(7)  # default min_score

    if min_score is not None and "fit_score" not in where and stage in ("scored", "tailored", "applied"):
        where += " AND fit_score >= ?"
        params.append(min_score)

    loc_order, loc_params = preferred_location_order(preferred_locations or [])
    params.extend(loc_params)

    query = (
        f"SELECT * FROM jobs WHERE {where} "
        f"ORDER BY fit_score DESC NULLS LAST, {loc_order}, discovered_at DESC"
    )
    if limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()

    # Convert sqlite3.Row objects to dicts
    if rows:
        columns = rows[0].keys()
        return [dict(zip(columns, row)) for row in rows]
    return []
