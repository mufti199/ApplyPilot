"""Cover letter generation: LLM-powered, profile-driven, with validation.

Generates concise, engineering-voice cover letters tailored to specific job
postings. All personal data (name, skills, achievements) comes from the user's
profile at runtime. No hardcoded personal information.
"""

import hashlib
import logging
import re
import time
from datetime import UTC, datetime

from applypilot.config import (
    COVER_LETTER_DIR,
    get_preferred_locations,
    get_resume_tracks,
    get_tailor_resumes,
    load_profile,
    load_search_config,
)
from applypilot.database import cover_letter_pending_where, get_connection, preferred_location_order
from applypilot.llm import DailyQuotaExceeded, get_client
from applypilot.scoring.validator import (
    BANNED_WORDS,
    LLM_LEAK_PHRASES,
    sanitize_text,
    validate_cover_letter,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


# ── Prompt Builder (profile-driven) ──────────────────────────────────────

def _build_cover_letter_prompt(profile: dict) -> str:
    """Build the cover letter system prompt from the user's profile.

    All personal data, skills, and sign-off name come from the profile.
    """
    personal = profile.get("personal", {})
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Preferred name for the sign-off (falls back to full name)
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")

    # Flatten all allowed skills
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "the tools listed in the resume"

    # Real metrics from resume_facts
    real_metrics = resume_facts.get("real_metrics", [])
    preserved_projects = resume_facts.get("preserved_projects", [])

    # Build achievement examples for the prompt
    projects_hint = ""
    if preserved_projects:
        projects_hint = f"\nKnown projects to reference: {', '.join(preserved_projects)}"

    metrics_hint = ""
    if real_metrics:
        metrics_hint = f"\nReal metrics to use: {', '.join(real_metrics)}"

    # Build the full banned list from the validator so the prompt stays in sync
    # with what will actually be rejected — the validator checks all of these.
    all_banned = ", ".join(f'"{w}"' for w in BANNED_WORDS)
    leak_banned = ", ".join(f'"{p}"' for p in LLM_LEAK_PHRASES)

    return f"""Write a cover letter for {sign_off_name}. The goal is to get an interview.

STRUCTURE: 3 short paragraphs. Under 250 words. Every sentence must earn its place.

PARAGRAPH 1 (2-3 sentences): Open with a specific thing YOU built that solves THEIR problem. Not "I'm excited about this role." Not "This role aligns with my experience." Start with the work.

PARAGRAPH 2 (3-4 sentences): Pick 2 achievements from the resume that are MOST relevant to THIS job. Use numbers. Frame as solving their problem, not listing your accomplishments.{projects_hint}{metrics_hint}

PARAGRAPH 3 (1-2 sentences): One specific thing about the company from the job description (a product, a technical challenge, a team structure). Then close. "Happy to walk through any of this in more detail." or "Let's discuss." Nothing else.
If COMPANY is "not stated", do not name or guess a company: refer to "your team" and use only details from the job description.

BANNED WORDS AND PHRASES (automated validator rejects ANY of these — do not use even once):
{all_banned}

ALSO BANNED (meta-commentary the validator catches):
{leak_banned}

BANNED PUNCTUATION: No em dashes (—) or en dashes (–). Use commas or periods.

VOICE:
- Write like a real engineer emailing someone they respect. Not formal, not casual. Just direct.
- NEVER narrate or explain what you're doing. BAD: "This demonstrates my commitment to X." GOOD: Just state the fact and move on.
- NEVER hedge. BAD: "might address some of your challenges." GOOD: "solves the same problem your team is facing."
- Every sentence should contain either a number, a tool name, or a specific outcome. If it doesn't, cut it.
- Read it out loud. If it sounds like a robot wrote it, rewrite it.

FABRICATION = INSTANT REJECTION:
The candidate's real tools are ONLY: {skills_str}.
Do NOT mention ANY tool not in this list. If the job asks for tools not listed, talk about the work you did, not the tools.

Sign off: just "{sign_off_name}"

Output ONLY the letter text. No subject lines. No "Here is the cover letter:" preamble. No notes after the sign-off.
Start DIRECTLY with "Dear Hiring Manager," and end with the name."""


# ── Helpers ──────────────────────────────────────────────────────────────

def _strip_preamble(text: str) -> str:
    """Remove LLM preamble before 'Dear Hiring Manager,' if present.

    Gemini and other models sometimes output "Here is the cover letter:" or
    similar meta-commentary before the actual letter text. Strip everything
    before the first occurrence of "Dear" so the validator's start-check passes.
    """
    dear_idx = text.lower().find("dear")
    if dear_idx > 0:
        return text[dear_idx:]
    return text


# ── Core Generation ──────────────────────────────────────────────────────

def generate_cover_letter(
    resume_text: str, job: dict, profile: dict,
    max_retries: int = 3, validation_mode: str = "normal",
) -> str:
    """Generate a cover letter with fresh context on each retry + auto-sanitize.

    Same design as tailor_resume: fresh conversation per attempt, issues noted
    in the prompt, no conversation history stacking.

    Args:
        resume_text:      The candidate's resume text (base or tailored).
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".

    Returns:
        The cover letter text (best attempt even if validation failed).
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job.get('company') or 'not stated'}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    avoid_notes: list[str] = []
    letter = ""
    client = get_client()
    cl_prompt_base = _build_cover_letter_prompt(profile)

    for attempt in range(max_retries + 1):
        # Fresh conversation every attempt
        prompt = cl_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES:\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"RESUME:\n{resume_text}\n\n---\n\n"
                f"TARGET JOB:\n{job_text}\n\n"
                "Write the cover letter:"
            )},
        ]

        letter = client.chat(messages, max_tokens=1024, temperature=0.7)
        letter = sanitize_text(letter)  # auto-fix em dashes, smart quotes
        letter = _strip_preamble(letter)  # remove any "Here is the letter:" prefix

        validation = validate_cover_letter(letter, mode=validation_mode)
        if validation["passed"]:
            return letter

        avoid_notes.extend(validation["errors"])
        # Warnings never block — only hard errors trigger a retry
        log.debug(
            "Cover letter attempt %d/%d failed: %s",
            attempt + 1, max_retries + 1, validation["errors"],
        )

    return letter  # last attempt even if failed


# ── Batch Entry Point ────────────────────────────────────────────────────

def run_cover_letters(min_score: int = 7, limit: int = 20,
                      validation_mode: str = "normal") -> dict:
    """Generate cover letters for high-scoring jobs, using each job's resume track.

    With tailoring on (`tailor_resumes: true`), only jobs with a tailored resume
    are eligible. Each letter is saved as soon as it is written.

    Args:
        min_score:       Minimum fit_score threshold.
        limit:           Maximum jobs to process.
        validation_mode: "strict", "normal", or "lenient".

    Returns:
        {"generated": int, "errors": int, "skipped_no_track": int, "elapsed": float}
    """
    profile = load_profile()
    search_cfg = load_search_config()
    tracks = get_resume_tracks(search_cfg)
    resumes = {name: t["text"].read_text(encoding="utf-8") for name, t in tracks.items()}
    conn = get_connection()

    tailoring = get_tailor_resumes(search_cfg)
    where, params = cover_letter_pending_where(min_score, MAX_ATTEMPTS, tailoring, list(tracks))
    loc_order, loc_params = preferred_location_order(get_preferred_locations(search_cfg))
    rows = conn.execute(
        f"SELECT * FROM jobs WHERE {where} ORDER BY fit_score DESC, {loc_order} LIMIT ?",
        (*params, *loc_params, limit),
    ).fetchall()
    jobs = [dict(row) for row in rows]
    no_track = _count_missing_track(conn, min_score, tailoring, list(tracks))
    if no_track:
        log.warning("%d jobs above score %d have no resume track; re-score them to get cover letters.",
                    no_track, min_score)

    if not jobs:
        log.info("No jobs needing cover letters (score >= %d).", min_score)
        return {"generated": 0, "errors": 0, "skipped_no_track": no_track, "elapsed": 0.0}

    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Generating cover letters for %d jobs (score >= %d)...", len(jobs), min_score)
    t0 = time.time()
    saved = 0
    error_count = 0

    for i, job in enumerate(jobs, start=1):
        try:
            resume_text = _resume_for_job(job, resumes)
            letter = generate_cover_letter(resume_text, job, profile,
                                           validation_mode=validation_mode)
            cl_path = COVER_LETTER_DIR / f"{_file_prefix(job)}_CL.txt"
            cl_path.write_text(letter, encoding="utf-8")
            _save_pdf_best_effort(cl_path)
        except DailyQuotaExceeded as e:
            # Not counted as an attempt: the job is retried on the next run.
            log.error("Cover letters stopped: %s (%d/%d done)", e, i - 1, len(jobs))
            break
        except Exception as e:
            error_count += 1
            _record_attempt(conn, job["url"], None)
            log.error("%d/%d [ERROR] %s -- %s", i, len(jobs), job["title"][:40], e)
            continue

        _record_attempt(conn, job["url"], str(cl_path))
        saved += 1
        log.info("%d/%d [OK] track=%s | %s", i, len(jobs), job.get("resume_track"), job["title"][:40])

    elapsed = time.time() - t0
    log.info("Cover letters done in %.1fs: %d generated, %d errors", elapsed, saved, error_count)
    return {"generated": saved, "errors": error_count, "skipped_no_track": no_track, "elapsed": elapsed}


def _resume_for_job(job: dict, resumes: dict[str, str]) -> str:
    """Resume text for the job's track; a single-track setup always uses its one resume.

    The pending query only returns jobs with a known track when there are
    several, so a miss here means the data changed underneath us.
    """
    if len(resumes) == 1:
        return next(iter(resumes.values()))
    track = job.get("resume_track")
    if track not in resumes:
        raise ValueError(f"Job has unknown resume track {track!r}; expected one of {list(resumes)}")
    return resumes[track]


def _count_missing_track(conn, min_score: int, tailoring: bool, tracks: list[str]) -> int:
    """Jobs that would need a letter but are excluded for lacking a known resume track."""
    if len(tracks) <= 1:
        return 0
    where, params = cover_letter_pending_where(min_score, MAX_ATTEMPTS, tailoring, [])
    marks = ",".join("?" * len(tracks))
    return conn.execute(
        f"SELECT COUNT(*) FROM jobs WHERE {where} AND (resume_track IS NULL OR resume_track NOT IN ({marks}))",
        (*params, *tracks),
    ).fetchone()[0]


def _file_prefix(job: dict) -> str:
    """Filename prefix: site + title + short URL hash, so same-title jobs don't collide."""
    safe_title = re.sub(r"[^\w\s-]", "", job["title"])[:50].strip().replace(" ", "_")
    safe_site = re.sub(r"[^\w\s-]", "", job["site"])[:20].strip().replace(" ", "_")
    url_id = hashlib.sha1(job["url"].encode("utf-8")).hexdigest()[:8]
    return f"{safe_site}_{safe_title}_{url_id}"


def _save_pdf_best_effort(cl_path) -> None:
    """Render the letter to PDF; a failure is logged but doesn't lose the .txt."""
    try:
        from applypilot.scoring.pdf import convert_to_pdf
        convert_to_pdf(cl_path)
    except Exception:
        log.warning("PDF generation failed for %s", cl_path, exc_info=True)


def _record_attempt(conn, url: str, path: str | None) -> None:
    """Count an attempt and, on success, store the letter path. Commits immediately."""
    if path:
        conn.execute(
            "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
            "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
            (path, datetime.now(UTC).isoformat(), url),
        )
    else:
        conn.execute(
            "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
            (url,),
        )
    conn.commit()
