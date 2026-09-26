"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import json
import logging
import re
import time
from datetime import UTC, datetime

from applypilot.config import get_preferred_locations, get_resume_tracks, load_search_config
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import DailyQuotaExceeded, get_client

log = logging.getLogger(__name__)

# Stop a scoring run after this many failures in a row (e.g. bad API key or no quota).
MAX_CONSECUTIVE_FAILURES = 5


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has some relevant skills but missing key requirements.
- 3-4: Weak match. Significant skill gaps, would need substantial ramp-up.
- 1-2: Poor match. Completely different field or experience level.

IMPORTANT FACTORS:
- Weight technical skills heavily (programming languages, frameworks, tools)
- Consider transferable experience (automation, scripting, API work)
- Factor in the candidate's project experience
- Be realistic about experience level vs. job requirements (years of experience, seniority)

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]"""

MULTI_RESUME_SUFFIX = """

MULTIPLE RESUMES:
You are given several resumes for the same candidate, each labelled with a TRACK name.
Pick the ONE resume that best fits this job and score the job against that resume only.
Put this extra line FIRST in your response, before SCORE:
TRACK: [exactly one of: {tracks}]"""


def _build_score_prompt(track_names: list[str]) -> str:
    """Return the scoring system prompt for one or more resume tracks."""
    if len(track_names) == 1:
        return SCORE_PROMPT
    return SCORE_PROMPT + MULTI_RESUME_SUFFIX.format(tracks=", ".join(track_names))


def _format_resumes(resumes: dict[str, str]) -> str:
    """Format resume text(s) for the user message."""
    if len(resumes) == 1:
        return f"RESUME:\n{next(iter(resumes.values()))}"
    return "\n\n".join(f"RESUME (TRACK: {name}):\n{text}" for name, text in resumes.items())


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str, "track": str | None}
    """
    score = 0
    keywords = ""
    reasoning = response
    track = None

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("TRACK:"):
            track = line.replace("TRACK:", "").strip().strip("[]").lower() or None
        elif line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning, "track": track}


def score_job(resumes: dict[str, str], job: dict) -> dict:
    """Score a single job against the best-fitting resume.

    Args:
        resumes: Resume text per track name. With one track, that track is used.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str, "track": str | None}
        score is 0 on an LLM error or an unusable TRACK answer.
    """
    track_names = list(resumes)
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    messages = [
        {"role": "system", "content": _build_score_prompt(track_names)},
        {"role": "user", "content": f"{_format_resumes(resumes)}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=512, temperature=0.2)
    except DailyQuotaExceeded:
        raise  # run_scoring stops the whole run
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}", "track": None}

    result = _parse_score_response(response)
    if len(track_names) == 1:
        result["track"] = track_names[0]
    elif result["track"] not in resumes:
        log.error("Invalid TRACK %r for job '%s' (expected one of %s)",
                  result["track"], job.get("title", "?"), track_names)
        return {"score": 0, "keywords": "", "reasoning": f"Invalid TRACK in LLM response: {result['track']!r}",
                "track": None}
    return result


def _save_score(conn, url: str, result: dict) -> None:
    """Persist one successful score and commit so it survives a crash."""
    conn.execute(
        "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, resume_track = ? WHERE url = ?",
        (result["score"], f"{result['keywords']}\n{result['reasoning']}",
         datetime.now(UTC).isoformat(), result["track"], url),
    )
    conn.commit()


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Each successful score is committed immediately. Failed jobs (score 0) are
    not written, so they stay unscored and are retried on the next run. The run
    stops early after MAX_CONSECUTIVE_FAILURES failures in a row.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list, "aborted": bool}
    """
    search_cfg = load_search_config()
    tracks = get_resume_tracks(search_cfg)
    resumes = {name: t["text"].read_text(encoding="utf-8") for name, t in tracks.items()}
    log.info("Scoring with resume tracks: %s", ", ".join(resumes))
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit,
                                 preferred_locations=get_preferred_locations(search_cfg))

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": [], "aborted": False}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    scored = 0
    errors = 0
    consecutive_failures = 0
    aborted = False

    for i, job in enumerate(jobs, start=1):
        try:
            result = score_job(resumes, job)
        except DailyQuotaExceeded as e:
            aborted = True
            log.error("Scoring stopped: %s (%d scored this run, %d left unscored)",
                      e, scored, len(jobs) - i + 1)
            break
        title = job.get("title", "?")[:60]

        if result["score"] == 0:
            errors += 1
            consecutive_failures += 1
            log.warning("[%d/%d] scoring failed, left unscored for retry: %s (%s)",
                        i, len(jobs), title, result["reasoning"])
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                aborted = True
                log.error(
                    "Scoring aborted after %d consecutive failures (%d scored this run). "
                    "Check your LLM provider, API key and billing/quota.",
                    consecutive_failures, scored,
                )
                break
            continue

        _save_score(conn, job["url"], result)
        scored += 1
        consecutive_failures = 0
        log.info("[%d/%d] score=%d track=%s  %s", i, len(jobs), result["score"], result["track"], title)

    elapsed = time.time() - t0
    log.info("Done: %d scored, %d errors in %.1fs", scored, errors, elapsed)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": scored,
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
        "aborted": aborted,
    }
