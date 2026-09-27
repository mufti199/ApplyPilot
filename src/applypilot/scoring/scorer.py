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

from applypilot.config import (
    get_permanent_full_time_only,
    get_preferred_locations,
    get_resume_tracks,
    get_score_group_size,
    load_search_config,
)
from applypilot.database import get_connection, get_jobs_by_stage, mark_duplicates
from applypilot.llm import DailyQuotaExceeded, get_client
from applypilot.scoring.employment import LLM_EXCLUDED_TYPES, normalise_llm_employment
from applypilot.scoring.employment import apply_rules as apply_employment_rules

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
- Hard requirements the candidate clearly lacks (citizenship or security clearance, mandatory
  certifications or licences, far more seniority) are major gaps: score them as such

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]
COMPANY: [the employer's name exactly as the posting states it, or unknown]
EMPLOYMENT: [permanent full-time, contract, part-time, casual, temporary, internship, or unknown]"""

MULTI_RESUME_SUFFIX = """

MULTIPLE RESUMES:
You are given several resumes for the same candidate, each labelled with a TRACK name.
Pick the ONE resume that best fits this job and score the job against that resume only.
Decide by the job's main day-to-day duties, not by its title or the order the resumes are listed.{guide}
Add this line directly before the SCORE line (for every job you score):
TRACK: [exactly one of: {tracks}]"""


BATCH_SUFFIX = """

MULTIPLE JOBS:
You are given {count} job postings, labelled JOB 1 to JOB {count}. Evaluate each one
independently against the resume(s); do not compare the jobs with each other.
For EACH job, output one block that starts with a line "JOB: <number>" followed by all
the lines in the format above (including TRACK, if asked for) for that job. Every block must
contain all of its own lines; never share a line between jobs. Output the blocks in order,
one per job, nothing else."""

# Output budget per job in a grouped request (visible answer + model reasoning).
_TOKENS_PER_JOB = 700
# Output budget for a single-job request; reasoning models need room before the answer.
_SINGLE_JOB_TOKENS = 4096


def _build_score_prompt(track_names: list[str], guides: dict[str, str] | None = None) -> str:
    """Return the scoring system prompt for one or more resume tracks.

    guides maps a track name to the kind of roles it is for (from `focus` in
    searches.yaml); it is shown to the model to help it pick the right resume.
    """
    if len(track_names) == 1:
        return SCORE_PROMPT
    lines = [f"- {name}: {guides[name]}" for name in track_names if guides and guides.get(name)]
    guide = ("\nWhich TRACK fits which roles:\n" + "\n".join(lines)) if lines else ""
    return SCORE_PROMPT + MULTI_RESUME_SUFFIX.format(tracks=", ".join(track_names), guide=guide)


def _format_resumes(resumes: dict[str, str]) -> str:
    """Format resume text(s) for the user message."""
    if len(resumes) == 1:
        return f"RESUME:\n{next(iter(resumes.values()))}"
    return "\n\n".join(f"RESUME (TRACK: {name}):\n{text}" for name, text in resumes.items())


def _clean_line(line: str) -> str:
    """Strip markdown decoration models add, e.g. '**SCORE:** 7' or '- TRACK : devops'."""
    line = line.strip().lstrip("-*> ").replace("**", "").replace("__", "")
    return re.sub(r"^([A-Z]+)\s*:", r"\1:", line)


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
    company = ""
    employment = None

    for line in response.split("\n"):
        line = _clean_line(line)
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
        elif line.startswith("COMPANY:"):
            company = line.replace("COMPANY:", "").strip().strip("[]")
        elif line.startswith("EMPLOYMENT:"):
            employment = normalise_llm_employment(line.replace("EMPLOYMENT:", ""))

    return {"score": score, "keywords": keywords, "reasoning": reasoning, "track": track,
            "company": None if company.lower() in ("", "unknown", "not stated", "n/a") else company,
            "employment": employment}


def score_job(resumes: dict[str, str], job: dict, guides: dict[str, str] | None = None) -> dict:
    """Score a single job against the best-fitting resume.

    Args:
        resumes: Resume text per track name. With one track, that track is used.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str, "track": str | None}
        score is 0 on an LLM error or an unusable TRACK answer.
    """
    track_names = list(resumes)
    job_text = _format_job(job)

    messages = [
        {"role": "system", "content": _build_score_prompt(track_names, guides)},
        {"role": "user", "content": f"{_format_resumes(resumes)}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client()
        response = client.chat(messages, max_tokens=_SINGLE_JOB_TOKENS, temperature=0.2)
    except DailyQuotaExceeded:
        raise  # run_scoring stops the whole run
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}", "track": None}

    return _check_track(_parse_score_response(response), resumes, job)


def _format_job(job: dict) -> str:
    return (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job.get('company') or 'not stated'}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )


def _failure(reason: str) -> dict:
    return {"score": 0, "keywords": "", "reasoning": reason, "track": None}


def _check_track(result: dict, resumes: dict[str, str], job: dict) -> dict:
    """Fill in the only track, or reject a missing/unknown TRACK answer."""
    track_names = list(resumes)
    if len(track_names) == 1:
        result["track"] = track_names[0]
    elif result["track"] not in resumes:
        log.error("Invalid TRACK %r for job '%s' (expected one of %s)",
                  result["track"], job.get("title", "?"), track_names)
        return _failure(f"Invalid TRACK in LLM response: {result['track']!r}")
    return result


def score_jobs(resumes: dict[str, str], jobs: list[dict],
               guides: dict[str, str] | None = None) -> list[dict]:
    """Score a group of jobs in one LLM request; one result per job, in order.

    A group of one uses score_job. Jobs missing or unreadable in the reply get
    a failure result (score 0) so they are retried later; an LLM error fails
    the whole group. DailyQuotaExceeded is re-raised for run_scoring.
    """
    if len(jobs) == 1:
        return [score_job(resumes, jobs[0], guides)]

    system = _build_score_prompt(list(resumes), guides) + BATCH_SUFFIX.format(count=len(jobs))
    postings = "\n\n".join(f"JOB {n}:\n{_format_job(job)}" for n, job in enumerate(jobs, start=1))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{_format_resumes(resumes)}\n\n---\n\nJOB POSTINGS:\n{postings}"},
    ]

    try:
        response = get_client().chat(messages, max_tokens=_TOKENS_PER_JOB * len(jobs), temperature=0.2)
    except DailyQuotaExceeded:
        raise
    except Exception as e:
        log.error("LLM error scoring a group of %d jobs: %s", len(jobs), e)
        return [_failure(f"LLM error: {e}") for _ in jobs]

    blocks = _split_job_blocks(response)
    results = []
    for n, job in enumerate(jobs, start=1):
        block = blocks.get(n)
        if block is None:
            results.append(_failure(f"JOB {n} missing from grouped reply"))
            continue
        parsed = _parse_score_response(block)
        if parsed["score"] == 0:
            results.append(_failure(f"JOB {n} had no usable SCORE in grouped reply"))
            continue
        results.append(_check_track(parsed, resumes, job))
    return results


def _split_job_blocks(response: str) -> dict[int, str]:
    """Split a grouped reply into {job number: block text}. Duplicate numbers keep the first."""
    blocks: dict[int, str] = {}
    current = None
    lines: list[str] = []
    for line in response.splitlines():
        match = re.match(r"^\s*\**\s*JOB\s*:?\s*(\d+)\s*\**\s*:?\s*$", line, flags=re.IGNORECASE)
        if match:
            if current is not None and current not in blocks:
                blocks[current] = "\n".join(lines)
            current, lines = int(match.group(1)), []
        elif current is not None:
            lines.append(line)
    if current is not None and current not in blocks:
        blocks[current] = "\n".join(lines)
    return blocks


def _save_score(conn, url: str, result: dict, exclude_non_permanent: bool = False) -> None:
    """Persist one successful score and commit so it survives a crash.

    With exclude_non_permanent, a job the scorer calls contract/part-time/etc.
    is also excluded (kept in the DB, skipped by later stages).
    """
    conn.execute(
        "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ?, resume_track = ?, "
        "company = COALESCE(company, ?) WHERE url = ?",
        (result["score"], f"{result['keywords']}\n{result['reasoning']}",
         datetime.now(UTC).isoformat(), result["track"], result.get("company"), url),
    )
    employment = result.get("employment")
    if exclude_non_permanent and employment in LLM_EXCLUDED_TYPES:
        conn.execute("UPDATE jobs SET excluded_reason = ? WHERE url = ? AND excluded_reason IS NULL",
                     (f"scorer says {employment}", url))
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
    guides = {name: t["focus"] for name, t in tracks.items() if t.get("focus")}
    log.info("Scoring with resume tracks: %s", ", ".join(resumes))
    conn = get_connection()
    dupes = mark_duplicates(conn)
    permanent_only = get_permanent_full_time_only(search_cfg)
    if permanent_only:
        apply_employment_rules(conn)
    log.info("Duplicate postings: %d jobs are copies and will not be scored", dupes)

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

    group_size = get_score_group_size(search_cfg)
    log.info("Scoring %d jobs, %d per request...", len(jobs), group_size)
    t0 = time.time()
    scored = 0
    errors = 0
    consecutive_failures = 0  # failed requests in a row (every job in the group failed)
    aborted = False

    for start in range(0, len(jobs), group_size):
        group = jobs[start:start + group_size]
        try:
            results = score_jobs(resumes, group, guides)
        except DailyQuotaExceeded as e:
            aborted = True
            log.error("Scoring stopped: %s (%d scored this run, %d left unscored)",
                      e, scored, len(jobs) - start)
            break

        group_ok = False
        for offset, (job, result) in enumerate(zip(group, results), start=1):
            i = start + offset
            title = job.get("title", "?")[:60]
            if result["score"] == 0:
                errors += 1
                log.warning("[%d/%d] scoring failed, left unscored for retry: %s (%s)",
                            i, len(jobs), title, result["reasoning"])
                continue
            _save_score(conn, job["url"], result, exclude_non_permanent=permanent_only)
            scored += 1
            group_ok = True
            log.info("[%d/%d] score=%d track=%s  %s", i, len(jobs), result["score"], result["track"], title)

        consecutive_failures = 0 if group_ok else consecutive_failures + 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            aborted = True
            log.error(
                "Scoring aborted after %d consecutive failed requests (%d scored this run). "
                "Check your LLM provider, API key and billing/quota.",
                consecutive_failures, scored,
            )
            break

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
