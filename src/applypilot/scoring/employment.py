"""Employment-type filter: keep only permanent, full-time roles.

Three layers, cheapest first:
  1. JobSpy's own job_type (stored at discovery).
  2. Wording rules on title / description / salary (no LLM).
  3. The scorer's EMPLOYMENT answer (same request as the score).
A job that fails any layer gets `excluded_reason` set; it is kept in the DB
(visible in the dashboard's "show excluded" view) but never scored further,
given a cover letter, or auto-applied.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# JobSpy job_type values (comma-separated when several) that are not permanent full-time.
_BAD_JOB_TYPES = {"parttime", "contract", "temporary", "internship", "casual"}

_TITLE_RE = re.compile(
    r"\b(contract(or)?|fixed[- ]term|part[- ]time|casual|temp(orary)?|locum|internship|intern|"
    r"maternity|parental leave|secondment|\d+\s*(-\s*\d+\s*)?months?)\b|"
    r"(\$\s?\d+\s*(/|per\s*)(hr|hour|day)|/\s*hr\b|per hour|day rate|daily rate)",
    re.IGNORECASE,
)

_DESC_RE = re.compile(
    r"(fixed[- ]term (contract|role|position|basis)|"
    r"(initial|\d+)[- ]?(\d+[- ])?months?[- ](contract|assignment|engagement)|"
    r"(this is a|daily rate|day rate) contract|contract (role|position|opportunity|assignment|engagement)|"
    r"\bday rate\b|\bdaily rate\b|per day rate|"
    r"part[- ]time (role|position|basis|hours|opportunity)|"
    r"casual (role|position|basis)|temporary (role|position|contract|assignment)|"
    r"parental leave (cover|backfill)|maternity (leave )?(cover|backfill))",
    re.IGNORECASE,
)

# Scorer EMPLOYMENT answers that mean "not permanent full-time". 'unknown' is kept.
LLM_EXCLUDED_TYPES = {"contract", "part-time", "casual", "temporary", "internship"}


def rule_exclusion_reason(title: str | None, description: str | None,
                          salary: str | None, job_type: str | None) -> str | None:
    """Reason a job is clearly not permanent full-time, or None if nothing says so."""
    if job_type:
        types = {t.strip().lower().replace("-", "").replace(" ", "") for t in job_type.split(",")}
        bad = types & _BAD_JOB_TYPES
        if bad:
            return f"job board type: {', '.join(sorted(bad))}"
    if title and (m := _TITLE_RE.search(title)):
        return f"title says '{m.group(0).strip()}'"
    if salary and re.search(r"/(hour|hourly|day|daily)\b", salary, re.IGNORECASE):
        return f"hourly/daily pay: {salary}"
    if description and (m := _DESC_RE.search(description)):
        return f"description says '{m.group(0).strip()}'"
    return None


def normalise_llm_employment(value: str | None) -> str | None:
    """Map the scorer's EMPLOYMENT answer to one word (or None if absent/unknown)."""
    if not value:
        return None
    v = value.strip().strip("[]").lower()
    if "permanent" in v or v in ("full-time", "full time", "fulltime"):
        return "permanent"
    for kind in ("contract", "part-time", "casual", "temporary", "internship"):
        if kind in v or kind.replace("-", " ") in v:
            return kind
    return None


def apply_rules(conn) -> int:
    """Exclude not-yet-excluded jobs that the wording rules flag. Returns how many were newly excluded."""
    rows = conn.execute(
        "SELECT rowid, title, full_description, salary, job_type FROM jobs WHERE excluded_reason IS NULL"
    ).fetchall()
    newly = 0
    for r in rows:
        reason = rule_exclusion_reason(r["title"], r["full_description"], r["salary"], r["job_type"])
        if reason:
            conn.execute("UPDATE jobs SET excluded_reason = ? WHERE rowid = ?", (reason, r["rowid"]))
            newly += 1
    conn.commit()
    if newly:
        log.info("Employment filter: %d jobs excluded as not permanent full-time", newly)
    return newly
