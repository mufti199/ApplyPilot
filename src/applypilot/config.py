"""ApplyPilot configuration: paths, platform detection, user data."""

import os
import platform
import re
import shutil
from pathlib import Path

# User data directory — all user-specific files live here
APP_DIR = Path(os.environ.get("APPLYPILOT_DIR", Path.home() / ".applypilot"))

# Core paths
DB_PATH = APP_DIR / "applypilot.db"
PROFILE_PATH = APP_DIR / "profile.json"
RESUME_PATH = APP_DIR / "resume.txt"
RESUME_PDF_PATH = APP_DIR / "resume.pdf"
SEARCH_CONFIG_PATH = APP_DIR / "searches.yaml"
ENV_PATH = APP_DIR / ".env"

# Generated output
TAILORED_DIR = APP_DIR / "tailored_resumes"
COVER_LETTER_DIR = APP_DIR / "cover_letters"
LOG_DIR = APP_DIR / "logs"

# Chrome worker isolation
CHROME_WORKER_DIR = APP_DIR / "chrome-workers"
APPLY_WORKER_DIR = APP_DIR / "apply-workers"

# Package-shipped config (YAML registries)
PACKAGE_DIR = Path(__file__).parent
CONFIG_DIR = PACKAGE_DIR / "config"


def get_chrome_path() -> str:
    """Auto-detect Chrome/Chromium executable path, cross-platform.

    Override with CHROME_PATH environment variable.
    """
    env_path = os.environ.get("CHROME_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    system = platform.system()

    if system == "Windows":
        candidates = [
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
    elif system == "Darwin":
        candidates = [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    else:  # Linux
        candidates = []
        for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))

    for c in candidates:
        if c and c.exists():
            return str(c)

    # Fall back to PATH search
    for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError(
        "Chrome/Chromium not found. Install Chrome or set CHROME_PATH environment variable."
    )


def get_chrome_user_data() -> Path:
    """Default Chrome user data directory, cross-platform."""
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "User Data"
    elif system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    else:
        return Path.home() / ".config" / "google-chrome"


def ensure_dirs():
    """Create all required directories."""
    for d in [APP_DIR, TAILORED_DIR, COVER_LETTER_DIR, LOG_DIR, CHROME_WORKER_DIR, APPLY_WORKER_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    """Load user profile from ~/.applypilot/profile.json."""
    import json
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Profile not found at {PROFILE_PATH}. Run `applypilot init` first."
        )
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def load_search_config() -> dict:
    """Load search configuration from ~/.applypilot/searches.yaml."""
    import yaml
    if not SEARCH_CONFIG_PATH.exists():
        # Fall back to package-shipped example
        example = CONFIG_DIR / "searches.example.yaml"
        if example.exists():
            return yaml.safe_load(example.read_text(encoding="utf-8"))
        return {}
    return yaml.safe_load(SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))


DISCOVERY_SOURCES = ("jobspy", "workday", "smartextract")


def get_discovery_sources(search_cfg: dict | None) -> dict[str, bool]:
    """Return which discovery sources are enabled.

    Reads the optional `discovery_sources` mapping from the search config.
    Sources not listed default to enabled.

    Raises:
        ValueError: if a source name is unknown.
        TypeError: if the setting is not a mapping or a value is not a boolean.
    """
    raw = (search_cfg or {}).get("discovery_sources")
    enabled = {name: True for name in DISCOVERY_SOURCES}
    if raw is None:
        return enabled
    if not isinstance(raw, dict):
        raise TypeError("discovery_sources must be a mapping of source name to true/false")

    for name, value in raw.items():
        if name not in enabled:
            raise ValueError(
                f"Unknown discovery source '{name}'. Valid: {', '.join(DISCOVERY_SOURCES)}"
            )
        if not isinstance(value, bool):
            raise TypeError(f"discovery_sources.{name} must be true or false, got {value!r}")
        enabled[name] = value
    return enabled


def get_preferred_locations(search_cfg: dict | None) -> list[str]:
    """Return the `location_preferred` patterns from the search config.

    Jobs whose location contains one of these (case-insensitive) are handled
    first among jobs with the same fit score. Missing setting -> [].

    Raises:
        TypeError: if the setting is not a list of strings.
        ValueError: if a pattern is blank.
    """
    raw = (search_cfg or {}).get("location_preferred")
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
        raise TypeError("location_preferred must be a list of strings")
    patterns = [p.strip() for p in raw]
    if any(not p for p in patterns):
        raise ValueError("location_preferred must not contain blank entries")
    return patterns


def get_auto_apply(search_cfg: dict | None) -> bool:
    """Whether `applypilot apply` may submit applications (`auto_apply`, default true).

    Raises:
        TypeError: if the setting is not true/false.
    """
    value = (search_cfg or {}).get("auto_apply", True)
    if not isinstance(value, bool):
        raise TypeError(f"auto_apply must be true or false, got {value!r}")
    return value


def get_permanent_full_time_only(search_cfg: dict | None) -> bool:
    """Whether to exclude contract, part-time, casual and temporary jobs.

    Reads `employment: {permanent_full_time_only: true}` (default false).

    Raises:
        TypeError: if the setting has the wrong shape.
    """
    employment = (search_cfg or {}).get("employment", {})
    if not isinstance(employment, dict):
        raise TypeError("employment must be a mapping, e.g. {permanent_full_time_only: true}")
    value = employment.get("permanent_full_time_only", False)
    if not isinstance(value, bool):
        raise TypeError(f"employment.permanent_full_time_only must be true or false, got {value!r}")
    return value


MAX_SCORE_GROUP_SIZE = 10


def get_score_group_size(search_cfg: dict | None) -> int:
    """Jobs scored per LLM request (`score_jobs_per_request`, default 1, max 10).

    Raises:
        TypeError: if the setting is not an integer.
        ValueError: if it is outside 1..MAX_SCORE_GROUP_SIZE.
    """
    value = (search_cfg or {}).get("score_jobs_per_request", 1)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"score_jobs_per_request must be a whole number, got {value!r}")
    if not 1 <= value <= MAX_SCORE_GROUP_SIZE:
        raise ValueError(f"score_jobs_per_request must be between 1 and {MAX_SCORE_GROUP_SIZE}, got {value}")
    return value


def get_tailor_resumes(search_cfg: dict | None) -> bool:
    """Return whether the pipeline tailors a resume per job (`tailor_resumes`, default true).

    Raises:
        TypeError: if the setting is not true/false.
    """
    value = (search_cfg or {}).get("tailor_resumes", True)
    if not isinstance(value, bool):
        raise TypeError(f"tailor_resumes must be true or false, got {value!r}")
    return value


_TRACK_NAME_RE = r"^[a-z0-9_-]+$"


def get_resume_tracks(search_cfg: dict | None) -> dict[str, dict]:
    """Return the resume tracks: {track_name: {"text": Path, "pdf": Path | None, "focus": str | None}}.

    Reads the optional `resumes` mapping from the search config, e.g.
        resumes:
          software: {text: ".../resume_software.txt", pdf: ".../resume.pdf",
                     focus: "application, backend and AI roles"}
    Without it, falls back to a single "default" track using resume.txt/.pdf.

    Raises:
        TypeError: if the setting has the wrong shape.
        ValueError: if a track name is invalid or a text path is missing.
        FileNotFoundError: if a configured file does not exist.
    """
    raw = (search_cfg or {}).get("resumes")
    if raw is None:
        pdf = RESUME_PDF_PATH if RESUME_PDF_PATH.exists() else None
        return {"default": {"text": RESUME_PATH, "pdf": pdf, "focus": None}}
    if not isinstance(raw, dict) or not raw:
        raise TypeError("resumes must be a non-empty mapping of track name to {text, pdf}")

    tracks: dict[str, dict[str, Path | None]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not re.match(_TRACK_NAME_RE, name):
            raise ValueError(f"Invalid resume track name {name!r}: use lowercase letters, digits, - or _")
        if not isinstance(entry, dict):
            raise TypeError(f"resumes.{name} must be a mapping with 'text' and optional 'pdf'")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"resumes.{name}.text must be a file path")
        pdf = entry.get("pdf")
        if pdf is not None and not isinstance(pdf, str):
            raise TypeError(f"resumes.{name}.pdf must be a file path")
        focus = entry.get("focus")
        if focus is not None and (not isinstance(focus, str) or not focus.strip()):
            raise TypeError(f"resumes.{name}.focus must be a short non-empty description")

        text_path = Path(text).expanduser()
        pdf_path = Path(pdf).expanduser() if pdf else None
        for label, path in (("text", text_path), ("pdf", pdf_path)):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"resumes.{name}.{label} not found: {path}")
        tracks[name] = {"text": text_path, "pdf": pdf_path, "focus": focus.strip() if focus else None}
    return tracks


def load_sites_config() -> dict:
    """Load sites.yaml configuration (sites list, manual_ats, blocked, etc.)."""
    import yaml
    path = CONFIG_DIR / "sites.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def is_manual_ats(url: str | None) -> bool:
    """Check if a URL routes through an ATS that requires manual application."""
    if not url:
        return False
    sites_cfg = load_sites_config()
    domains = sites_cfg.get("manual_ats", [])
    url_lower = url.lower()
    return any(domain in url_lower for domain in domains)


def load_blocked_sites() -> tuple[set[str], list[str]]:
    """Load blocked sites and URL patterns from sites.yaml.

    Returns:
        (blocked_site_names, blocked_url_patterns)
    """
    cfg = load_sites_config()
    blocked = cfg.get("blocked", {})
    sites = set(blocked.get("sites", []))
    patterns = blocked.get("url_patterns", [])
    return sites, patterns


def load_blocked_sso() -> list[str]:
    """Load blocked SSO domains from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("blocked_sso", [])


def load_base_urls() -> dict[str, str | None]:
    """Load site base URLs for URL resolution from sites.yaml."""
    cfg = load_sites_config()
    return cfg.get("base_urls", {})


# ---------------------------------------------------------------------------
# Default values — referenced across modules instead of magic numbers
# ---------------------------------------------------------------------------

DEFAULTS = {
    "min_score": 7,
    "max_apply_attempts": 3,
    "max_tailor_attempts": 5,
    "poll_interval": 60,
    "apply_timeout": 300,
    "viewport": "1280x900",
}


def load_env():
    """Load environment variables from ~/.applypilot/.env if it exists."""
    from dotenv import load_dotenv
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    # Also try CWD .env as fallback
    load_dotenv()


# ---------------------------------------------------------------------------
# Tier system — feature gating by installed dependencies
# ---------------------------------------------------------------------------

TIER_LABELS = {
    1: "Discovery",
    2: "AI Scoring & Tailoring",
    3: "Full Auto-Apply",
}

TIER_COMMANDS: dict[int, list[str]] = {
    1: ["init", "run discover", "run enrich", "status", "dashboard"],
    2: ["run score", "run tailor", "run cover", "run pdf", "run"],
    3: ["apply"],
}


def get_tier() -> int:
    """Detect the current tier based on available dependencies.

    Tier 1 (Discovery):            Python + pip
    Tier 2 (AI Scoring & Tailoring): + LLM API key
    Tier 3 (Full Auto-Apply):       + Claude Code CLI + Chrome
    """
    load_env()

    has_llm = any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_URL"))
    if not has_llm:
        return 1

    has_claude = shutil.which("claude") is not None
    try:
        get_chrome_path()
        has_chrome = True
    except FileNotFoundError:
        has_chrome = False

    if has_claude and has_chrome:
        return 3

    return 2


def check_tier(required: int, feature: str) -> None:
    """Raise SystemExit with a clear message if the current tier is too low.

    Args:
        required: Minimum tier needed (1, 2, or 3).
        feature: Human-readable description of the feature being gated.
    """
    current = get_tier()
    if current >= required:
        return

    from rich.console import Console
    _console = Console(stderr=True)

    missing: list[str] = []
    if required >= 2 and not any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_URL")):
        missing.append("LLM API key — run [bold]applypilot init[/bold] or set GEMINI_API_KEY")
    if required >= 3:
        if not shutil.which("claude"):
            missing.append("Claude Code CLI — install from [bold]https://claude.ai/code[/bold]")
        try:
            get_chrome_path()
        except FileNotFoundError:
            missing.append("Chrome/Chromium — install or set CHROME_PATH")

    _console.print(
        f"\n[red]'{feature}' requires {TIER_LABELS.get(required, f'Tier {required}')} (Tier {required}).[/red]\n"
        f"Current tier: {TIER_LABELS.get(current, f'Tier {current}')} (Tier {current})."
    )
    if missing:
        _console.print("\n[yellow]Missing:[/yellow]")
        for m in missing:
            _console.print(f"  - {m}")
    _console.print()
    raise SystemExit(1)
