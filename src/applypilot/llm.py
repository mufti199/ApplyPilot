"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-3.5-flash-lite)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for any provider.
LLM_RPM env var caps requests per minute (pacing for free tiers).
LLM_REASONING_EFFORT (low|medium|high) is sent as reasoning_effort when set.
"""

import logging
import os
import re
import threading
import time

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider() -> tuple[str, str, str]:
    """Return (base_url, model, api_key) based on environment variables.

    Reads env at call time (not module import time) so that load_env() called
    in _bootstrap() is always visible here.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    model_override = os.environ.get("LLM_MODEL", "")

    if gemini_key and not local_url:
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            model_override or "gemini-3.5-flash-lite",
            gemini_key,
        )

    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 120  # seconds

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, base_url: str, model: str, api_key: str, rpm: float | None = None,
                 reasoning_effort: str | None = None) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self._client = httpx.Client(timeout=_TIMEOUT)
        # Pacing: minimum seconds between request starts, shared across threads
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._pace_lock = threading.Lock()
        self._next_slot = 0.0
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    def _wait_for_slot(self) -> None:
        """Block until this request may start, keeping under LLM_RPM."""
        if not self._min_interval:
            return
        with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self._min_interval
        if start > now:
            time.sleep(start - now)

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        self._wait_for_slot()
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        candidate = data["candidates"][0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise TruncatedResponse(
                f"Gemini reply for model '{self.model}' hit the {max_tokens}-token output limit"
            )
        return candidate["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        self._wait_for_slot()
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp, max_tokens)

    def _handle_compat_response(self, resp: httpx.Response, max_tokens: int) -> str:
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise TruncatedResponse(
                f"{self._provider_name()} reply for model '{self.model}' "
                f"hit the {max_tokens}-token output limit"
            )
        return choice["message"]["content"]

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks.
        if "qwen" in self.model.lower() and messages:
            first = messages[0]
            if first.get("role") == "user" and not first["content"].startswith("/no_think"):
                messages = [{"role": first["role"], "content": f"/no_think\n{first['content']}"}] + messages[1:]

        for attempt in range(_MAX_RETRIES):
            try:
                # Route to native Gemini if we've already confirmed it's needed
                if self._use_native_gemini:
                    return self._chat_native_gemini(messages, temperature, max_tokens)

                return self._chat_compat(messages, temperature, max_tokens)

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    return self._chat_native_gemini(messages, temperature, max_tokens)
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                provider = self._provider_name()
                code, detail = _error_detail(resp)
                if code in _NON_RETRYABLE_CODES:
                    raise RuntimeError(
                        f"{provider} API error HTTP {resp.status_code} ({code}): {detail}"
                    ) from exc
                if resp.status_code == 429 and _is_daily_quota(resp):
                    raise DailyQuotaExceeded(
                        f"{provider} daily request quota reached for model '{self.model}'. "
                        "Free-tier daily quotas reset at midnight Pacific time; run again after that."
                    ) from exc
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    wait = _retry_wait(resp, attempt)
                    log.warning(
                        "%s rate limited (HTTP %s): %s. Waiting %ds before retry %d/%d.",
                        provider, resp.status_code, detail, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"{provider} API error HTTP {resp.status_code}: {detail}") from exc

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def _provider_name(self) -> str:
        if self._is_gemini or self._use_native_gemini:
            return "Gemini"
        if self.base_url.startswith("https://api.openai.com"):
            return "OpenAI"
        return f"LLM ({self.base_url})"

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


# Error codes where retrying cannot help (account/billing problems).
_NON_RETRYABLE_CODES = {"insufficient_quota", "billing_hard_limit_reached"}


class DailyQuotaExceeded(RuntimeError):
    """The provider's per-day request quota is used up; retrying today won't help."""


class TruncatedResponse(RuntimeError):
    """The reply was cut off at the output token limit, so it is incomplete."""


def _is_daily_quota(resp: httpx.Response) -> bool:
    """True if a 429 is for a per-day quota (Gemini names it e.g. '...PerDay...')."""
    return "perday" in resp.text.lower()


def _retry_wait(resp: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying a 429/503.

    Uses the Retry-After header, else Gemini's RetryInfo "retryDelay": "30s"
    in the body, else exponential backoff capped at 60s.
    """
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', resp.text)
    if match:
        return float(match.group(1)) + 1  # small margin past the reset
    return min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)


def _error_detail(resp: httpx.Response) -> tuple[str | None, str]:
    """Extract (error code, message) from an API error response.

    Handles the OpenAI-style {"error": {"code", "message"}} shape, and Gemini's
    variant (sometimes wrapped in a list, numeric code plus a string status);
    falls back to the raw body text.
    """
    try:
        body = resp.json()
        if isinstance(body, list) and body:
            body = body[0]
        err = body.get("error", {})
        if isinstance(err, dict) and err.get("message"):
            code = err.get("code")
            if not isinstance(code, str):
                code = err.get("status")
            return code, str(err["message"])[:300]
    except (ValueError, AttributeError):
        pass
    return None, resp.text[:300] or resp.reason_phrase


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: LLMClient | None = None


def get_client() -> LLMClient:
    """Return (or create) the module-level LLMClient singleton."""
    global _instance
    if _instance is None:
        base_url, model, api_key = _detect_provider()
        rpm = _read_rpm()
        effort = _read_reasoning_effort()
        log.info("LLM provider: %s  model: %s  rpm_limit: %s  reasoning_effort: %s",
                 base_url, model, rpm or "none", effort or "default")
        _instance = LLMClient(base_url, model, api_key, rpm=rpm, reasoning_effort=effort)
    return _instance


_REASONING_EFFORTS = ("low", "medium", "high")


def _read_reasoning_effort() -> str | None:
    """Parse LLM_REASONING_EFFORT from the environment. Unset/empty -> provider default."""
    raw = os.environ.get("LLM_REASONING_EFFORT", "").strip().lower()
    if not raw:
        return None
    if raw not in _REASONING_EFFORTS:
        raise ValueError(
            f"LLM_REASONING_EFFORT must be one of {', '.join(_REASONING_EFFORTS)}, got {raw!r}"
        )
    return raw


def _read_rpm() -> float | None:
    """Parse LLM_RPM from the environment. Unset/empty -> no pacing."""
    raw = os.environ.get("LLM_RPM", "").strip()
    if not raw:
        return None
    try:
        rpm = float(raw)
    except ValueError:
        raise ValueError(f"LLM_RPM must be a number, got {raw!r}") from None
    if rpm <= 0:
        raise ValueError(f"LLM_RPM must be greater than 0, got {raw!r}")
    return rpm
