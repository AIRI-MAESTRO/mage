"""
LLM client abstraction for MAGE.

Provides a unified async interface to OpenAI-compatible LLM endpoints
(OpenAI, OpenRouter, Ollama, vLLM, etc.) with per-stage config overrides.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from .exceptions import MAGELLMError
from .schemas import MAGEConfig, StageUsage

# Generic bound for `generate_structured`: the method returns an instance of
# whatever Pydantic model class is passed as `response_model`.
ModelT = TypeVar("ModelT", bound=BaseModel)

# Type alias for the streaming chunk callback. Receives (stage, delta) per
# token / chunk. Generator wires `progress.on_llm_chunk` into this slot.
StreamCallback = Callable[[str, str], None]

# Type alias for the retry-notification callback. Receives
# (stage, attempt, error) when an LLM call attempt fails AND a retry
# is queued. Generator wires `progress.on_retry` into this slot.
RetryCallback = Callable[[str, int, Exception], None]


def _extract_chunk_delta(chunk: Any) -> str:
    """Extract the text delta from a streaming chunk.

    The OpenAI SDK exposes chunks as objects with
    ``chunk.choices[0].delta.content``. Some local servers / older
    proxies return dicts instead. Returns ``""`` when there's no usable
    content (e.g. role-only deltas, tool-call deltas).
    """
    try:
        if isinstance(chunk, dict):
            choices = chunk.get("choices")
        else:
            choices = getattr(chunk, "choices", None)
    except Exception:  # noqa: BLE001
        return ""
    if not choices:
        return ""
    first = choices[0]
    if isinstance(first, dict):
        delta_obj: Any = first.get("delta")
    else:
        delta_obj = getattr(first, "delta", None)
    if delta_obj is None:
        return ""
    if isinstance(delta_obj, dict):
        content = delta_obj.get("content")
    else:
        content = getattr(delta_obj, "content", None)
    if content is None:
        return ""
    return str(content)


def _noop_stream_callback(stage: str, delta: str) -> None:
    return


# §6 P2 — appended to the system prompt when the provider doesn't
# support OpenAI-style JSON mode. Strict-JSON instruction matches what
# `_extract_json_object` is willing to parse.
_JSON_ONLY_SUFFIX = (
    "\n\nIMPORTANT: Return ONLY a single JSON object — no prose, no "
    "markdown fences, no comments. The first character must be `{` and "
    "the last must be `}`."
)


def _extract_json_object(text: str) -> str | None:
    """Pull the first balanced ``{...}`` JSON object out of arbitrary text.

    Used by :meth:`LLMClient.generate_json` when
    ``MAGEConfig.json_mode_fallback=True``. Handles the common cases
    where local models wrap their JSON in markdown fences or add
    explanatory prose around it. Returns ``None`` when no balanced
    object is found.

    Brace-counting is preferred over a regex because models often emit
    nested objects (``{"a": {"b": 1}}``) that regex alternations can't
    handle correctly.
    """
    if not text:
        return None
    # Strip markdown fences if present so the brace search has a clean target.
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (possibly with a language hint) and any
        # closing fence.
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()

    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    return None


def _balance_json(text: str) -> str | None:
    """Repair a TRUNCATED / unbalanced JSON object (T21).

    Starts at the first ``{``, tracks string + bracket state, drops any trailing
    junk once the object balances, and — when the reply was cut off mid-output —
    closes an open string and appends the missing closing brackets. Returns the
    repaired substring, or ``None`` when there's no ``{``.
    """
    start = text.find("{")
    if start == -1:
        return None
    stack: list[str] = []
    in_string = False
    escape = False
    out: list[str] = []
    for ch in text[start:]:
        out.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
                if not stack:
                    return "".join(out)  # balanced — drop any trailing prose
    s = "".join(out)
    if in_string:
        s += '"'
    s += "".join(reversed(stack))
    return s


def _repair_json(raw: str) -> Any | None:
    """Best-effort recovery of a JSON object from a noisy / truncated reply (T21).

    A single unparseable reply must not restart the whole generation pipeline
    (the run-773 failure). Tries the brace-balanced extraction (strips markdown
    fences + trailing prose), then the bracket-balancing repair that closes a
    cut-off object. Returns the parsed object, or ``None`` if nothing parses.
    """
    for candidate in (_extract_json_object(raw), _balance_json(raw)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


logger = logging.getLogger(__name__)


class LLMClient:
    """Async wrapper around an OpenAI-compatible chat completions API.

    Supports multiple providers (OpenAI, OpenRouter, local, custom) and
    per-stage LLM configuration overrides.

    Parameters
    ----------
    config : MAGEConfig
        MAGE configuration containing model, temperature, api_key, base_url,
        provider settings, and stage overrides.

    Examples
    --------
    >>> client = LLMClient(MAGEConfig(model="gpt-4o", api_key="sk-..."))
    >>> result = await client.generate("You are a helper.", "Hello!")
    """

    def __init__(self, config: MAGEConfig) -> None:
        self._config = config
        api_key = config.api_key or os.getenv("MAGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "no-key"

        http_client = httpx.AsyncClient(
            verify=config.ssl_verify,
            timeout=httpx.Timeout(config.http_timeout),
        )

        extra_headers: dict[str, str] = {}
        if config.provider == "openrouter":
            extra_headers["HTTP-Referer"] = config.openrouter_site_url or "https://github.com/Glazkoff/carl-mage"
            extra_headers["X-Title"] = config.openrouter_app_name or "carl-mage"

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "http_client": http_client,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if extra_headers:
            kwargs["default_headers"] = extra_headers

        self._client = AsyncOpenAI(**kwargs)
        self.model = config.model
        self.temperature = config.temperature
        # WP-5: per-stage token usage accumulated from each call's response.usage.
        self.usage_by_stage: dict[str, StageUsage] = {}
        self.max_retries = config.max_retries

        # Default retry callback applied to every `generate()` call that
        # doesn't pass an explicit one. Generator sets this once per run
        # so CARE's TUI is notified without every agent having to
        # thread the callback through.
        self._retry_callback: RetryCallback | None = None

    def _record_usage(self, stage: str, usage: Any) -> None:
        """Fold one call's ``response.usage`` into ``usage_by_stage`` (WP-5)."""
        entry = self.usage_by_stage.setdefault(stage or "", StageUsage())
        entry.calls += 1
        if usage is None:
            entry.estimated = True
            return
        entry.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        entry.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            entry.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)

    def usage_snapshot(self) -> dict[str, StageUsage]:
        """Deep copy of per-stage token usage accumulated so far (WP-5)."""
        return {k: v.model_copy(deep=True) for k, v in self.usage_by_stage.items()}

    def total_usage(self) -> StageUsage:
        """Sum of usage across all stages (WP-5)."""
        total = StageUsage()
        for u in self.usage_by_stage.values():
            total.prompt_tokens += u.prompt_tokens
            total.completion_tokens += u.completion_tokens
            total.reasoning_tokens += u.reasoning_tokens
            total.calls += u.calls
            total.estimated = total.estimated or u.estimated
        return total

    def set_retry_callback(self, cb: RetryCallback | None) -> None:
        """Install a default retry callback for subsequent ``generate()`` calls."""
        self._retry_callback = cb

    # ------------------------------------------------------------------
    # Per-stage config
    # ------------------------------------------------------------------

    def get_stage_config(self, stage: str) -> dict[str, Any]:
        """Return model/temperature for a given pipeline stage.

        Parameters
        ----------
        stage : str
            Pipeline stage name (e.g. ``"domain_analysis"``).

        Returns
        -------
        dict[str, Any]
            ``{"model": ..., "temperature": ...}`` with overrides applied.
        """
        # §6 P2: stage_llm_overrides is now `dict[str, StageLLMOverride]`,
        # though pre-Pydantic-validation tests may still pass bare dicts —
        # tolerate both via getattr-with-fallback.
        override = self._config.stage_llm_overrides.get(stage)
        model_override: str | None = None
        temp_override: float | None = None
        if override is not None:
            if isinstance(override, dict):
                model_override = override.get("model")
                temp_override = override.get("temperature")
            else:
                model_override = getattr(override, "model", None)
                temp_override = getattr(override, "temperature", None)
        return {
            "model": model_override if model_override is not None else self.model,
            "temperature": temp_override if temp_override is not None else self.temperature,
        }

    # ------------------------------------------------------------------
    # Auto-detect model
    # ------------------------------------------------------------------

    async def resolve_model(self) -> None:
        """If model is ``"__auto__"``, detect the first available model."""
        if self.model != "__auto__":
            return
        try:
            models = await self._client.models.list()
            if models.data:
                self.model = models.data[0].id
                logger.info("Auto-detected model: %s", self.model)
            else:
                raise MAGELLMError("No models available on the server")
        except MAGELLMError:
            raise
        except Exception as exc:
            raise MAGELLMError(f"Failed to auto-detect model: {exc}") from exc

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        response_schema: dict[str, Any] | None = None,
        stage: str | None = None,
        stream_callback: StreamCallback | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant content.

        Parameters
        ----------
        system_prompt : str
            System message.
        user_prompt : str
            User message.
        json_mode : bool
            If ``True``, request ``response_format={"type": "json_object"}``.
        stage : str | None
            Pipeline stage name (used for per-stage overrides and error messages).
        stream_callback : StreamCallback | None
            When supplied, the underlying API is invoked with
            ``stream=True`` and each delta is forwarded as
            ``stream_callback(stage, delta)``. The accumulated content is
            still returned in full. Non-fatal: if the API or transport
            doesn't support streaming the call falls back to a normal
            non-streaming request.

        Returns
        -------
        str
            Raw assistant message content.

        Raises
        ------
        MAGELLMError
            On any API or parsing failure.
        """
        stage_cfg = self.get_stage_config(stage) if stage else {"model": self.model, "temperature": self.temperature}

        # §6 P2 — local-provider JSON-mode fallback. When the provider
        # doesn't honor `response_format={"type":"json_object"}`, instead
        # append a strict-JSON instruction to the system prompt and let
        # `generate_json` extract the first {...} block via regex.
        use_json_fallback = bool(json_mode and self._config.json_mode_fallback)
        if use_json_fallback:
            system_prompt = system_prompt.rstrip() + _JSON_ONLY_SUFFIX

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": stage_cfg["model"],
            "messages": messages,
            "temperature": stage_cfg["temperature"],
        }
        if json_mode and not use_json_fallback:
            if self._config.use_schema_constrained_output and response_schema is not None:
                # P-7 — schema-guided / structured decoding where the provider
                # supports it (vLLM guided decoding, OpenRouter structured outputs).
                kwargs["response_format"] = {"type": "json_schema", "json_schema": response_schema}
            else:
                kwargs["response_format"] = {"type": "json_object"}

        stage_label = stage or ""
        effective_retry_cb = retry_callback if retry_callback is not None else self._retry_callback

        last_error: Exception | None = None
        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                if stream_callback is not None:
                    content = await self._generate_streaming(
                        kwargs, stream_callback, stage_label=stage_label, stage=stage
                    )
                else:
                    response = await self._client.chat.completions.create(**kwargs)
                    self._record_usage(stage_label, getattr(response, "usage", None))
                    content = response.choices[0].message.content
                if content is None:
                    raise MAGELLMError("LLM returned empty content", stage=stage)
                return content
            except MAGELLMError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("LLM call attempt %d/%d failed: %s", attempt, max_attempts, exc)
                # T9: a failed json_schema request usually means the provider/model
                # doesn't support structured outputs (e.g. deepseek-flash on
                # OpenRouter) — degrade to plain json_object for the remaining
                # attempts (T21 repair then handles any noisy JSON) instead of
                # hard-failing the whole call. Makes use_schema_constrained_output
                # safe to enable even when the runtime model can't honor it.
                rf = kwargs.get("response_format")
                if isinstance(rf, dict) and rf.get("type") == "json_schema":
                    logger.info("Downgrading response_format json_schema -> json_object (stage=%s)", stage)
                    kwargs["response_format"] = {"type": "json_object"}
                if attempt < max_attempts and effective_retry_cb is not None:
                    try:
                        effective_retry_cb(stage_label, attempt, exc)
                    except Exception as cb_exc:  # noqa: BLE001 — callback bugs must not break generation
                        logger.warning("retry_callback raised %s; continuing", cb_exc)

        raise MAGELLMError(
            f"LLM call failed after {max_attempts} attempts: {last_error}",
            stage=stage,
        )

    async def _generate_streaming(
        self,
        kwargs: dict[str, Any],
        stream_callback: StreamCallback,
        *,
        stage_label: str,
        stage: str | None,
    ) -> str:
        """Streaming variant of ``generate()``.

        Accumulates deltas, invokes ``stream_callback(stage, delta)`` per
        chunk, and returns the full content. Falls back to a non-streaming
        request when the provider doesn't honor ``stream=True`` (some
        local servers, custom proxies) — that fallback is detected by an
        AttributeError on the returned object, or a TypeError because the
        provider returned the non-streaming response shape directly.
        """
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        # WP-5: ask providers that support it to emit a final usage chunk.
        stream_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = await self._client.chat.completions.create(**stream_kwargs)
        except Exception as exc:
            # The provider may reject stream=True / stream_options entirely (e.g.
            # some response_format combinations). Fall back to non-streaming
            # rather than propagating — streaming is purely cosmetic.
            logger.warning("Streaming request rejected (%s); falling back to non-streaming", exc)
            response = await self._client.chat.completions.create(**kwargs)
            self._record_usage(stage_label, getattr(response, "usage", None))
            return response.choices[0].message.content

        # Streaming responses are AsyncIterables yielding chunks with a
        # `.choices[0].delta.content` attribute. If the provider returned
        # a non-streaming response object instead, treat that as a
        # graceful fallback.
        if not hasattr(stream, "__aiter__"):
            logger.warning("Streaming not supported by provider; returning non-streaming response")
            try:
                self._record_usage(stage_label, getattr(stream, "usage", None))
                return stream.choices[0].message.content
            except Exception as exc:  # noqa: BLE001 — defensive
                raise MAGELLMError(f"Unexpected non-streaming response shape: {exc}", stage=stage) from exc

        chunks: list[str] = []
        stream_usage: Any = None
        try:
            async for chunk in stream:
                # The final chunk (stream_options.include_usage) carries usage
                # and usually no delta — capture it before the empty-delta skip.
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    stream_usage = chunk_usage
                delta = _extract_chunk_delta(chunk)
                if not delta:
                    continue
                chunks.append(delta)
                try:
                    stream_callback(stage_label, delta)
                except Exception as exc:  # noqa: BLE001 — callback bugs must not break generation
                    logger.warning("stream_callback raised %s; continuing without further chunks", exc)
                    stream_callback = _noop_stream_callback
        except Exception as exc:
            raise MAGELLMError(f"Streaming failed mid-response: {exc}", stage=stage) from exc

        self._record_usage(stage_label, stream_usage)
        return "".join(chunks)

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stage: str | None = None,
        response_schema: dict[str, Any] | None = None,
        stream_callback: StreamCallback | None = None,
        retry_callback: RetryCallback | None = None,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response.

        Parameters
        ----------
        system_prompt : str
            System message.
        user_prompt : str
            User message.
        stage : str | None
            Pipeline stage name.
        stream_callback : StreamCallback | None
            Optional per-chunk callback (see :meth:`generate`).
        retry_callback : RetryCallback | None
            Optional retry-notification callback (see :meth:`generate`).

        Returns
        -------
        dict[str, Any]
            Parsed JSON object.

        Raises
        ------
        MAGELLMError
            On API or JSON parsing failure.
        """
        # T8: re-request the SAME stage on a parse failure instead of letting it
        # bubble up and restart the whole 7-stage pipeline (runs 731/773). The
        # earlier stages (domain/planning/dag) stay valid — only this call repeats.
        attempts = max(1, self._config.json_parse_retries + 1)
        last_exc: json.JSONDecodeError | None = None
        for attempt in range(1, attempts + 1):
            raw = await self.generate(
                system_prompt,
                user_prompt,
                json_mode=True,
                response_schema=response_schema,
                stage=stage,
                stream_callback=stream_callback,
                retry_callback=retry_callback,
            )
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                # T21: best-effort JSON repair before re-requesting — strip
                # markdown fences, drop trailing prose, close a truncated object.
                # Runs for ALL providers, not just the local json_mode_fallback path.
                repaired = _repair_json(raw)
                if isinstance(repaired, dict):
                    logger.info("generate_json: recovered malformed JSON via repair (stage=%s)", stage)
                    return repaired
                last_exc = exc
                if attempt < attempts:
                    logger.info(
                        "generate_json: parse failed (stage=%s, attempt %d/%d) — re-requesting",
                        stage, attempt, attempts,
                    )
                    continue
                logger.warning("generate_json: unparseable JSON output (stage=%s): %s", stage, raw[:500])
                raise MAGELLMError(f"Failed to parse LLM JSON output: {exc}", stage=stage) from exc
        # The loop always returns or raises above; this satisfies the type-checker.
        raise MAGELLMError(f"Failed to parse LLM JSON output: {last_exc}", stage=stage)

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ModelT],
        *,
        stage: str | None = None,
    ) -> ModelT:
        """Generate a response and validate it against a Pydantic model.

        Parameters
        ----------
        system_prompt : str
            System message.
        user_prompt : str
            User message.
        response_model : type[BaseModel]
            Pydantic model class to validate against.
        stage : str | None
            Pipeline stage name.

        Returns
        -------
        ModelT
            Validated Pydantic model instance of ``response_model``.

        Raises
        ------
        MAGELLMError
            On API, JSON, or validation failure.
        """
        # P-7 — when schema-constrained output is enabled, hand the provider the
        # model's JSON schema so it guides/constrains decoding (default off).
        response_schema = {
            "name": response_model.__name__,
            "schema": response_model.model_json_schema(),
            "strict": True,
        }
        data = await self.generate_json(
            system_prompt, user_prompt, stage=stage, response_schema=response_schema
        )
        try:
            return response_model.model_validate(data)
        except Exception as exc:
            raise MAGELLMError(
                f"Failed to validate LLM output against {response_model.__name__}: {exc}",
                stage=stage,
            ) from exc
