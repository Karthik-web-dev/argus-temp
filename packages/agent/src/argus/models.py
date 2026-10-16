"""Single gateway for structured LLM/VLM calls.

Providers are OpenAI-compatible HTTP endpoints configured by tier. Application
roles depend on :class:`ModelGateway`, never on a provider SDK. Cassette replay
is deterministic and makes tests/demo runs independent of network access.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - surfaced when schema validation is used
    Draft202012Validator = None  # type: ignore[assignment,misc]

ModelTier = Literal["cheap", "strong", "strong_alt", "vlm"]
Prompt = str | list[dict[str, Any]]
TIERS: tuple[ModelTier, ...] = ("cheap", "strong", "strong_alt", "vlm")


class BudgetExceeded(RuntimeError):
    """A run attempted to exceed its configured model-call or token budget."""


class MalformedModelResponse(ValueError):
    """Provider output could not be decoded or did not satisfy its schema."""


class ModelConfigurationError(ValueError):
    """Model routes are missing or violate the role separation rules."""


class StructuredModel(Protocol):
    async def generate(self, *, tier: ModelTier, run_id: str, system: str, prompt: Prompt,
                       schema: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]: ...


class DisabledModel:
    async def generate(self, *, tier: ModelTier, run_id: str, system: str, prompt: Prompt,
                       schema: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
        raise RuntimeError("No model provider is configured; deterministic workflows remain available.")


@dataclass(frozen=True)
class ModelRoute:
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""

    @property
    def identity(self) -> tuple[str, str]:
        return self.provider.lower(), self.model

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass(frozen=True)
class ModelSettings:
    routes: dict[ModelTier, ModelRoute]
    explorer_tier: ModelTier = "cheap"
    red_team_tier: ModelTier = "strong_alt"
    mode: Literal["live", "replay", "record"] = "live"
    cassette_dir: Path = Path("fixtures/model_cassettes")
    requests_per_minute: int = 60
    max_calls_per_run: int = 100
    max_tokens_per_run: int = 100_000
    max_tokens_per_call: int = 8192
    request_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> ModelSettings:
        from dotenv import load_dotenv

        import argus.config  # noqa: F401 - ensure environment is loaded
        load_dotenv()
        routes: dict[ModelTier, ModelRoute] = {}
        for tier in TIERS:
            prefix = f"ARGUS_MODEL_{tier.upper()}_"
            routes[tier] = ModelRoute(
                provider=os.getenv(prefix + "PROVIDER", "openai_compatible"),
                model=os.getenv(prefix + "NAME", ""),
                base_url=os.getenv(prefix + "BASE_URL", os.getenv("ARGUS_MODEL_BASE_URL", "")),
                api_key=os.getenv(prefix + "API_KEY", os.getenv("ARGUS_MODEL_API_KEY", "")),
            )
        return cls(
            routes=routes,
            explorer_tier=os.getenv("ARGUS_EXPLORER_MODEL_TIER", "cheap"),  # type: ignore[arg-type]
            red_team_tier="strong_alt",
            mode=os.getenv("ARGUS_MODEL_MODE", "live"),  # type: ignore[arg-type]
            cassette_dir=Path(os.getenv("ARGUS_MODEL_CASSETTE_DIR", "fixtures/model_cassettes")),
            requests_per_minute=int(os.getenv("ARGUS_MODEL_RPM", "60")),
            max_calls_per_run=int(os.getenv("ARGUS_MODEL_MAX_CALLS", "100")),
            max_tokens_per_run=int(os.getenv("ARGUS_MODEL_MAX_TOKENS", "100000")),
            max_tokens_per_call=int(os.getenv("ARGUS_MODEL_MAX_TOKENS_PER_CALL", "8192")),
            request_timeout_seconds=float(os.getenv("ARGUS_MODEL_TIMEOUT_SECONDS", "60")),
        )

    def validate(self) -> None:
        explorer = self.routes[self.explorer_tier]
        red_team = self.routes[self.red_team_tier]
        # Guard only when both routes name actual models, so local deterministic
        # execution can start without credentials or provider configuration.
        if explorer.model and red_team.model and explorer.identity == red_team.identity:
            raise ModelConfigurationError(
                f"Explorer ({self.explorer_tier}) and Red Team ({self.red_team_tier}) "
                "must resolve to different provider/model pairs."
            )
        if self.requests_per_minute < 1 or self.max_calls_per_run < 1 or self.max_tokens_per_run < 1:
            raise ModelConfigurationError("Rate and run budgets must be positive integers.")
        if self.request_timeout_seconds <= 0:
            raise ModelConfigurationError("Model request timeout must be positive.")


@dataclass
class _RunUsage:
    calls: int = 0
    tokens: int = 0


class RunBudget:
    """Thread-safe per-run usage accounting shared by all gateway callers."""

    def __init__(self, *, max_calls: int, max_tokens: int) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self._usage: dict[str, _RunUsage] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, run_id: str, max_tokens: int) -> None:
        async with self._lock:
            usage = self._usage.setdefault(run_id, _RunUsage())
            if usage.calls + 1 > self.max_calls:
                raise BudgetExceeded(f"Run {run_id!r} exceeded the {self.max_calls}-call model budget.")
            if usage.tokens + max_tokens > self.max_tokens:
                raise BudgetExceeded(f"Run {run_id!r} would exceed the {self.max_tokens}-token model budget.")
            usage.calls += 1
            usage.tokens += max_tokens

    async def reconcile(self, run_id: str, reserved_tokens: int, actual_tokens: int) -> None:
        async with self._lock:
            usage = self._usage.setdefault(run_id, _RunUsage())
            updated = max(0, usage.tokens - reserved_tokens) + max(0, actual_tokens)
            if updated > self.max_tokens:
                usage.tokens = updated
                raise BudgetExceeded(f"Run {run_id!r} exceeded the {self.max_tokens}-token model budget.")
            usage.tokens = updated

    async def snapshot(self, run_id: str) -> dict[str, int]:
        async with self._lock:
            usage = self._usage.get(run_id, _RunUsage())
            return {"calls": usage.calls, "tokens": usage.tokens}


class SharedRateLimiter:
    """A shared sliding-window limiter for all tiers of one gateway."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.limit = requests_per_minute
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self.limit:
                    self._timestamps.append(now)
                    return
                delay = max(0.01, 60 - (now - self._timestamps[0]))
            await asyncio.sleep(delay)


@dataclass
class CassetteStore:
    """JSON cassette reader/writer keyed by the full request contract."""

    directory: Path
    mode: Literal["live", "replay", "record"]
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @staticmethod
    def key(*, tier: ModelTier, route: ModelRoute, system: str, prompt: Prompt,
           schema: dict[str, Any]) -> str:
        canonical = json.dumps({"tier": tier, "provider": route.provider, "model": route.model,
                               "system": system, "prompt": prompt, "schema": schema},
                              sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def read(self, key: str) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        return await asyncio.to_thread(lambda: json.loads(path.read_text(encoding="utf-8")))

    async def write(self, key: str, response: dict[str, Any]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write_sync, key, response)

    def _write_sync(self, key: str, response: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


class OpenAICompatibleProvider:
    """Small stdlib adapter; model/provider changes are configuration-only."""

    async def generate(self, *, route: ModelRoute, system: str, prompt: Prompt,
                       schema: dict[str, Any], max_tokens: int,
                       timeout_seconds: float = 60.0) -> tuple[Any, int]:
        if not route.configured:
            raise ModelConfigurationError("Model route is unconfigured; set its NAME and BASE_URL environment variables.")
        url = route.base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body: dict[str, Any] = {
            "model": route.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            # strict=True plus the full contract schema routinely stalls OpenRouter
            # and Gemini-compatible endpoints for minutes. Local jsonschema
            # validation in ModelGateway still fail-closes on malformed output.
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.get("title", "argus_response"), "strict": False, "schema": schema,
            }},
            # Keep reasoning models bounded so they don't consume all max_tokens on internal CoT
            "reasoning": {"effort": "low"},
        }
        headers = {"Content-Type": "application/json"}
        if route.api_key:
            headers["Authorization"] = f"Bearer {route.api_key}"

        max_http_attempts = 3
        data: dict[str, Any] | None = None
        for attempt_idx in range(max_http_attempts):
            try:
                # The outer deadline is a wall-clock limit; HTTP client read timeouts
                # alone can be extended indefinitely by a slow trickle of bytes.
                async with asyncio.timeout(timeout_seconds):
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(
                            timeout_seconds,
                            connect=min(10.0, timeout_seconds),
                        )
                    ) as client:
                        response = await client.post(url, json=body, headers=headers)
                        response.raise_for_status()
                        data = response.json()
                        break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt_idx < max_http_attempts - 1:
                    retry_after = exc.response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else float(2 * (attempt_idx + 1))
                    except ValueError:
                        delay = float(2 * (attempt_idx + 1))
                    delay = min(delay, 15.0)
                    logging.getLogger(__name__).warning(
                        "Model provider returned HTTP 429; retrying in %.1fs (attempt %d/%d)",
                        delay, attempt_idx + 1, max_http_attempts,
                    )
                    await asyncio.sleep(delay)
                    continue
                if exc.response.status_code == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    suffix = f" Retry-After: {retry_after} seconds." if retry_after else ""
                    raise RuntimeError(
                        "Model provider rate-limited this request (HTTP 429). Check the account quota, "
                        f"rate limit, and model access.{suffix}"
                    ) from exc
                raise RuntimeError(
                    f"Model provider returned HTTP {exc.response.status_code}."
                ) from exc
            except TimeoutError as exc:
                raise RuntimeError(
                    f"Model provider request exceeded its {timeout_seconds:g}s deadline."
                ) from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"Model provider request failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise MalformedModelResponse("Provider returned an invalid JSON response body.") from exc

        if data is None:
            raise RuntimeError("Model provider returned no response data.")

        try:
            content = data["choices"][0]["message"]["content"]
            if content is None:
                raise MalformedModelResponse("Provider returned null content; token limit may have been exhausted by model reasoning.")
            if isinstance(content, str):
                cleaned = content.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    cleaned = "\n".join(lines).strip()
                value = json.loads(cleaned)
            else:
                value = content
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise MalformedModelResponse("Provider returned no valid JSON response content.") from exc
        usage = data.get("usage", {})
        tokens = int(usage.get("total_tokens", 0))
        return value, tokens


class ModelGateway:
    """The only model call path, with schema validation, budgets and cassettes."""

    def __init__(self, settings: ModelSettings | None = None, *,
                 provider: OpenAICompatibleProvider | None = None,
                 cassette_store: CassetteStore | None = None) -> None:
        self.settings = settings or ModelSettings.from_env()
        self.settings.validate()
        self.provider = provider or OpenAICompatibleProvider()
        self.cassettes = cassette_store or CassetteStore(self.settings.cassette_dir, self.settings.mode)
        self.rate_limiter = SharedRateLimiter(self.settings.requests_per_minute)
        self.budget = RunBudget(max_calls=self.settings.max_calls_per_run,
                                max_tokens=self.settings.max_tokens_per_run)

    async def generate(self, *, tier: ModelTier, run_id: str, system: str, prompt: Prompt,
                       schema: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
        route = self.settings.routes[tier]
        token_cap = min(max_tokens or self.settings.max_tokens_per_call,
                        self.settings.max_tokens_per_call)
        key = self.cassettes.key(tier=tier, route=route, system=system, prompt=prompt, schema=schema)
        cached = await self.cassettes.read(key)
        if self.settings.mode == "replay":
            if cached is None:
                raise FileNotFoundError(f"No model cassette for request {key} ({tier}).")
            self._validate(cached["response"], schema)
            await self.budget.reserve(run_id, min(int(cached.get("tokens", 0)), token_cap))
            return cached["response"]

        for attempt in range(2):
            await self.budget.reserve(run_id, token_cap)
            try:
                await self.rate_limiter.acquire()
                raw, tokens = await self.provider.generate(route=route, system=system, prompt=prompt,
                                                           schema=schema, max_tokens=token_cap,
                                                           timeout_seconds=self.settings.request_timeout_seconds)
            except MalformedModelResponse:
                await self.budget.reconcile(run_id, token_cap, 0)
                if attempt == 0:
                    logging.getLogger(__name__).warning(
                        "Malformed response from model tier %s; retrying once with the same schema.", tier,
                    )
                    continue
                raise
            except Exception:
                await self.budget.reconcile(run_id, token_cap, 0)
                raise
            try:
                result = self._validate(raw, schema)
            except MalformedModelResponse:
                await self.budget.reconcile(run_id, token_cap, tokens)
                if attempt == 0:
                    logging.getLogger(__name__).warning(
                        "Model output for tier %s failed schema validation; retrying once.", tier,
                    )
                    continue
                raise
            except Exception:
                await self.budget.reconcile(run_id, token_cap, tokens)
                raise
            await self.budget.reconcile(run_id, token_cap, tokens)
            if self.settings.mode == "record":
                await self.cassettes.write(key, {"request_key": key, "tier": tier,
                                                 "model": route.model, "tokens": tokens,
                                                 "response": result})
            return result
        raise MalformedModelResponse("Model output failed schema validation after one retry.")

    @staticmethod
    def _validate(value: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise MalformedModelResponse("Structured model output must be a JSON object.")
        if Draft202012Validator is None:
            raise RuntimeError("JSON Schema validation requires the 'jsonschema' package.")
        try:
            Draft202012Validator.check_schema(schema)
            errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
        except Exception as exc:
            raise ValueError(f"Invalid response JSON Schema: {exc}") from exc
        if errors:
            details = "; ".join(error.message for error in errors[:3])
            raise MalformedModelResponse(f"Model output does not match its JSON Schema: {details}")
        return value


def validate_model_configuration(settings: ModelSettings | None = None) -> None:
    """Startup hook for CLI/API entrypoints that initialize model services."""
    (settings or ModelSettings.from_env()).validate()
