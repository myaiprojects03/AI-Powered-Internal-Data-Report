"""The only module that talks to a language model.

Every provider implements one method: `complete(system, user) -> str` returning
the model's raw text (expected to be JSON). Nothing else in the codebase knows
which model is in use, so swapping models is a config change:

    LLM_PROVIDER=ollama  LLM_MODEL=llama3.1:8b        (local, free)
    LLM_PROVIDER=gemini  GEMINI_API_KEY=...            (free tier, default gemini-3.8-flash)
    LLM_PROVIDER=groq    GROQ_API_KEY=...              (free tier)
    LLM_PROVIDER=openai_compatible  OPENAI_COMPAT_BASE_URL=... (LM Studio, OpenRouter free models, vLLM)
    LLM_PROVIDER=offline                               (no model; rule-based, for tests and demos)

To add a provider: subclass `LLMClient`, implement `complete`, register it in `PROVIDERS`.
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod

import requests

from .config import SETTINGS


class LLMError(RuntimeError):
    """The model could not be reached or used (auth, network, bad config). Retrying will not help."""


class LLMOutputError(LLMError):
    """The model was reached but failed to produce a usable answer (truncated, invalid JSON).

    This is a failed attempt, not a dead provider: the caller should try again.
    """


def _is_generation_failure(status: int, body: str) -> bool:
    return status == 400 and any(k in body for k in ("json_validate_failed", "failed_generation", "max completion tokens"))


class LLMClient(ABC):
    provider: str = "base"
    default_model: str = ""

    def __init__(self, model: str | None = None, temperature: float | None = None, timeout: int | None = None):
        self.model = model or self.default_model
        self.temperature = SETTINGS.llm_temperature if temperature is None else temperature
        self.timeout = timeout or SETTINGS.llm_timeout_s
        self.max_tokens = SETTINGS.llm_max_tokens

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the raw model output for one system + user message pair."""

    # Shared HTTP helper with backoff on rate limits and transient server errors.
    def _post(self, url: str, payload: dict, headers: dict | None = None, retries: int = 3) -> dict:
        for attempt in range(retries):
            try:
                r = requests.post(url, json=payload, headers=headers or {}, timeout=self.timeout)
            except requests.ReadTimeout as e:
                # Connected, but the model was still writing: a slow answer, not a dead provider.
                # Hand it back as a failed attempt so the caller retries with the shorter prompt.
                raise LLMOutputError(f"{self.provider}: no answer within {self.timeout}s") from e
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise LLMError(f"{self.provider}: connection failed: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(float(r.headers.get("retry-after", 2 ** (attempt + 1))))
                continue
            if r.status_code >= 400 and _is_generation_failure(r.status_code, r.text):
                raise LLMOutputError(f"{self.provider}: model failed to finish a valid answer: {r.text[:300]}")
            if r.status_code == 404:
                raise LLMError(f"{self.provider}: model '{self.model}' not found (HTTP 404). "
                               f"Check the model name against the provider's model list. {r.text[:200]}")
            if r.status_code >= 400:
                raise LLMError(f"{self.provider}: HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        raise LLMError(f"{self.provider}: exhausted retries")


def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise LLMError(f"Environment variable {name} is not set")
    return val


class OllamaClient(LLMClient):
    """Local models via Ollama. Free, no data leaves the machine."""
    provider = "ollama"
    default_model = "llama3.1:8b"

    def complete(self, system: str, user: str) -> str:
        data = self._post(f"{SETTINGS.ollama_host.rstrip('/')}/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        })
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise LLMError(f"ollama: unexpected response: {json.dumps(data)[:300]}") from e


class GeminiClient(LLMClient):
    """Google Gemini REST API (free tier covers Flash models)."""
    provider = "gemini"
    default_model = "gemini-3.8-flash"     # 2.5-flash is closed to new accounts; check the models endpoint
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def complete(self, system: str, user: str) -> str:
        data = self._post(
            f"{self.base_url}/models/{self.model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json",
                                     "maxOutputTokens": self.max_tokens},
            },
            headers={"x-goog-api-key": _require_env("GEMINI_API_KEY")},
        )
        try:
            cand = data["candidates"][0]
            if cand.get("finishReason") == "MAX_TOKENS":
                raise LLMOutputError("gemini: answer cut off at the token limit")
            return "".join(p.get("text", "") for p in cand["content"]["parts"])
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"gemini: unexpected response: {json.dumps(data)[:300]}") from e


class OpenAICompatibleClient(LLMClient):
    """Any server exposing /chat/completions: Groq, OpenRouter, LM Studio, vLLM, llama.cpp server."""
    provider = "openai_compatible"
    default_model = ""
    base_url_env = "OPENAI_COMPAT_BASE_URL"
    api_key_env = "OPENAI_COMPAT_API_KEY"
    default_base_url = ""
    token_param = "max_tokens"          # widely supported name (LM Studio, vLLM, llama.cpp, OpenRouter)

    def extra_params(self) -> dict:
        """Provider/model specific request fields. Override in subclasses."""
        return {}

    def complete(self, system: str, user: str) -> str:
        base = os.getenv(self.base_url_env, self.default_base_url).rstrip("/")
        if not base:
            raise LLMError(f"Environment variable {self.base_url_env} is not set")
        key = os.getenv(self.api_key_env, "")
        data = self._post(
            f"{base}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                self.token_param: self.max_tokens,
                **self.extra_params(),
            },
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )
        try:
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise LLMOutputError(f"{self.provider}: answer cut off at the token limit")
            content = choice["message"]["content"]
            if not content:
                raise LLMOutputError(f"{self.provider}: empty answer")
            return content
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"{self.provider}: unexpected response: {json.dumps(data)[:300]}") from e


class GroqClient(OpenAICompatibleClient):
    provider = "groq"
    default_model = "openai/gpt-oss-120b"    # Groq production model; check console.groq.com/docs/models
    base_url_env = "GROQ_BASE_URL"
    api_key_env = "GROQ_API_KEY"
    default_base_url = "https://api.groq.com/openai/v1"
    token_param = "max_completion_tokens"

    def extra_params(self) -> dict:
        # Reasoning models (gpt-oss) think before answering; that thinking uses the output budget.
        if self.model.startswith("openai/gpt-oss") and SETTINGS.llm_reasoning_effort:
            return {"reasoning_effort": SETTINGS.llm_reasoning_effort}
        return {}

    def complete(self, system: str, user: str) -> str:
        _require_env(self.api_key_env)
        return super().complete(system, user)


class OfflineClient(LLMClient):
    """Deterministic rule-based writer with the same interface. No model involved.

    Used for tests, CI and demos where no model is reachable. Reports produced with it
    are labelled as such so nobody mistakes them for model output.
    """
    provider = "offline"
    default_model = "rule-based"

    def complete(self, system: str, user: str) -> str:
        from .offline_writer import write_from_prompt
        return write_from_prompt(user)


PROVIDERS: dict[str, type[LLMClient]] = {
    "ollama": OllamaClient,
    "gemini": GeminiClient,
    "groq": GroqClient,
    "openai_compatible": OpenAICompatibleClient,
    "offline": OfflineClient,
}


def get_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    name = (provider or SETTINGS.llm_provider).lower()
    if name not in PROVIDERS:
        raise LLMError(f"Unknown provider '{name}'. Options: {', '.join(PROVIDERS)}")
    # LLM_MODEL belongs to LLM_PROVIDER. When another provider is chosen, use that provider's default
    # unless a model is passed explicitly; never send one provider's model name to another.
    configured = SETTINGS.llm_model if name == SETTINGS.llm_provider.lower() else ""
    return PROVIDERS[name](model=model or configured or None)
