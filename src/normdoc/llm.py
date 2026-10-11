"""LLM-слой проекта: единая обёртка над несколькими провайдерами генерации.

Поддерживаются два протокола:

* ``LLM_PROVIDER=openai`` (по умолчанию) — любой OpenAI-совместимый HTTP API:
  локальный ``llama-server`` (llama.cpp), Ollama, vLLM, LM Studio. Под этим
  протоколом запускаются все локальные РФ-модели в GGUF: bonsai, Meno-Lite,
  GigaChat3.x, T-lite, QVikhr, RuadaptQwen, Saiga и др.
* ``LLM_PROVIDER=gigachat`` — облачный API Sber GigaChat (OAuth + REST,
  сертификаты Минцифры). Требует ``GIGACHAT_CREDENTIALS``.

Функция :func:`call_llm` — единая точка входа; :class:`LlamaCpp` — адаптер
под LlamaIndex. Для обратной совместимости сохранено имя :func:`call_llamacpp`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Sequence

import requests
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import CustomLLM

from normdoc.config import (
    GIGACHAT_API_URL,
    GIGACHAT_AUTH_URL,
    GIGACHAT_CREDENTIALS,
    GIGACHAT_MODEL,
    GIGACHAT_SCOPE,
    GIGACHAT_VERIFY_SSL,
    LLM_BASE_URL,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_REASONING,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)

_ROLE_MAP = {
    MessageRole.SYSTEM: "system",
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
}


# ---------------------------------------------------------------------------
# OpenAI-совместимый провайдер (llama.cpp / Ollama / vLLM)
# ---------------------------------------------------------------------------
def call_llamacpp(
    messages: list[dict[str, str]],
    *,
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout: float = LLM_TIMEOUT,
    reasoning: str = LLM_REASONING,
) -> tuple[str, dict]:
    """Отправляет сообщения в OpenAI-совместимый сервер, возвращает (текст, usage)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning and reasoning not in {"auto", "off"}:
        payload["reasoning_effort"] = reasoning

    resp = requests.post(
        f"{base_url}/chat/completions", json=payload, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return content, usage


# Историческая совместимость: раньше поддерживался только llama.cpp.
call_openai_compat = call_llamacpp


# ---------------------------------------------------------------------------
# GigaChat API (Sber): OAuth-токен + chat/completions
# ---------------------------------------------------------------------------
_TOKEN_CACHE: dict[str, Any] = {"value": None, "expires_at": 0.0}


def _gigachat_access_token(
    *,
    auth_url: str = GIGACHAT_AUTH_URL,
    credentials: str = GIGACHAT_CREDENTIALS,
    scope: str = GIGACHAT_SCOPE,
    timeout: float = LLM_TIMEOUT,
    verify: bool = GIGACHAT_VERIFY_SSL,
) -> str:
    """Возвращает access_token GigaChat (с кэшем до истечения срока)."""
    now = time.time()
    cached = _TOKEN_CACHE.get("value")
    if cached and float(_TOKEN_CACHE.get("expires_at", 0)) - 30 > now:
        return str(cached)
    if not credentials:
        raise RuntimeError(
            "GIGACHAT_CREDENTIALS не задан. Укажите base64(client_id:client_secret) "
            "из developers.sber.ru или используйте LLM_PROVIDER=openai."
        )
    headers = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    resp = requests.post(
        auth_url, headers=headers, data={"scope": scope}, timeout=timeout, verify=verify
    )
    resp.raise_for_status()
    data = resp.json()
    _TOKEN_CACHE["value"] = data["access_token"]
    expires_at = data.get("expires_at")
    _TOKEN_CACHE["expires_at"] = (float(expires_at) / 1000.0) if expires_at else now + 1800
    return str(data["access_token"])


def call_gigachat(
    messages: list[dict[str, str]],
    *,
    api_url: str = GIGACHAT_API_URL,
    model: str = GIGACHAT_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout: float = LLM_TIMEOUT,
    credentials: str = GIGACHAT_CREDENTIALS,
    scope: str = GIGACHAT_SCOPE,
    verify: bool = GIGACHAT_VERIFY_SSL,
) -> tuple[str, dict]:
    """Запрос к облачному GigaChat. Возвращает (текст, usage)."""
    token = _gigachat_access_token(
        credentials=credentials, scope=scope, timeout=timeout, verify=verify
    )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(
        f"{api_url}/chat/completions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
        verify=verify,
    )
    resp.raise_for_status()
    data = resp.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {}) or {}
    # GigaChat отдаёт prompt_tokens/completion_tokens/total_tokens — совместимо.
    return content, usage


def call_llm(
    messages: list[dict[str, str]],
    *,
    provider: str = LLM_PROVIDER,
    **kwargs: Any,
) -> tuple[str, dict]:
    """Единая точка входа: маршрутизирует по провайдеру генерации."""
    if provider == "gigachat":
        allowed = {
            k: v
            for k, v in kwargs.items()
            if k in {"api_url", "model", "temperature", "max_tokens", "timeout",
                     "credentials", "scope", "verify"}
        }
        return call_gigachat(messages, **allowed)
    return call_llamacpp(messages, **kwargs)


# ---------------------------------------------------------------------------
# Адаптер под LlamaIndex
# ---------------------------------------------------------------------------
class LlamaCpp(CustomLLM):
    """LlamaIndex-совместимая обёртка над выбранным провайдером LLM."""

    base_url: str = LLM_BASE_URL
    model: str = LLM_MODEL
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = LLM_MAX_TOKENS
    context_window: int = LLM_CONTEXT_WINDOW
    timeout: float = LLM_TIMEOUT
    reasoning: str = LLM_REASONING
    provider: str = LLM_PROVIDER

    # Заполняется последним вызовом chat(): полезно для Langfuse-трасс.
    last_usage: dict = {}

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=True,
        )

    def _api_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [
            {"role": _ROLE_MAP.get(m.role, "user"), "content": m.content or ""}
            for m in messages
        ]

    def _call(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
        if self.provider == "gigachat":
            return call_llm(
                messages,
                provider="gigachat",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        return call_llm(
            messages,
            provider="openai",
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            reasoning=self.reasoning,
        )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        text, usage = self._call(self._api_messages(messages))
        self.last_usage = usage
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            raw=usage,
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text, usage = self._call([{"role": "user", "content": prompt}])
        self.last_usage = usage
        return CompletionResponse(text=text, raw=usage)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        def _gen() -> CompletionResponseGen:
            yield self.complete(prompt, formatted=formatted, **kwargs)

        return _gen()


def build_llm() -> LlamaCpp:
    return LlamaCpp()
