"""Совместимость ragas с новым langchain-community.

ragas (0.4.x) при импорте делает
`from langchain_community.chat_models.vertexai import ChatVertexAI`,
но в langchain-community >= 0.4 этот модуль удалён. Мы не используем
VertexAI, поэтому подсовываем пустую заглушку в sys.modules до импорта ragas.

Импортировать этот модуль ПЕРВЫМ, перед `import ragas`.
"""

from __future__ import annotations

import sys
import types


def install_ragas_compat() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        import langchain_community  # noqa: F401
    except Exception:
        return

    module = types.ModuleType(name)

    class ChatVertexAI:  # pragma: no cover - заглушка, не используется
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise RuntimeError("ChatVertexAI недоступен: используется локальный LLM")

    module.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[name] = module
