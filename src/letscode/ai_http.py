from __future__ import annotations

import threading
from typing import Any

import httpx
from langchain_openai import ChatOpenAI


class OpenAIHttpClients:
    """Shared OpenAI HTTP clients that can be closed during app shutdown."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: httpx.Client | None = None

    def chat_model(self, model_slug: str) -> ChatOpenAI:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client()

            return ChatOpenAI(
                model=model_slug,
                http_client=self._client,
            )

    def close(self) -> None:
        with self._lock:
            client = self._client
            self._client = None

        if client is not None:
            client.close()


default_openai_http_clients = OpenAIHttpClients()


def create_chat_model(
    model_slug: str,
    *,
    http_clients: OpenAIHttpClients | None = None,
    model: Any | None = None,
) -> Any:
    if model is not None:
        return model

    clients = http_clients or default_openai_http_clients
    return clients.chat_model(model_slug)
