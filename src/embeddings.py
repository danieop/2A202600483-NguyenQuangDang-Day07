from __future__ import annotations

import hashlib
import math
import os

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"
OPENAI_API_KEY_ENV = "SHOPAIKEY_API_KEY"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_USER_AGENT_ENV = "OPENAI_USER_AGENT"

DEFAULT_OPENAI_BASE_URL = "https://api.shopaikey.com/v1"
DEFAULT_OPENAI_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"Missing environment variable: {OPENAI_API_KEY_ENV}")
        base_url = os.getenv(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)
        user_agent = os.getenv(OPENAI_USER_AGENT_ENV, DEFAULT_OPENAI_USER_AGENT)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"User-Agent": user_agent},
        )

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [[float(value) for value in item.embedding] for item in response.data]


class OpenAIChatLLM:
    """OpenAI chat-completions wrapper used by the RAG agent."""

    def __init__(self, model_name: str = OPENAI_CHAT_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        api_key = os.getenv(OPENAI_API_KEY_ENV)
        if not api_key:
            raise ValueError(f"Missing environment variable: {OPENAI_API_KEY_ENV}")
        base_url = os.getenv(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)
        user_agent = os.getenv(OPENAI_USER_AGENT_ENV, DEFAULT_OPENAI_USER_AGENT)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"User-Agent": user_agent},
        )

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        choice = response.choices[0]
        content = choice.message.content if choice and choice.message else ""
        return content or ""


_mock_embed = MockEmbedder()
