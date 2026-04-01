from settings import settings
from rag.providers.embeddings.huggingface import HuggingFaceEmbeddingProvider


def create_embedding_provider():
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddingProvider(settings.embedding_model_name)

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
