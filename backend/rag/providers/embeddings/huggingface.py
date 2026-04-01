from sentence_transformers import SentenceTransformer

from settings import settings


class HuggingFaceEmbeddingProvider:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device=settings.embedding_device)

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
