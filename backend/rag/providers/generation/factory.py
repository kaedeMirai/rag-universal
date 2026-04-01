from settings import settings
from rag.providers.generation.huggingface import HuggingFaceGenerationProvider
from rag.types import GenerationConfig


def create_generation_provider(config: GenerationConfig):
    if settings.generation_provider == "huggingface":
        return HuggingFaceGenerationProvider(
            model_name=config.model_name,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )

    raise ValueError(f"Unsupported generation provider: {settings.generation_provider}")
