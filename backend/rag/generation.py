import torch

from rag.prompts import PromptBuilder
from rag.providers.generation.factory import create_generation_provider
from rag.types import GenerationConfig, RetrievedChunk


class GeneratorEngine:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.prompt_builder = PromptBuilder(config)
        self.provider = create_generation_provider(config)

    def generate_answer(self, *, query: str, chunks: list[RetrievedChunk], intent: str) -> str:
        if not chunks:
            return "Не удалось найти релевантную информацию в базе знаний."

        return self._generate_with_retries(query=query, chunks=chunks, intent=intent)

    def _generate_with_retries(self, *, query: str, chunks: list[RetrievedChunk], intent: str) -> str:
        is_cuda = torch.cuda.is_available()
        context_budgets = self.config.gpu_context_budgets if is_cuda else self.config.cpu_context_budgets
        output_budgets = self.config.gpu_max_new_tokens if is_cuda else self.config.cpu_max_new_tokens

        attempts = list(zip(context_budgets, output_budgets, strict=False))
        if not attempts:
            fallback_output_budget = output_budgets[0] if output_budgets else 250
            attempts = [(self.config.max_context_tokens, fallback_output_budget)]

        for context_budget, output_budget in attempts:
            try:
                context = self.build_context(chunks, max_context_tokens=context_budget)
                messages = self.prompt_builder.build_messages(query=query, context=context, intent=intent)
                return self.provider.generate_from_messages(
                    messages,
                    max_new_tokens=output_budget,
                ).strip()
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        raise RuntimeError(
            "Недостаточно GPU-памяти для генерации ответа. "
            "Попробуйте уменьшить RAG_MAX_CONTEXT_TOKENS, выбрать модель меньше "
            "или запустить backend на сервере с большей VRAM."
        )

    def build_context(self, chunks: list[RetrievedChunk], *, max_context_tokens: int) -> str:
        context_sections: list[str] = []
        used_tokens = 0

        for idx, chunk in enumerate(chunks, start=1):
            locator_line = (
                f"Локация: {chunk.source_locator}\n" if chunk.source_locator else ""
            )
            section = (
                f"[Источник {idx}] {chunk.title}\n"
                f"Путь: {chunk.path}\n"
                f"{locator_line}"
                f"Текст: {chunk.text}"
            )
            section_tokens = self.provider.count_tokens(section)

            if used_tokens and used_tokens + section_tokens > max_context_tokens:
                break

            if not used_tokens and section_tokens > max_context_tokens:
                section = self.provider.truncate_text(section, max_context_tokens)
                section_tokens = self.provider.count_tokens(section)

            context_sections.append(section)
            used_tokens += section_tokens

        if not context_sections:
            context_sections.append(self.provider.truncate_text(chunks[0].text, max_context_tokens))

        return "\n\n".join(context_sections)
