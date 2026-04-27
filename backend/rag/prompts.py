from rag.types import GenerationConfig


class PromptBuilder:
    def __init__(self, config: GenerationConfig):
        self.config = config

    def build_messages(self, *, query: str, context: str, intent: str) -> list[dict[str, str]]:
        system_prompt = self.config.system_prompt
        if intent in {"document_lookup", "reference_lookup"}:
            system_prompt += f" {self.config.document_lookup_prompt_suffix}"

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Контекст:\n{context}\n\n"
                    f"Вопрос: {query}\n\n"
                    "Сформируй краткий и точный ответ. Если в контексте есть готовая "
                    "прямая формулировка, приведи короткую цитату в кавычках. "
                    "Если у источника указана страница, упомяни ее. "
                    "Не выдумывай цитаты и страницы."
                ),
            },
        ]
