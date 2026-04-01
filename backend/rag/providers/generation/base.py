from typing import Protocol


class GenerationProvider(Protocol):
    def generate_from_messages(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...

    def truncate_text(self, text: str, max_tokens: int) -> str:
        ...
