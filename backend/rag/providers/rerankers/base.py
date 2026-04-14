from typing import Protocol


class RerankerProvider(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        ...
