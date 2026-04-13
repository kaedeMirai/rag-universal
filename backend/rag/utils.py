import re


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]


def normalize_scores(raw_scores: dict[int, float], *, reverse: bool = False) -> dict[int, float]:
    if not raw_scores:
        return {}

    values = list(raw_scores.values())
    min_value = min(values)
    max_value = max(values)

    if max_value == min_value:
        return {doc_id: 1.0 for doc_id in raw_scores}

    normalized: dict[int, float] = {}
    for doc_id, value in raw_scores.items():
        score = (value - min_value) / (max_value - min_value)
        normalized[doc_id] = 1.0 - score if reverse else score

    return normalized
