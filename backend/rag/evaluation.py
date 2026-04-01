import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from rag.service import RAGService


@dataclass(frozen=True)
class EvaluationSample:
    query: str
    expected_source_contains: list[str]
    expected_answer_contains: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    total: int
    retrieval_hit_rate: float
    answer_hit_rate: float


def load_evaluation_dataset(path: Path) -> list[EvaluationSample]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))

    return [
        EvaluationSample(
            query=record["query"],
            expected_source_contains=record.get("expected_source_contains", []),
            expected_answer_contains=record.get("expected_answer_contains", []),
        )
        for record in records
    ]


def evaluate_rag_service(service: RAGService, samples: list[EvaluationSample]) -> EvaluationResult:
    if not samples:
        return EvaluationResult(total=0, retrieval_hit_rate=0.0, answer_hit_rate=0.0)

    retrieval_scores: list[float] = []
    answer_scores: list[float] = []

    for sample in samples:
        retrieval_result = service.retrieval.search(sample.query)
        chat_result = service.chat(sample.query)

        retrieval_hit = _contains_expected_source(
            sources=[chunk.path for chunk in retrieval_result.chunks],
            expected_fragments=sample.expected_source_contains,
        )
        answer_hit = _contains_expected_answer(
            answer=chat_result.answer,
            expected_fragments=sample.expected_answer_contains,
        )

        retrieval_scores.append(1.0 if retrieval_hit else 0.0)
        answer_scores.append(1.0 if answer_hit else 0.0)

    return EvaluationResult(
        total=len(samples),
        retrieval_hit_rate=round(mean(retrieval_scores), 3),
        answer_hit_rate=round(mean(answer_scores), 3),
    )


def _contains_expected_source(*, sources: list[str], expected_fragments: list[str]) -> bool:
    if not expected_fragments:
        return True

    lowered_sources = [source.lower() for source in sources]
    return any(
        any(fragment.lower() in source for source in lowered_sources)
        for fragment in expected_fragments
    )


def _contains_expected_answer(*, answer: str, expected_fragments: list[str]) -> bool:
    if not expected_fragments:
        return True

    lowered_answer = answer.lower()
    return all(fragment.lower() in lowered_answer for fragment in expected_fragments)
