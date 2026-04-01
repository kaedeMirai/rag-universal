from rag.generation import GeneratorEngine
from rag.retrieval import RetrievalEngine
from rag.types import ChatResult, RAGProfile


class RAGService:
    def __init__(self, profile: RAGProfile):
        self.profile = profile
        self.retrieval = RetrievalEngine(profile.retrieval)
        self.generator = GeneratorEngine(profile.generation)

    def chat(self, query: str) -> ChatResult:
        retrieval_result = self.retrieval.search(query)

        if not retrieval_result.chunks:
            return ChatResult(
                answer="Не удалось найти релевантную информацию в базе знаний.",
                sources=[],
                confidence=0.0,
            )

        answer = self.generator.generate_answer(
            query=query,
            chunks=retrieval_result.chunks,
            intent=retrieval_result.intent,
        )
        return ChatResult(
            answer=answer,
            sources=self.extract_sources(retrieval_result.chunks),
            confidence=self.estimate_confidence(
                retrieval_result.chunks,
                intent=retrieval_result.intent,
            ),
        )

    def extract_sources(self, chunks) -> list[str]:
        unique_sources: list[str] = []
        seen = set()

        for chunk in chunks:
            if not chunk.path or chunk.path in seen:
                continue
            unique_sources.append(chunk.path)
            seen.add(chunk.path)

        return unique_sources

    def estimate_confidence(self, chunks, *, intent: str) -> float:
        top_score = chunks[0].score
        second_score = chunks[1].score if len(chunks) > 1 else 0.0
        score_gap = max(0.0, top_score - second_score)
        exact_match = chunks[0].exact_reference_match

        confidence = (top_score * 0.65) + (score_gap * 0.35)
        if intent == "document_lookup":
            confidence += 0.15 * exact_match

        return round(max(0.0, min(1.0, confidence)), 3)
