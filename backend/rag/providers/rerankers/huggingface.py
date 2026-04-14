import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from settings import settings


class HuggingFaceRerankerProvider:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = self._resolve_device()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        scores: list[float] = []
        batch_size = max(settings.reranker_batch_size, 1)
        for start in range(0, len(passages), batch_size):
            batch_passages = passages[start : start + batch_size]
            inputs = self.tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                padding=True,
                truncation=True,
                max_length=settings.reranker_max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            if logits.ndim == 2:
                if logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits[:, -1]
            else:
                batch_scores = logits

            scores.extend(batch_scores.detach().float().cpu().tolist())

        return scores

    def _resolve_device(self) -> torch.device:
        requested_device = settings.reranker_device
        if requested_device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested_device)
        return torch.device("cpu")
