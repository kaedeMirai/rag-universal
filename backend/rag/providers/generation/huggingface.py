import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import settings


class HuggingFaceGenerationProvider:
    def __init__(
        self,
        *,
        model_name: str,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dtype = self._resolve_dtype(settings.generation_dtype)
        device_map = settings.generation_device_map if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=model_dtype,
            device_map=device_map,
        )
        self.model.eval()
        self.input_device = self._resolve_input_device()

    def generate_from_messages(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}

        with torch.no_grad():
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.do_sample:
                generation_kwargs["temperature"] = self.temperature
                generation_kwargs["top_p"] = self.top_p

            generated = self.model.generate(
                **generation_kwargs,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            generated[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

    def count_tokens(self, text: str) -> int:
        return self.tokenizer(text, return_tensors="pt")["input_ids"].shape[1]

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if tokens.shape[1] <= max_tokens:
            return text

        truncated = tokens[:, :max_tokens]
        return self.tokenizer.decode(truncated[0], skip_special_tokens=True)

    def _resolve_input_device(self) -> torch.device:
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map:
            for target in hf_device_map.values():
                if isinstance(target, str) and target.startswith("cuda"):
                    return torch.device(target)
            for target in hf_device_map.values():
                if isinstance(target, str) and target == "cpu":
                    return torch.device("cpu")

        try:
            for parameter in self.model.parameters():
                if parameter.device.type != "meta":
                    return parameter.device
        except Exception:
            pass

        return torch.device("cpu")

    def _resolve_dtype(self, value: str) -> torch.dtype:
        mapping = {
            "auto": torch.float16 if torch.cuda.is_available() else torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        normalized = value.strip().lower()
        if normalized not in mapping:
            raise ValueError(f"Unsupported generation dtype: {value}")
        return mapping[normalized]
