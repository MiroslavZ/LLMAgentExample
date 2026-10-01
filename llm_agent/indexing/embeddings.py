"""Локальные эмбеддинги: attention-mask mean pooling и L2 normalization."""

from importlib.metadata import version
from pathlib import Path

E5 = "intfloat/multilingual-e5-small"
RUBERT = "DeepPavlov/rubert-base-cased"


class Embedder:
    def __init__(self, model=E5, revision=None, cache=None, offline=False):
        if model not in (E5, RUBERT):
            raise ValueError("Поддерживаются multilingual-e5-small и rubert-base-cased")
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils.hub import cached_file

        torch.set_num_threads(min(4, torch.get_num_threads()))
        kwargs = dict(revision=revision, cache_dir=cache, local_files_only=offline)
        config_path = cached_file(model, "config.json", **kwargs)
        resolved_revision = Path(config_path).parent.name
        kwargs["revision"] = resolved_revision
        # Both supported models are BERT-family tokenizers, not Mistral.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=True, fix_mistral_regex=False, **kwargs)
        self.model = AutoModel.from_pretrained(model, **kwargs).eval()
        self.limit = min(512, self.model.config.max_position_embeddings)
        self.metadata = {
            "model": model, "revision": resolved_revision,
            "dimension": self.model.config.hidden_size, "limit": self.limit,
            "passage_prefix": "passage: " if model == E5 else "",
            "query_prefix": "query: " if model == E5 else "",
            "pooling": "attention_mask_mean", "normalized": True,
            "versions": {p: version(p) for p in ("torch", "transformers", "numpy")},
        }

    def encode(self, texts, *, query=False, batch_size=16):
        import torch
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")
        if any(not text.strip() for text in texts):
            raise ValueError("Пустой вход модели")
        prefix = self.metadata["query_prefix" if query else "passage_prefix"]
        inputs = [prefix + text for text in texts]
        for text in inputs:
            length = len(self.tokenizer.encode(text, add_special_tokens=True))
            if length > self.limit:
                raise ValueError(f"Вход модели: {length} токенов, лимит {self.limit}")
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                batch = self.tokenizer(inputs[start:start + batch_size], padding=True,
                                       truncation=False, return_tensors="pt")
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                vectors.extend(torch.nn.functional.normalize(pooled, p=2, dim=1).tolist())
        return vectors
