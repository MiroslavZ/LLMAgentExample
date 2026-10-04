"""Локальные эмбеддинги с настройками модели и совместимостью старых индексов."""

from importlib.metadata import version
from pathlib import Path

E5 = "intfloat/multilingual-e5-small"
RUBERT = "DeepPavlov/rubert-base-cased"
_E5_RETRIEVAL = {E5, "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large"}


def embedding_options(metadata: dict) -> dict:
    """Восстановить настройки нового адаптера; старый формат остаётся неизменным."""
    if "backend" not in metadata:
        return {}
    return {key: metadata[key] for key in ("backend", "query_prefix", "passage_prefix")}


class Embedder:
    def __init__(self, model=E5, revision=None, cache=None, offline=False, *,
                 backend=None, query_prefix=None, passage_prefix=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Укажите идентификатор embedding-модели Hugging Face")
        if any(value is not None and not isinstance(value, str) for value in (query_prefix, passage_prefix)):
            raise ValueError("Префиксы модели должны быть строками")
        if backend is None:
            backend = "legacy" if model in (E5, RUBERT) and query_prefix is None and passage_prefix is None else "sentence_transformers"
        if backend not in ("legacy", "sentence_transformers"):
            raise ValueError("Неизвестный адаптер эмбеддингов")
        if backend == "legacy" and (model not in (E5, RUBERT) or query_prefix is not None or passage_prefix is not None):
            raise ValueError("Старый адаптер предназначен только для исходных индексов E5-small и RuBERT")
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils.hub import cached_file

        torch.set_num_threads(min(4, torch.get_num_threads()))
        kwargs = dict(revision=revision, cache_dir=cache, local_files_only=offline)
        config_path = cached_file(model, "config.json", **kwargs)
        resolved_revision = Path(config_path).parent.name
        kwargs["revision"] = resolved_revision
        if backend == "sentence_transformers":
            self._load_sentence_transformer(model, kwargs, query_prefix, passage_prefix)
            return
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

    def _load_sentence_transformer(self, model, kwargs, query_prefix, passage_prefix):
        from sentence_transformers import SentenceTransformer
        from transformers.utils.hub import cached_file

        # Не допускаем автоматический fallback ST на случайный mean pooling
        # для произвольной базовой языковой модели без embedding-конфигурации.
        modules = cached_file(model, "modules.json", **kwargs, _raise_exceptions_for_missing_entries=False)
        if modules is None:
            raise ValueError("Модель должна содержать конфигурацию Sentence Transformers (modules.json)")
        self.model = SentenceTransformer(
            model, revision=kwargs["revision"], cache_folder=kwargs["cache_dir"],
            local_files_only=kwargs["local_files_only"], device="cpu", trust_remote_code=False,
        ).eval()
        self.tokenizer = self.model.tokenizer
        if not self.tokenizer.is_fast:
            raise ValueError("Для разбиения документов требуется fast tokenizer с offsets")
        self.limit = self.model.max_seq_length
        dimension = self.model.get_embedding_dimension()
        if type(self.limit) is not int or self.limit < 1 or type(dimension) is not int or dimension < 1:
            raise ValueError("Модель не задала длину входа или размерность текстового вектора")
        prompts = self.model.prompts
        default = prompts.get(self.model.default_prompt_name, "")
        if query_prefix is None:
            query_prefix = "query: " if model in _E5_RETRIEVAL else prompts.get("query", default)
        if passage_prefix is None:
            passage_prefix = "passage: " if model in _E5_RETRIEVAL else next(
                (prompts[key] for key in ("document", "passage", "corpus") if key in prompts), default,
            )
        self.metadata = {
            "model": model, "revision": kwargs["revision"], "backend": "sentence_transformers",
            "dimension": dimension, "limit": self.limit,
            "passage_prefix": passage_prefix, "query_prefix": query_prefix,
            "pooling": "model_defined", "normalized": True,
            "versions": {p: version(p) for p in ("torch", "transformers", "numpy", "sentence-transformers")},
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
        if self.metadata.get("backend") == "sentence_transformers":
            if not texts:
                return []
            encode = self.model.encode_query if query else self.model.encode_document
            return encode(
                texts, prompt=prefix, batch_size=batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ).tolist()
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
