from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from transformers import BertTokenizerFast, TFBertModel

from train_cv_ner import GlobalContextAttention

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "exported_artifacts"

MODEL_NAME = "bert-base-multilingual-cased"
MAX_SEQ_LEN = 128

MODEL_PATH = ARTIFACT_DIR / "cv_ner_multilingual.keras"
LABEL_MAP_PATH = ARTIFACT_DIR / "cv_ner_label_mappings.pkl"
TOKENIZER_PATH = ARTIFACT_DIR / "cv_ner_tokenizer"


def load_cv_artifacts() -> tuple[tf.keras.Model, BertTokenizerFast, dict[int, str]]:
    """Load CV NER model, tokenizer, and label mapping for inference."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model CV NER tidak ditemukan: {MODEL_PATH}")
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label mapping tidak ditemukan: {LABEL_MAP_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "GlobalContextAttention": GlobalContextAttention,
            "TFBertModel": TFBertModel,
        },
        compile=False,
    )
    tokenizer_source = TOKENIZER_PATH if TOKENIZER_PATH.exists() else MODEL_NAME
    tokenizer = BertTokenizerFast.from_pretrained(
        tokenizer_source,
        local_files_only=TOKENIZER_PATH.exists(),
    )
    label_maps: dict[str, Any] = joblib.load(LABEL_MAP_PATH)

    id2label_raw = label_maps["id2label"]
    id2label = {int(label_id): label for label_id, label in id2label_raw.items()}
    return model, tokenizer, id2label


def extract_entities(
    tokens: list[str],
    labels: list[str],
    entity_type: str,
) -> list[str]:
    """Convert IOB token labels into entity spans."""
    entities: list[str] = []
    current_tokens: list[str] = []

    begin_label = f"B-{entity_type}"
    inside_label = f"I-{entity_type}"

    for token, label in zip(tokens, labels):
        if label == begin_label:
            if current_tokens:
                entities.append(" ".join(current_tokens))
            current_tokens = [token]
        elif label == inside_label and current_tokens:
            current_tokens.append(token)
        else:
            if current_tokens:
                entities.append(" ".join(current_tokens))
                current_tokens = []

    if current_tokens:
        entities.append(" ".join(current_tokens))

    return entities


class CVParser:
    """Singleton inference: load CV NER artifacts once, predict many times."""

    def __init__(self) -> None:
        self._model: tf.keras.Model | None = None
        self._tokenizer: BertTokenizerFast | None = None
        self._id2label: dict[int, str] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        model, tokenizer, id2label = load_cv_artifacts()
        self._model = model
        self._tokenizer = tokenizer
        self._id2label = id2label

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def parse(self, raw_text: str) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("CVParser belum di-load. Panggil .load() dulu.")
        if self._model is None or self._tokenizer is None or self._id2label is None:
            raise RuntimeError("Artefak CVParser belum lengkap.")

        tokens = raw_text.strip().split()
        if not tokens:
            return {"tokens": [], "labels": [], "skills": [], "roles": []}

        encoding = self._tokenizer(
            [tokens],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            return_attention_mask=True,
            return_tensors="np",
        )
        logits = self._model.predict(
            {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            },
            verbose=0,
        )

        pred_ids = np.argmax(logits[0], axis=-1)
        word_ids = encoding.word_ids(batch_index=0)

        word_labels: list[str] = []
        previous_word_id: int | None = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous_word_id:
                previous_word_id = word_id
                continue

            word_labels.append(self._id2label[int(pred_ids[token_idx])])
            previous_word_id = word_id

        aligned_tokens = tokens[: len(word_labels)]
        skills = extract_entities(aligned_tokens, word_labels, entity_type="SKILL")
        roles = extract_entities(aligned_tokens, word_labels, entity_type="ROLE")

        return {
            "tokens": aligned_tokens,
            "labels": word_labels,
            "skills": skills,
            "roles": roles,
        }


cv_parser = CVParser()
