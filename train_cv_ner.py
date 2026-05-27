from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import BertTokenizerFast, TFAutoModel
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
ARTIFACTS_DIR = PROJECT_ROOT / "exported_artifacts"

LABEL_LIST = ["O", "B-SKILL", "I-SKILL", "B-ROLE", "I-ROLE"]

MODEL_NAME = "bert-base-multilingual-cased"
MAX_SEQ_LEN = 128
IGNORE_LABEL_ID = -100

BATCH_SIZE = 2
EPOCHS = 5

def load_cv_data(synthetic_dir: Path = SYNTHETIC_DIR) -> list[dict[str, Any]]:
    if not synthetic_dir.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {synthetic_dir}")

    paths = sorted(synthetic_dir.glob("cv_*.json"))
    if not paths:
        raise FileNotFoundError(f"Tidak ada file cv_*.json di {synthetic_dir}") 

    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            records.append(json.load(f))
    return records

def inspect_cv_dataset(records: list[dict[str, Any]]) -> None:
  """Cetak ringkasan dataset untuk validasi awal."""
  print("==== MAPAN CV NER — Inspeksi Dataset =====\n")
  print(f"Jumlah CV: {len(records)}\n")
  sectors: dict[str, int] = {}
  token_lengths: list[int] = []
  label_mismatches = 0
  for rec in records:
    sector = rec.get("sector", "?")
    sectors[sector] = sectors.get(sector, 0) + 1
    tokens = rec.get("tokens", [])
    labels = rec.get("labels", [])
    token_lengths.append(len(tokens))
    if len(tokens) != len(labels):
      label_mismatches += 1
  print("==== Per sektor =====")
  for sector, count in sorted(sectors.items()):
    print(f"  {sector}: {count}")
  print()
  print("==== Panjang tokens per CV =====")
  print(f"  min: {min(token_lengths)}")
  print(f"  max: {max(token_lengths)}")
  print(f"  rata-rata: {sum(token_lengths) / len(token_lengths):.1f}")
  print()
  print(f"==== CV dengan tokens != labels: {label_mismatches} =====\n")
  # Contoh 1 CV
  sample = records[0]
  print("==== Contoh CV pertama =====")
  print(f"  id: {sample.get('id')}")
  print(f"  sector: {sample.get('sector')}")
  print(f"  tokens (5 pertama): {sample.get('tokens', [])[:5]}")
  print(f"  labels (5 pertama): {sample.get('labels', [])[:5]}")


# 2. Validasi Label IOB dan split
def validate_records(records: list[dict[str, Any]]) -> None:
  allowed = set(LABEL_LIST)
  errors: list[str] = []

  for rec in records:
    cv_id = rec.get("id", "?")
    labels = rec.get("labels", [])

    for i, lab in enumerate(labels):
      if lab not in allowed:
        errors.append(f"{cv_id}: label tidak dikenal -> {lab}")
        continue
      if lab.startswith("I-"):
        ent = lab[2:]
        if i == 0:
          errors.append(f"{cv_id}: I- di posisi 0 -> {lab}")
        else:
          prev = labels[i - 1]
          if prev not in {f"B-{ent}", f"I-{ent}"}:
            errors.append(
              f"{cv_id}: IOB invalid {prev} -> {lab} (index {i})"
            )

  if errors:
    print("==== ERROR validasi label =====")
    for e in errors[:20]:
      print(" ", e)
    raise ValueError(f"Validasi gagal: {len(errors)} masalah")
  print("==== Validasi label IOB: OK ====\n")

def build_label_mappings() -> tuple[dict[str, int], dict[int, str]]:
  label2id = {label: idx for idx, label in enumerate(LABEL_LIST)}
  id2label = {idx: label for label, idx in label2id.items()}
  return label2id, id2label

def split_train_validation(
  records: list[dict[str, Any]],
  test_size: float = 0.2,
  random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  sectors = [rec["sector"] for rec in records]
  train_recs,val_recs = train_test_split(
    records, test_size = test_size,
    random_state = random_state,
  )
  return train_recs, val_recs

def inspect_split(
  train_recs: list[dict[str, Any]],
  val_recs: list[dict[str, Any]],
  label2id: dict[str, int],
) -> None:
  print("==== Train / Validation split =====\n")
  print(f"Train: {len(train_recs)} CV")
  print(f"Val:   {len(val_recs)} CV\n")
  def count_sectors(recs: list[dict[str, Any]]) -> dict[str, int]:
      out: dict[str, int] = {}
      for r in recs:
          out[r["sector"]] = out.get(r["sector"], 0) + 1
      return out
  print("Train per sektor:", count_sectors(train_recs))
  print("Val per sektor:  ", count_sectors(val_recs))
  print()
  print("Label -> ID:", label2id)


# 3. Tokenizer BERT + align label ke subword

def tokenize_and_align_labels(
    records: list[dict[str, Any]],
    label2id: dict[str, int],
    tokenizer: BertTokenizerFast,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Tokenisasi daftar CV menggunakan BERT tokenizer dan align label IOB
  ke level subword.

  - Subword pertama dari setiap kata menggunakan label asli (di-mapping ke id).
  - Subword lanjutan dan token spesial ([CLS], [SEP], [PAD]) diberi IGNORE_LABEL_ID.
  """
  # Kita pakai tokens yang sudah ada di JSON agar konsisten dengan labeling.
  token_seqs = [rec["tokens"] for rec in records]

  encodings = tokenizer(
      token_seqs,
      is_split_into_words=True,
      truncation=True,
      padding="max_length",
      max_length=MAX_SEQ_LEN,
      return_attention_mask=True,
  )

  all_label_ids: list[list[int]] = []

  for batch_idx, rec in enumerate(records):
    word_ids = encodings.word_ids(batch_index=batch_idx)
    labels = rec["labels"]
    aligned_ids: list[int] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
      if word_id is None:
        # Token spesial atau padding.
        aligned_ids.append(IGNORE_LABEL_ID)
      else:
        label_str = labels[word_id]
        label_id = label2id[label_str]
        if word_id != previous_word_id:
          # Subword pertama untuk kata ini.
          aligned_ids.append(label_id)
        else:
          # Subword lanjutan untuk kata yang sama.
          aligned_ids.append(IGNORE_LABEL_ID)
      previous_word_id = word_id

    all_label_ids.append(aligned_ids)

  input_ids = np.array(encodings["input_ids"], dtype=np.int32)
  attention_mask = np.array(encodings["attention_mask"], dtype=np.int32)
  label_ids = np.array(all_label_ids, dtype=np.int64)

  return input_ids, attention_mask, label_ids


class GlobalContextAttention(tf.keras.layers.Layer):
  """
  Attention sederhana untuk memberi bobot penting tiap token
  berdasarkan konteks global sequence.
  """

  def __init__(self, hidden_size: int, **kwargs: Any):
    super().__init__(**kwargs)
    self.hidden_size = hidden_size
    self.proj = tf.keras.layers.Dense(hidden_size, activation="tanh")
    self.score = tf.keras.layers.Dense(1, activation=None)

  def call(self, sequence_output: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    # sequence_output: (batch, seq_len, hidden)
    # attention_mask : (batch, seq_len)
    h = self.proj(sequence_output)  # (B, L, H)
    logits = self.score(h)  # (B, L, 1)

    # Mask padding token supaya tidak ikut attention.
    mask = tf.cast(attention_mask[:, :, tf.newaxis], tf.float32)  # (B, L, 1)
    minus_inf = tf.ones_like(logits) * (-1e9)
    masked_logits = tf.where(mask > 0, logits, minus_inf)

    weights = tf.nn.softmax(masked_logits, axis=1)  # (B, L, 1)
    context = tf.reduce_sum(weights * sequence_output, axis=1)  # (B, H)

    # Broadcast context kembali ke setiap token.
    seq_len = tf.shape(sequence_output)[1]
    context_expanded = tf.repeat(context[:, tf.newaxis, :], repeats=seq_len, axis=1)
    # (B, L, H)

    # Gabungkan local token + global context.
    return tf.concat([sequence_output, context_expanded], axis=-1)  # (B, L, 2H)

  def get_config(self) -> dict[str, Any]:
    config = super().get_config()
    config.update({"hidden_size": self.hidden_size})
    return config


def build_ner_model(
  model_name: str = MODEL_NAME,
  num_labels: int = len(LABEL_LIST),
) -> tf.keras.Model:
  input_ids = tf.keras.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")
  attention_mask = tf.keras.Input(
      shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="attention_mask"
  )

  base_model = TFAutoModel.from_pretrained(model_name)
  bert_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
  sequence_output = bert_outputs.last_hidden_state  # (B, L, H)

  hidden_size = sequence_output.shape[-1]
  if hidden_size is None:
    raise ValueError("Hidden size BERT tidak terdeteksi.")

  attn_layer = GlobalContextAttention(hidden_size=int(hidden_size))
  enriched = attn_layer(sequence_output, attention_mask)  # (B, L, 2H)

  x = tf.keras.layers.Dropout(0.2)(enriched)
  logits = tf.keras.layers.Dense(num_labels, name="token_logits")(x)  # (B, L, C)

  model = tf.keras.Model(
      inputs={"input_ids": input_ids, "attention_mask": attention_mask},
      outputs=logits,
      name="cv_ner_multilingual_bert",
  )
  return model


def masked_sparse_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """
  Hitung loss hanya untuk token dengan label != IGNORE_LABEL_ID (-100).
  """
  y_true = tf.cast(y_true, tf.int32)
  mask = tf.not_equal(y_true, IGNORE_LABEL_ID)

  safe_y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
  per_token_loss = tf.keras.losses.sparse_categorical_crossentropy(
      safe_y_true, y_pred, from_logits=True
  )
  mask_f = tf.cast(mask, tf.float32)
  return tf.reduce_sum(per_token_loss * mask_f) / (tf.reduce_sum(mask_f) + 1e-8)


def masked_token_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  y_true = tf.cast(y_true, tf.int32)
  pred_ids = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
  mask = tf.not_equal(y_true, IGNORE_LABEL_ID)

  matches = tf.equal(y_true, pred_ids)
  matches_f = tf.cast(tf.logical_and(matches, mask), tf.float32)
  mask_f = tf.cast(mask, tf.float32)
  return tf.reduce_sum(matches_f) / (tf.reduce_sum(mask_f) + 1e-8)


if __name__ == "__main__":
  records = load_cv_data()
  inspect_cv_dataset(records)

  validate_records(records)
  label2id, id2label = build_label_mappings()

  train_recs, val_recs = split_train_validation(records)
  inspect_split(train_recs, val_recs, label2id)

  # STEP 3: contoh tokenisasi + align label untuk train dan validation.
  print("\n==== Tokenisasi BERT + align label (contoh) =====\n")
  tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

  train_input_ids, train_attention_mask, train_label_ids = tokenize_and_align_labels(
      train_recs, label2id, tokenizer
  )
  val_input_ids, val_attention_mask, val_label_ids = tokenize_and_align_labels(
      val_recs, label2id, tokenizer
  )

  print("Train shapes:")
  print("  input_ids     :", train_input_ids.shape)
  print("  attention_mask:", train_attention_mask.shape)
  print("  labels        :", train_label_ids.shape)

  print("\nVal shapes:")
  print("  input_ids     :", val_input_ids.shape)
  print("  attention_mask:", val_attention_mask.shape)
  print("  labels        :", val_label_ids.shape)

  print("\n==== Build & compile NER model (Step 4) =====\n")
  ner_model = build_ner_model(model_name=MODEL_NAME, num_labels=len(LABEL_LIST))
  ner_model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
      loss=masked_sparse_categorical_crossentropy,
      metrics=[masked_token_accuracy],
  )
  ner_model.summary(line_length=120)

  # STEP 5: Training sederhana + simpan artefak
  print("\n==== Training NER model (Step 5) =====\n")

  # Konversi label ke int32 untuk TensorFlow.
  train_label_ids_tf = train_label_ids.astype("int32")
  val_label_ids_tf = val_label_ids.astype("int32")

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (
          {
              "input_ids": train_input_ids,
              "attention_mask": train_attention_mask,
          },
          train_label_ids_tf,
      )
  ).shuffle(buffer_size=len(train_recs)).batch(BATCH_SIZE)

  val_dataset = tf.data.Dataset.from_tensor_slices(
      (
          {
              "input_ids": val_input_ids,
              "attention_mask": val_attention_mask,
          },
          val_label_ids_tf,
      )
  ).batch(BATCH_SIZE)

  callbacks: list[tf.keras.callbacks.Callback] = [
      tf.keras.callbacks.EarlyStopping(
          monitor="val_loss",
          patience=2,
          restore_best_weights=True,
      )
  ]

  history = ner_model.fit(
      train_dataset,
      validation_data=val_dataset,
      epochs=EPOCHS,
      callbacks=callbacks,
  )

  print("\n==== Evaluasi di validation set =====\n")
  eval_results = ner_model.evaluate(val_dataset, return_dict=True)
  print(eval_results)

  # Simpan artefak model + label mapping.
  ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
  model_path = ARTIFACTS_DIR / "cv_ner_multilingual.keras"
  print(f"\nMenyimpan model ke: {model_path}")
  ner_model.save(model_path)

  label_map_path = ARTIFACTS_DIR / "cv_ner_label_mappings.pkl"
  print(f"Menyimpan label mappings ke: {label_map_path}")
  joblib.dump(
      {
          "label2id": dict(label2id),
          "id2label": dict(id2label),
      },
      label_map_path,
  )

  tokenizer_path = ARTIFACTS_DIR / "cv_ner_tokenizer"
  print(f"Menyimpan tokenizer ke: {tokenizer_path}")
  tokenizer.save_pretrained(tokenizer_path)

  print("\n==== Step 5 selesai. Artefak NER siap untuk inference. =====\n")