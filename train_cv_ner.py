from __future__ import annotations

from cProfile import label
import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
ARTIFACTS_DIR = PROJECT_ROOT / "exported_artifacts"

LABEL_LIST = ["O", "B-SKILL", "I-SKILL", "B-ROLE", "I-ROLE"]

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


if __name__ == "__main__":
  records = load_cv_data()
  inspect_cv_dataset(records)

  validate_records(records)
  label2id, id2label = build_label_mappings()

  train_recs, val_recs = split_train_validation(records)
  inspect_split(train_recs, val_recs, label2id)