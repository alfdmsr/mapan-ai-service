from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


if __name__ == "__main__":
  records = load_cv_data()
  inspect_cv_dataset(records)