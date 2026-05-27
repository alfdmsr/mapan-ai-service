"""
MAPAN Model 3 — Step 9.3: generate training pairs (CV skills × role requirements).

Usage:
  python generate_match_pairs.py
  python generate_match_pairs.py --inspect-arrays
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np

from app.services.career_data import (
    RoleEntry,
    RolesCatalog,
    SkillVocab,
    load_roles_catalog,
    load_skill_vocab,
    role_requirements_multihot,
    skills_to_multihot,
)
from app.services.skill_normalization import normalize_skill_list

PROJECT_ROOT = Path(__file__).resolve().parent
CV_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_PATH = PROJECT_ROOT / "data" / "career" / "match_pairs.jsonl"
MATCH_PAIRS_PATH = OUTPUT_PATH

_TOKEN_TRAILING = str.maketrans("", "", string.punctuation)


def load_cv_records(cv_dir: Path = CV_DIR) -> list[dict]:
    records: list[dict] = []
    for path in sorted(cv_dir.glob("cv_*.json")):
        with path.open(encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def extract_raw_skills_from_cv(tokens: list[str], labels: list[str]) -> list[str]:
    if len(tokens) != len(labels):
        raise ValueError(
            f"Panjang tokens ({len(tokens)}) dan labels ({len(labels)}) tidak sama"
        )

    raw_skills: list[str] = []
    i = 0
    while i < len(labels):
        if labels[i] != "B-SKILL":
            i += 1
            continue

        parts: list[str] = [tokens[i].strip().translate(_TOKEN_TRAILING).strip()]
        j = i + 1
        while j < len(labels) and labels[j] == "I-SKILL":
            token = tokens[j].strip().translate(_TOKEN_TRAILING).strip()
            if token:
                parts.append(token)
            j += 1

        phrase = " ".join(p for p in parts if p).strip()
        if phrase:
            raw_skills.append(phrase)
        i = j
    return raw_skills


def cv_to_user_canonical_skills(record: dict, vocab: SkillVocab) -> list[str]:
    raw = extract_raw_skills_from_cv(record["tokens"], record["labels"])
    return normalize_skill_list(
        raw,
        alias_to_canonical=vocab.alias_to_canonical,
        canonical_set=set(vocab.canonical_set),
    )


def compute_overlap_label(
    user_skills: list[str],
    required_skills: list[str],
    threshold: float,
) -> tuple[float, int]:
    required_set = set(required_skills)
    if not required_set:
        return 0.0, 0

    overlap_count = len(set(user_skills) & required_set)
    ratio = overlap_count / len(required_set)
    label = 1 if ratio >= threshold else 0
    return ratio, label


def build_pair_record(
    record: dict,
    role: RoleEntry,
    user_skills: list[str],
    threshold: float,
) -> dict:
    ratio, label = compute_overlap_label(
        user_skills, role.required_skills, threshold
    )
    return {
        "pair_id": f"{record['id']}__{role.role_id}",
        "cv_id": record["id"],
        "cv_sector": record["sector"],
        "role_id": role.role_id,
        "role_sector": role.sector,
        "user_canonical_skills": user_skills,
        "overlap_ratio": round(ratio, 4),
        "label": label,
    }


def generate_all_pairs(
    records: list[dict],
    catalog: RolesCatalog,
    vocab: SkillVocab,
) -> list[dict]:
    pairs: list[dict] = []
    threshold = catalog.default_match_threshold

    for record in records:
        user_skills = cv_to_user_canonical_skills(record, vocab)
        if not user_skills:
            print(f"[WARN] {record['id']}: tidak ada skill ter-map, dilewati")
            continue

        for role in catalog.roles:
            pairs.append(build_pair_record(record, role, user_skills, threshold))
    return pairs


def validate_pairs(
    pairs: list[dict],
    vocab: SkillVocab,
    catalog: RolesCatalog,
    threshold: float,
) -> None:
    for pair in pairs:
        if pair["label"] not in (0, 1):
            raise ValueError(f"Label invalid pada {pair['pair_id']}")

        role = catalog.role_by_id.get(pair["role_id"])
        if role is None:
            raise ValueError(f"role_id tidak dikenal: {pair['role_id']}")

        for skill in pair["user_canonical_skills"]:
            if skill not in vocab.canonical_set:
                raise ValueError(f"Skill tidak di vocab: {skill!r} pada {pair['pair_id']}")

        expected_ratio, expected_label = compute_overlap_label(
            pair["user_canonical_skills"],
            role.required_skills,
            threshold,
        )
        if pair["label"] != expected_label:
            raise ValueError(
                f"Label tidak konsisten pada {pair['pair_id']}: "
                f"stored={pair['label']} expected={expected_label}"
            )
        if abs(pair["overlap_ratio"] - round(expected_ratio, 4)) > 1e-6:
            raise ValueError(f"overlap_ratio tidak konsisten pada {pair['pair_id']}")


def write_match_pairs_jsonl(pairs: list[dict], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def load_match_pairs_jsonl(path: Path = OUTPUT_PATH) -> list[dict]:
    pairs: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def pairs_to_arrays(
    pairs: list[dict],
    vocab: SkillVocab,
    catalog: RolesCatalog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(pairs)
    x_user = np.zeros((n, vocab.dim), dtype=np.float32)
    x_role = np.zeros((n, vocab.dim), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)

    for i, pair in enumerate(pairs):
        x_user[i] = skills_to_multihot(pair["user_canonical_skills"], vocab)
        role = catalog.role_by_id[pair["role_id"]]
        x_role[i] = role_requirements_multihot(role, vocab)
        y[i] = float(pair["label"])

    return x_user, x_role, y


def train_val_split_by_cv(
    pairs: list[dict],
    *,
    val_cv_ids: set[str] | None = None,
    val_fraction: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    all_cv_ids = sorted({p["cv_id"] for p in pairs})
    if val_cv_ids is None:
        n_val = max(1, int(len(all_cv_ids) * val_fraction))
        val_cv_ids = set(all_cv_ids[-n_val:])

    train_pairs = [p for p in pairs if p["cv_id"] not in val_cv_ids]
    val_pairs = [p for p in pairs if p["cv_id"] in val_cv_ids]
    return train_pairs, val_pairs


def print_pair_statistics(pairs: list[dict]) -> None:
    pos = sum(p["label"] for p in pairs)
    neg = len(pairs) - pos
    print(f"  Total pasangan : {len(pairs)}")
    print(f"  Positif (label=1): {pos} ({100 * pos / max(len(pairs), 1):.1f}%)")
    print(f"  Negatif (label=0): {neg} ({100 * neg / max(len(pairs), 1):.1f}%)")

    by_sector_match: dict[str, int] = {}
    for p in pairs:
        if p["label"] == 1 and p["cv_sector"] == p["role_sector"]:
            by_sector_match[p["cv_sector"]] = by_sector_match.get(p["cv_sector"], 0) + 1
    if by_sector_match:
        print("  Positif same-sector:")
        for sector in sorted(by_sector_match):
            print(f"    {sector}: {by_sector_match[sector]}")


def run_generate(*, inspect_arrays: bool = False) -> list[dict]:
    print("==== MAPAN Step 9.3 — Generate match_pairs =====\n")

    vocab = load_skill_vocab()
    catalog = load_roles_catalog(vocab=vocab)
    records = load_cv_records()
    print(f"CV loaded: {len(records)}")
    print(f"Vocab dim: {vocab.dim} | Roles: {len(catalog.roles)}")
    print(f"Threshold: {catalog.default_match_threshold}\n")

    # Contoh trace satu CV
    sample = next((r for r in records if r["id"] == "cv_keu_0001"), records[0])
    raw = extract_raw_skills_from_cv(sample["tokens"], sample["labels"])
    canonical = cv_to_user_canonical_skills(sample, vocab)
    print(f"Trace {sample['id']}:")
    print(f"  raw skills ({len(raw)}): {raw}")
    print(f"  canonical ({len(canonical)}): {canonical}\n")

    pairs = generate_all_pairs(records, catalog, vocab)
    validate_pairs(pairs, vocab, catalog, catalog.default_match_threshold)

    print("Statistik pasangan:")
    print_pair_statistics(pairs)

    write_match_pairs_jsonl(pairs)
    print(f"\nDisimpan: {OUTPUT_PATH} ({len(pairs)} baris)\n")

    if inspect_arrays:
        train_pairs, val_pairs = train_val_split_by_cv(pairs)
        x_user_tr, x_role_tr, y_tr = pairs_to_arrays(train_pairs, vocab, catalog)
        x_user_va, x_role_va, y_va = pairs_to_arrays(val_pairs, vocab, catalog)
        print("Array shapes (train):")
        print(f"  X_user: {x_user_tr.shape}  X_role: {x_role_tr.shape}  y: {y_tr.shape}")
        print("Array shapes (val):")
        print(f"  X_user: {x_user_va.shape}  X_role: {x_role_va.shape}  y: {y_va.shape}")
        val_cv = sorted({p['cv_id'] for p in val_pairs})
        print(f"  Val CV ids: {val_cv}")

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MAPAN career match pairs")
    parser.add_argument(
        "--inspect-arrays",
        action="store_true",
        help="Cetak shape tensor train/val setelah generate",
    )
    args = parser.parse_args()
    run_generate(inspect_arrays=args.inspect_arrays)


if __name__ == "__main__":
    main()
