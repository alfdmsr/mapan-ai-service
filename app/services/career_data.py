"""
MAPAN Model 3 — load skill vocabulary & role catalog (Step 9.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.skill_normalization import build_alias_lookup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CAREER_DATA_DIR = PROJECT_ROOT / "data" / "career"
SKILL_VOCAB_PATH = CAREER_DATA_DIR / "skill_vocab.json"
ROLES_CATALOG_PATH = CAREER_DATA_DIR / "roles_catalog.json"

PILOT_SECTORS = frozenset(
    {
        "Teknologi",
        "Keuangan_Admin",
        "Manufaktur_Engineering",
        "Kreatif_Media",
        "Sales_Pelayanan",
    }
)


@dataclass(frozen=True)
class SkillVocab:
    vocab_version: str
    dim: int
    canonical_skills: list[str]
    skill_to_index: dict[str, int]
    alias_to_canonical: dict[str, str]
    canonical_set: frozenset[str]


@dataclass(frozen=True)
class RoleEntry:
    role_id: str
    sector: str
    role_name: str
    role_name_id: str | None
    experience_band: str | None
    required_skills: list[str]
    optional_skills: list[str]


@dataclass(frozen=True)
class RolesCatalog:
    catalog_version: str
    default_match_threshold: float
    roles: list[RoleEntry]
    role_by_id: dict[str, RoleEntry]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File data karier tidak ditemukan: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_skill_vocab(path: Path = SKILL_VOCAB_PATH) -> SkillVocab:
    data = _read_json(path)
    skills_raw = data["skills"]
    dim = int(data["dim"])

    canonical_skills: list[str] = []
    per_skill_aliases: dict[str, list[str]] = {}

    for entry in sorted(skills_raw, key=lambda x: int(x["index"])):
        index = int(entry["index"])
        canonical = str(entry["canonical"])
        if index != len(canonical_skills):
            raise ValueError(
                f"Index skill tidak berurutan di {path}: expected {len(canonical_skills)}, got {index}"
            )
        canonical_skills.append(canonical)
        aliases = entry.get("aliases") or []
        if aliases:
            per_skill_aliases[canonical] = list(aliases)

    if len(canonical_skills) != dim:
        raise ValueError(f"dim={dim} tidak sama dengan jumlah skill ({len(canonical_skills)})")

    if len(set(canonical_skills)) != dim:
        raise ValueError("Duplikat canonical skill di vocab")

    global_aliases = data.get("global_aliases") or {}
    alias_to_canonical = build_alias_lookup(
        canonical_skills,
        per_skill_aliases=per_skill_aliases,
        global_aliases=global_aliases,
    )
    skill_to_index = {name: i for i, name in enumerate(canonical_skills)}
    canonical_set = frozenset(canonical_skills)

    return SkillVocab(
        vocab_version=str(data.get("vocab_version", "unknown")),
        dim=dim,
        canonical_skills=canonical_skills,
        skill_to_index=skill_to_index,
        alias_to_canonical=alias_to_canonical,
        canonical_set=canonical_set,
    )


def load_roles_catalog(
    path: Path = ROLES_CATALOG_PATH,
    *,
    vocab: SkillVocab | None = None,
) -> RolesCatalog:
    data = _read_json(path)
    if vocab is None:
        vocab = load_skill_vocab()

    roles: list[RoleEntry] = []
    for raw in data["roles"]:
        sector = str(raw["sector"])
        if sector not in PILOT_SECTORS:
            raise ValueError(f"Sektor tidak valid: {sector!r} pada role {raw.get('role_id')}")

        required = list(raw["required_skills"])
        optional = list(raw.get("optional_skills") or [])
        for skill in required + optional:
            if skill not in vocab.canonical_set:
                raise ValueError(
                    f"Skill {skill!r} pada {raw['role_id']} tidak ada di skill_vocab"
                )

        roles.append(
            RoleEntry(
                role_id=str(raw["role_id"]),
                sector=sector,
                role_name=str(raw["role_name"]),
                role_name_id=raw.get("role_name_id"),
                experience_band=raw.get("experience_band"),
                required_skills=required,
                optional_skills=optional,
            )
        )

    role_ids = [r.role_id for r in roles]
    if len(role_ids) != len(set(role_ids)):
        raise ValueError("Duplikat role_id di roles_catalog")

    role_by_id = {r.role_id: r for r in roles}
    return RolesCatalog(
        catalog_version=str(data.get("catalog_version", "unknown")),
        default_match_threshold=float(data.get("default_match_threshold", 0.6)),
        roles=roles,
        role_by_id=role_by_id,
    )


def skills_to_multihot(
    canonical_skills: list[str],
    vocab: SkillVocab,
) -> np.ndarray:
    """Vektor multi-hot shape (dim,) float32."""
    vec = np.zeros(vocab.dim, dtype=np.float32)
    for skill in canonical_skills:
        idx = vocab.skill_to_index.get(skill)
        if idx is not None:
            vec[idx] = 1.0
    return vec


def role_requirements_multihot(role: RoleEntry, vocab: SkillVocab) -> np.ndarray:
    """Multi-hot dari required_skills saja (untuk training label overlap)."""
    return skills_to_multihot(role.required_skills, vocab)


def inspect_career_data() -> None:
    vocab = load_skill_vocab()
    catalog = load_roles_catalog(vocab=vocab)

    print("==== MAPAN Career Data (Step 9.2) =====\n")
    print(f"skill_vocab: v{vocab.vocab_version} | dim={vocab.dim}")
    print(f"roles_catalog: v{catalog.catalog_version} | roles={len(catalog.roles)}")
    print(f"default_match_threshold: {catalog.default_match_threshold}\n")

    by_sector: dict[str, int] = {}
    for role in catalog.roles:
        by_sector[role.sector] = by_sector.get(role.sector, 0) + 1
    print("Role per sektor:")
    for sector in sorted(by_sector):
        print(f"  {sector}: {by_sector[sector]}")
    print()

    from app.services.skill_normalization import normalize_skill

    tests = ["Python,", "RESTful API", "excel", "k8s", "UNKNOWN_SKILL_XYZ"]
    print("Uji normalisasi (sample):")
    for raw in tests:
        c = normalize_skill(
            raw,
            alias_to_canonical=vocab.alias_to_canonical,
            canonical_set=set(vocab.canonical_set),
        )
        print(f"  {raw!r} -> {c!r}")


if __name__ == "__main__":
    inspect_career_data()
