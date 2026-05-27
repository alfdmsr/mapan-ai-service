"""
MAPAN — standardisasi string skill (Model 2 NER → Model 3 multi-hot).

Modul terpisah agar training, inference, dan generator dataset memakai aturan identik.
Alias map diisi dari skill_vocab.json pada Step 9.2.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Mapping

_TRAILING_PUNCT = re.compile(
    rf"^[\s{re.escape(string.punctuation)}]+|[\s{re.escape(string.punctuation)}]+$"
)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_lookup_key(text: str) -> str:
    """Kunci dictionary: lowercase, trim, hapus tanda baca tepi, collapse spasi."""
    if not text or not str(text).strip():
        return ""

    s = unicodedata.normalize("NFKC", str(text).strip())
    s = s.lower()
    s = _TRAILING_PUNCT.sub("", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s


def normalize_skill(
    raw_text: str,
    *,
    alias_to_canonical: Mapping[str, str],
    canonical_set: set[str] | None = None,
) -> str | None:
    """
    Map satu string skill mentah → canonical vocab, atau None jika tidak dikenal.

    Args:
        raw_text: Output NER / teks mentah (mis. "Python,", "RESTful API").
        alias_to_canonical: Kunci sudah dinormalisasi → nilai canonical display.
        canonical_set: Set canonical valid; dipakai validasi & fallback exact match.
    """
    key = normalize_lookup_key(raw_text)
    if not key:
        return None

    canonical = alias_to_canonical.get(key)
    if canonical is None and canonical_set is not None:
        for candidate in canonical_set:
            if normalize_lookup_key(candidate) == key:
                canonical = candidate
                break

    if canonical is None:
        return None
    if canonical_set is not None and canonical not in canonical_set:
        return None
    return canonical


def normalize_skill_list(
    raw_skills: list[str],
    *,
    alias_to_canonical: Mapping[str, str],
    canonical_set: set[str] | None = None,
) -> list[str]:
    """Normalisasi daftar skill; hapus duplikat canonical, pertahankan urutan masuk."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in raw_skills:
        canonical = normalize_skill(
            raw,
            alias_to_canonical=alias_to_canonical,
            canonical_set=canonical_set,
        )
        if canonical is not None and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def build_alias_lookup(
    canonical_skills: list[str],
    per_skill_aliases: Mapping[str, list[str]] | None = None,
    global_aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Bangun map lookup_key → canonical untuk dipakai normalize_skill().

    Dipanggil sekali saat load skill_vocab.json (Step 9.2).
    """
    lookup: dict[str, str] = {}
    canonical_set = set(canonical_skills)

    def _register(alias: str, canonical: str) -> None:
        key = normalize_lookup_key(alias)
        if not key:
            return
        if canonical not in canonical_set:
            raise ValueError(f"Canonical tidak ada di vocab: {canonical!r}")
        lookup[key] = canonical

    for canonical in canonical_skills:
        _register(canonical, canonical)

    if per_skill_aliases:
        for canonical, aliases in per_skill_aliases.items():
            for alias in aliases:
                _register(alias, canonical)

    if global_aliases:
        for alias, canonical in global_aliases.items():
            _register(alias, canonical)

    return lookup
