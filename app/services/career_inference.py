"""
MAPAN Model 3 — Career recommendation inference (Dual-Tower + role catalog).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "exported_artifacts"

MODEL_PATH = ARTIFACT_DIR / "career_dual_tower.keras"
META_PATH = ARTIFACT_DIR / "career_training_meta.pkl"

MODEL_VERSION_DEFAULT = "career_dual_tower_v1"


def load_career_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model career tidak ditemukan: {MODEL_PATH}. "
            "Jalankan train_recommendation.py terlebih dahulu."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def load_career_meta() -> dict[str, Any]:
    if META_PATH.exists():
        return joblib.load(META_PATH)
    return {}


def compute_overlap_details(
    user_skills: list[str],
    role: RoleEntry,
) -> tuple[float, list[str], list[str]]:
    """Overlap rule-based (sama generator label) + matched/missing lists."""
    user_set = set(user_skills)
    required = role.required_skills
    required_set = set(required)
    if not required_set:
        return 0.0, [], []

    matched = sorted(user_set & required_set, key=required.index)
    missing = [s for s in required if s not in user_set]
    ratio = len(matched) / len(required_set)
    return ratio, matched, missing


class CareerRecommender:
    """Singleton: load dual-tower + vocab + catalog, rank roles per user profile."""

    def __init__(self) -> None:
        self._model: tf.keras.Model | None = None
        self._vocab: SkillVocab | None = None
        self._catalog: RolesCatalog | None = None
        self._meta: dict[str, Any] = {}
        self._role_matrix: np.ndarray | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model = load_career_model()
        self._vocab = load_skill_vocab()
        self._catalog = load_roles_catalog(vocab=self._vocab)
        self._meta = load_career_meta()
        self._build_role_matrix()

    def _build_role_matrix(self) -> None:
        if self._catalog is None or self._vocab is None:
            return
        matrices = [
            role_requirements_multihot(role, self._vocab)
            for role in self._catalog.roles
        ]
        self._role_matrix = np.stack(matrices, axis=0).astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_version(self) -> str:
        vocab_v = self._meta.get("vocab_version", "?")
        catalog_v = self._meta.get("catalog_version", "?")
        return f"{MODEL_VERSION_DEFAULT}_v{vocab_v}_c{catalog_v}"

    def canonicalize_skills(self, raw_skills: list[str]) -> list[str]:
        if self._vocab is None:
            raise RuntimeError("CareerRecommender belum di-load.")
        return normalize_skill_list(
            raw_skills,
            alias_to_canonical=self._vocab.alias_to_canonical,
            canonical_set=set(self._vocab.canonical_set),
        )

    def resolve_user_skills(
        self,
        *,
        raw_text: str | None = None,
        skills: list[str] | None = None,
    ) -> list[str]:
        """
        Dapatkan skill canonical dari teks CV (via NER) atau daftar skill langsung.
        """
        if skills:
            return self.canonicalize_skills(skills)

        if raw_text:
            from app.services.cv_inference import cv_parser

            if not cv_parser.is_loaded:
                raise RuntimeError(
                    "CVParser belum dimuat. Server harus load cv_parser di lifespan."
                )
            parsed = cv_parser.parse(raw_text)
            return self.canonicalize_skills(parsed.get("skills", []))

        raise ValueError("Berikan raw_text atau skills.")

    def recommend(
        self,
        *,
        raw_text: str | None = None,
        skills: list[str] | None = None,
        top_k: int = 5,
        sector_filter: str | None = None,
        min_match_score: float = 0.0,
    ) -> dict[str, Any]:
        if not self.is_loaded or self._model is None:
            raise RuntimeError("CareerRecommender belum di-load.")
        if self._vocab is None or self._catalog is None or self._role_matrix is None:
            raise RuntimeError("Artefak CareerRecommender belum lengkap.")

        user_canonical = self.resolve_user_skills(raw_text=raw_text, skills=skills)
        if not user_canonical:
            return {
                "user_skills_canonical": [],
                "recommendations": [],
                "model_version": self.model_version,
                "message": "Tidak ada skill yang ter-map ke vocabulary.",
            }

        user_vec = skills_to_multihot(user_canonical, self._vocab)
        roles = self._catalog.roles
        if sector_filter:
            roles = [r for r in roles if r.sector == sector_filter]
            if not roles:
                return {
                    "user_skills_canonical": user_canonical,
                    "recommendations": [],
                    "model_version": self.model_version,
                    "message": f"Tidak ada role untuk sektor: {sector_filter}",
                }

        role_indices = [self._catalog.roles.index(r) for r in roles]
        role_batch = self._role_matrix[role_indices]
        n = len(roles)
        user_batch = np.repeat(user_vec[np.newaxis, :], n, axis=0)

        scores = self._model.predict([user_batch, role_batch], verbose=0).ravel()

        recommendations: list[dict[str, Any]] = []
        for role, score in zip(roles, scores):
            score_f = float(score)
            if score_f < min_match_score:
                continue
            overlap, matched, missing = compute_overlap_details(user_canonical, role)
            recommendations.append(
                {
                    "role_id": role.role_id,
                    "role_name": role.role_name,
                    "role_name_id": role.role_name_id,
                    "sector": role.sector,
                    "match_score": round(score_f, 4),
                    "match_percent": round(score_f * 100.0, 1),
                    "overlap_ratio": round(overlap, 4),
                    "matched_skills": matched,
                    "missing_skills": missing,
                }
            )

        recommendations.sort(
            key=lambda x: (x["match_score"], x["overlap_ratio"]),
            reverse=True,
        )
        for rank, item in enumerate(recommendations[:top_k], start=1):
            item["rank"] = rank

        top = recommendations[:top_k]

        return {
            "user_skills_canonical": user_canonical,
            "recommendations": top,
            "model_version": self.model_version,
            "message": None,
        }


career_recommender = CareerRecommender()
