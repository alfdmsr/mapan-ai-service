"""
MAPAN Model 3 — Career Path Recommendation (Dual-Tower Matcher).

Step 9.1: Arsitektur Functional API + EarlyStoppingAt85Accuracy.
Step 9.4: Training dari match_pairs.jsonl + export .keras.

Usage:
  python train_recommendation.py --preview
  python train_recommendation.py
  python train_recommendation.py --epochs 80 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from app.services.career_data import load_roles_catalog, load_skill_vocab
from generate_match_pairs import (
    MATCH_PAIRS_PATH,
    load_match_pairs_jsonl,
    pairs_to_arrays,
    train_val_split_by_cv,
)

# Default = dim skill_vocab.json (Step 9.2).
try:
    DEFAULT_SKILL_VOCAB_DIM = load_skill_vocab().dim
except FileNotFoundError:
    DEFAULT_SKILL_VOCAB_DIM = 72

DEFAULT_TOWER_UNITS: tuple[int, ...] = (128, 64)
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_DROPOUT = 0.25
TARGET_VAL_ACCURACY = 0.85

ARTIFACT_DIR = Path(__file__).resolve().parent / "exported_artifacts"
MODEL_PATH = ARTIFACT_DIR / "career_dual_tower.keras"
META_PATH = ARTIFACT_DIR / "career_training_meta.pkl"


def _build_skill_tower(
    inputs: tf.Tensor,
    *,
    name_prefix: str,
    hidden_units: Sequence[int],
    embedding_dim: int,
    dropout_rate: float,
) -> tf.Tensor:
    """Satu tower: multi-hot (N,) → Dense → … → embedding (D,)."""
    x = inputs
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            units,
            activation="relu",
            name=f"{name_prefix}_dense_{i}",
        )(x)
        x = tf.keras.layers.Dropout(
            dropout_rate,
            name=f"{name_prefix}_dropout_{i}",
        )(x)
    return tf.keras.layers.Dense(
        embedding_dim,
        activation="relu",
        name=f"{name_prefix}_embedding",
    )(x)


def build_dual_tower_model(
    skill_vocab_dim: int = DEFAULT_SKILL_VOCAB_DIM,
    *,
    tower_hidden_units: Sequence[int] = DEFAULT_TOWER_UNITS,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    dropout_rate: float = DEFAULT_DROPOUT,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Dual-Tower biner: (user_skills, role_requirements) → P(cocok)."""
    user_input = tf.keras.Input(
        shape=(skill_vocab_dim,),
        dtype="float32",
        name="user_skills",
    )
    role_input = tf.keras.Input(
        shape=(skill_vocab_dim,),
        dtype="float32",
        name="role_requirements",
    )

    user_emb = _build_skill_tower(
        user_input,
        name_prefix="user_tower",
        hidden_units=tower_hidden_units,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
    )
    role_emb = _build_skill_tower(
        role_input,
        name_prefix="role_tower",
        hidden_units=tower_hidden_units,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
    )

    merged = tf.keras.layers.Concatenate(name="tower_concat")([user_emb, role_emb])

    x = tf.keras.layers.Dense(32, activation="relu", name="match_dense")(merged)
    match_prob = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="match_probability",
    )(x)

    model = tf.keras.Model(
        inputs=[user_input, role_input],
        outputs=match_prob,
        name="career_dual_tower_matcher",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class EarlyStoppingAt85Accuracy(tf.keras.callbacks.Callback):
    """Hentikan training saat val_accuracy >= target (.cursorrules)."""

    def __init__(
        self,
        monitor: str = "val_accuracy",
        target_accuracy: float = TARGET_VAL_ACCURACY,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.target_accuracy = target_accuracy
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        if value >= self.target_accuracy:
            if self.verbose:
                print(
                    f"\n[MAPAN Career] {self.monitor}={value:.4f} "
                    f">= target {self.target_accuracy:.2f}. Menghentikan training."
                )
            self.model.stop_training = True

    def get_config(self) -> dict[str, object]:
        return {
            "monitor": self.monitor,
            "target_accuracy": self.target_accuracy,
            "verbose": self.verbose,
        }


def compute_binary_class_weights(y: np.ndarray) -> dict[int, float]:
    """Bobot seimbang untuk kelas 0/1 (atasi imbalance 97:3)."""
    labels = y.ravel().astype(int)
    classes = np.array([0, 1])
    if len(np.unique(labels)) < 2:
        return {0: 1.0, 1: 1.0}
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def load_training_data(
    pairs_path: Path,
    *,
    split_mode: str = "stratified",
    val_fraction: float = 0.2,
    random_state: int = 42,
):
    """
    Muat match_pairs.jsonl → tensor train/val.

    split_mode:
      - stratified: train_test_split per label (lebih adil untuk kelas langka)
      - by_cv: split per cv_id (tidak bocor CV, val mungkin tanpa positif)
    """
    vocab = load_skill_vocab()
    catalog = load_roles_catalog(vocab=vocab)
    pairs = load_match_pairs_jsonl(pairs_path)

    if not pairs:
        raise ValueError(f"Tidak ada pasangan di {pairs_path}. Jalankan generate_match_pairs.py dulu.")

    labels = [p["label"] for p in pairs]
    pos_count = sum(labels)

    if split_mode == "stratified" and pos_count >= 2 and pos_count < len(pairs) - 1:
        train_pairs, val_pairs = train_test_split(
            pairs,
            test_size=val_fraction,
            random_state=random_state,
            stratify=labels,
        )
    else:
        train_pairs, val_pairs = train_val_split_by_cv(
            pairs, val_fraction=val_fraction
        )

    x_user_tr, x_role_tr, y_tr = pairs_to_arrays(train_pairs, vocab, catalog)
    x_user_va, x_role_va, y_va = pairs_to_arrays(val_pairs, vocab, catalog)

    return {
        "vocab": vocab,
        "catalog": catalog,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "x_user_train": x_user_tr,
        "x_role_train": x_role_tr,
        "y_train": y_tr,
        "x_user_val": x_user_va,
        "x_role_val": x_role_va,
        "y_val": y_va,
        "split_mode": split_mode,
    }


def train_career_model(
    model: tf.keras.Model,
    data: dict,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    class_weight: dict[int, float] | None = None,
) -> tf.keras.callbacks.History:
    if class_weight is None:
        class_weight = compute_binary_class_weights(data["y_train"])

    print("==== Training Career Dual-Tower (Step 9.4) =====\n")
    print(f"Train: {data['y_train'].shape[0]} | Val: {data['y_val'].shape[0]}")
    print(
        f"  Positif train: {int(data['y_train'].sum())} | "
        f"Positif val: {int(data['y_val'].sum())}"
    )
    print(f"  class_weight: {class_weight}")
    print(f"  split_mode: {data['split_mode']}\n")

    callbacks: list[tf.keras.callbacks.Callback] = [
        EarlyStoppingAt85Accuracy(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        [data["x_user_train"], data["x_role_train"]],
        data["y_train"],
        validation_data=([data["x_user_val"], data["x_role_val"]], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_career_model(model: tf.keras.Model, data: dict) -> dict[str, float]:
    print("\n==== Evaluasi Model 3 =====\n")
    results = model.evaluate(
        [data["x_user_val"], data["x_role_val"]],
        data["y_val"],
        verbose=0,
    )
    metric_names = model.metrics_names
    metrics = {name: float(val) for name, val in zip(metric_names, results)}

    y_pred = (model.predict(
        [data["x_user_val"], data["x_role_val"]], verbose=0
    ) >= 0.5).astype(int)
    y_true = data["y_val"].astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    print(f"  val_loss:      {metrics.get('loss', 0):.4f}")
    print(f"  val_accuracy:  {metrics.get('accuracy', 0):.4f}  (target >= {TARGET_VAL_ACCURACY})")
    print(f"  val_precision: {metrics.get('precision', 0):.4f}")
    print(f"  val_recall:    {metrics.get('recall', 0):.4f}")
    print(f"  TP={tp} FP={fp} FN={fn}")

    if metrics.get("accuracy", 0) >= TARGET_VAL_ACCURACY:
        print(f"\n✓ val_accuracy memenuhi target >= {TARGET_VAL_ACCURACY:.0%} (.cursorrules).")
    else:
        print(f"\n✗ val_accuracy belum >= {TARGET_VAL_ACCURACY:.0%} — perlu lebih banyak data/tuning.")

    if tp == 0 and int(y_true.sum()) > 0:
        print("  Catatan: model tidak memprediksi satupun positif di val (cek imbalance).")

    return metrics


def save_career_artifacts(
    model: tf.keras.Model,
    data: dict,
    metrics: dict[str, float],
    *,
    class_weight: dict[int, float],
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    meta = {
        "vocab_version": data["vocab"].vocab_version,
        "vocab_dim": data["vocab"].dim,
        "catalog_version": data["catalog"].catalog_version,
        "match_threshold": data["catalog"].default_match_threshold,
        "split_mode": data["split_mode"],
        "class_weight": class_weight,
        "val_metrics": metrics,
        "train_size": int(data["y_train"].shape[0]),
        "val_size": int(data["y_val"].shape[0]),
        "train_positives": int(data["y_train"].sum()),
        "val_positives": int(data["y_val"].sum()),
    }
    joblib.dump(meta, META_PATH)

    print(f"\nArtefak disimpan:")
    print(f"  {MODEL_PATH}")
    print(f"  {META_PATH}")


def preview_dual_tower_model(
    skill_vocab_dim: int = DEFAULT_SKILL_VOCAB_DIM,
    batch_size: int = 4,
) -> tf.keras.Model:
    """Cetak summary arsitektur + forward pass dummy."""
    model = build_dual_tower_model(skill_vocab_dim=skill_vocab_dim)

    print("==== MAPAN Model 3 — Dual-Tower (Step 9.1) =====\n")
    model.summary()

    dummy_user = np.zeros((batch_size, skill_vocab_dim), dtype=np.float32)
    dummy_user[:, :3] = 1.0
    dummy_role = np.zeros((batch_size, skill_vocab_dim), dtype=np.float32)
    dummy_role[:, [0, 1, 4]] = 1.0

    scores = model.predict([dummy_user, dummy_role], verbose=0)
    print(f"\nDummy batch ({batch_size} pasangan)")
    print(f"  user_skills shape:         {dummy_user.shape}")
    print(f"  role_requirements shape:   {dummy_role.shape}")
    print(f"  match_probability shape:   {scores.shape}")
    print(f"  contoh skor sigmoid:       {scores.ravel().tolist()}\n")
    return model


def run_training(
    *,
    pairs_path: Path = MATCH_PAIRS_PATH,
    epochs: int = 100,
    batch_size: int = 32,
    split_mode: str = "stratified",
) -> None:
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"{pairs_path} tidak ditemukan. Jalankan: python generate_match_pairs.py"
        )

    data = load_training_data(pairs_path, split_mode=split_mode)
    class_weight = compute_binary_class_weights(data["y_train"])

    model = build_dual_tower_model(skill_vocab_dim=data["vocab"].dim)
    train_career_model(
        model,
        data,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
    )
    metrics = evaluate_career_model(model, data)
    save_career_artifacts(model, data, metrics, class_weight=class_weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPAN Career Dual-Tower training")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Hanya tampilkan arsitektur model (Step 9.1)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--split",
        choices=("stratified", "by_cv"),
        default="stratified",
        help="Cara split train/val (default: stratified untuk kelas langka)",
    )
    parser.add_argument(
        "--pairs-path",
        type=Path,
        default=MATCH_PAIRS_PATH,
    )
    args = parser.parse_args()

    if args.preview:
        preview_dual_tower_model()
    else:
        run_training(
            pairs_path=args.pairs_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            split_mode=args.split,
        )


if __name__ == "__main__":
    main()
