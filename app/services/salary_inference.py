from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Sama dengan train_salary.py
FEATURE_COLUMNS = ["Judul Pekerjaan", "Perusahaan", "Lokasi"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "exported_artifacts"

MODEL_PATH = ARTIFACT_DIR / "salary_estimator.keras"
OHE_PATH = ARTIFACT_DIR / "salary_ohe.pkl"
SCALER_PATH = ARTIFACT_DIR / "salary_scaler.pkl"


def load_salary_artifacts() -> tuple[tf.keras.Model, OneHotEncoder, MinMaxScaler]:
    """Muat model + encoder + scaler sekali (untuk dipakai ulang di FastAPI)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

    # compile=False → cukup untuk predict, hindari masalah custom Huber loss
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    ohe: OneHotEncoder = joblib.load(OHE_PATH)
    scaler: MinMaxScaler = joblib.load(SCALER_PATH)
    return model, ohe, scaler


def predict_salary_idr(
    job_title: str,
    company: str,
    location: str,
    model: tf.keras.Model,
    ohe: OneHotEncoder,
    scaler: MinMaxScaler,
) -> float:
    """Prediksi gaji dalam Rupiah (integer)."""
    row = pd.DataFrame(
        [{
            "Judul Pekerjaan": job_title,
            "Perusahaan": company,
            "Lokasi": location,
        }]
    )
    X_sparse = ohe.transform(row[FEATURE_COLUMNS])
    y_scaled = model.predict(X_sparse, verbose=0).ravel()[0]
    y_idr = scaler.inverse_transform([[y_scaled]])[0, 0]
    return float(round(y_idr))

class SalaryPredictor:
    """Singleton inference: load artefak sekali, prediksi berkali-kali."""

    def __init__(self) -> None:
        self._model: tf.keras.Model | None = None
        self._ohe: OneHotEncoder | None = None
        self._scaler: MinMaxScaler | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        model, ohe, scaler = load_salary_artifacts()
        self._model = model
        self._ohe = ohe
        self._scaler = scaler

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(
        self,
        job_title: str,
        company: str,
        location: str,
    ) -> float:
        if not self.is_loaded:
            raise RuntimeError("SalaryPredictor belum di-load. Panggil .load() dulu.")
        return predict_salary_idr(
            job_title=job_title,
            company=company,
            location=location,
            model=self._model,
            ohe=self._ohe,
            scaler=self._scaler,
        )


# Instance global untuk dipakai router (load di startup FastAPI)
salary_predictor = SalaryPredictor()


if __name__ == "__main__":
    print("==== MAPAN Salary Inference — smoke test =====\n")
    salary_predictor.load()
    print(f"Loaded: {salary_predictor.is_loaded}")
    print(f"OHE features: {len(salary_predictor._ohe.get_feature_names_out())}\n")
    sample = salary_predictor.predict(
        job_title="Software Engineer",
        company="PT Contoh Teknologi Indonesia",
        location="Jakarta Selatan",
    )
    print(f"Prediksi gaji contoh: Rp{sample:,.0f}")
