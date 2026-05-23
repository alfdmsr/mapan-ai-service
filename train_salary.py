from pathlib import Path

import pandas as pd

# 1. Muat data
DATA_PATH = Path(__file__).resolve().parent / "data" / "Job Salary_Final Dataset.csv"

def load_salary_dataset(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")
    return pd.read_csv(csv_path)

def inspect_salary_dataset(df: pd.DataFrame) -> None:
    print("==== Mapan Salary Estimator - Inspeksi Awal =====\n")
    print(f"Shape: {df.shape[0]:,} baris * {df.shape[1]} kolom\n")
    
    print("==== 5 baris pertama =====\n")
    print(df.head(), "\n")
    
    print("==== Info (tipe & non-null) =====\n")
    df.info()

    print("\n==== Missing Values per kolom =====\n")
    print(df.isnull().sum(), "\n")

    print("==== Duplikat baris penuh =====\n")
    print(df.duplicated().sum(), "\n")

    print("==== Kardinalitas (jumlah nilai unik) =====\n")
    for col in df.columns:
        print(f"{col}: {df[col].nunique():,}")

    print("\n==== Contoh nilai Gaji_Rata2 (5 Pertama) =====\n")
    print(df["Gaji_Rata2"].head().tolist())


# 2. Parse Gaji Rupiah 
def parse_salary_idr(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return cleaned.astype("int64")

def inspect_parsed_salary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["salary_idr"] = parse_salary_idr(df["Gaji_Rata2"])

    print("==== Setelah parse gaji (salary_idr) =====\n")
    print(df[["Gaji_Rata2", "salary_idr"]].head(), "\n")
    print(df["salary_idr"].describe(), "\n")
    print(f"Min: Rp{df['salary_idr'].min():,} | Max: Rp{df['salary_idr'].max():,}\n")

    invalid = df["salary_idr"].isna().sum()
    print(f"Baris invalid (Nan setelah parse): {invalid}")

    return df


# Bersihkan baris duplikasi + Satu Gaji per pekerjaan
GROUP_KEYS = ["Judul Pekerjaan", "Perusahaan", "Lokasi"]

def clean_salary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if "salary_idr" not in df.columns:
        raise ValueError("Kolom salary_idr belum ada. Jalankan inspect_parsed_salary() dulu.")

    before = len(df)
    clean = (
        df.groupby(GROUP_KEYS, as_index=False)["salary_idr"]
        .median()
        .sort_values(GROUP_KEYS)
        .reset_index(drop=True)
    )
    after = len(clean)

    print("==== Setelah pembersihan (clean) =====\n")
    print(f"Baris: {before:,} → {after:,} (dihapus/digabung: {before - after:,})\n")
    print(clean["salary_idr"].describe(), "\n")
    print("==== 5 baris pertama (siap modeling) =====\n")
    print(clean.head())

    return clean


# 4. Pisahkan data 

from sklearn.model_selection import train_test_split

FEATURES_COLUMNS = GROUP_KEYS 
TARGET_COLUMN = "salary_idr"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def split_train_validation(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):

    X = df[FEATURES_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    
    print("==== Train / Validation split =====\n")
    print(f"Total: {len(df):,}")
    print(f"Train: {len(X_train):,} ({len(X_train) / len(df):.1%})")
    print(f"Val:   {len(X_validation):,} ({len(X_validation) / len(df):.1%})\n")
    print("---- Distribusi gaji (train) ----")
    print(y_train.describe(), "\n")
    print("---- Distribusi gaji (val) ----")
    print(y_validation.describe())
    return X_train, X_validation, y_train, y_validation

# 5. One-Hot Encoding 

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_features(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, OneHotEncoder]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True,)

    X_train_sparse = encoder.fit_transform(X_train)
    X_val_sparse = encoder.transform(X_validation)

    print("==== One-Hot encoding (sparse) =====\n")
    print(f"X_train: {X_train.shape} → {X_train_sparse.shape}")
    print(f"X_val:   {X_validation.shape} → {X_val_sparse.shape}")
    print(f"Jumlah kolom one-hot: {encoder.get_feature_names_out().shape[0]:,}")
    density = X_train_sparse.nnz / (X_train_sparse.shape[0] * X_train_sparse.shape[1])
    print(f"Kepadatan non-zero: {density:.4%}\n")

    return X_train_sparse, X_val_sparse, encoder

# 6. MinMaxScaler (hanya y_train)

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_target_salary(y_train: pd.Series, y_validation: pd.Series,) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))

    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_validation_scaled = scaler.transform(y_validation.values.reshape(-1, 1)).ravel()
    
    print("==== MinMaxScaler target (0-1) =====\n")
    print("Train scaled:")
    print(f"  min={y_train_scaled.min():.6f}  max={y_train_scaled.max():.6f}  mean={y_train_scaled.mean():.6f}")
    print("Val scaled:")
    print(f"  min={y_validation_scaled.min():.6f}  max={y_validation_scaled.max():.6f}  mean={y_validation_scaled.mean():.6f}")
    print(f"\nContoh inverse (train[0]): Rp{scaler.inverse_transform([[y_train_scaled[0]]])[0,0]:,.0f}")

    return y_train_scaled, y_validation_scaled, scaler

# 7. A. HUber loss + model functional API
import tensorflow as tf

def huber_loss(delta: float = 0.05):
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = 0.5 * tf.square(error)
        linear = delta * (abs_error - 0.5 * delta)
        return tf.reduce_mean(tf.where(abs_error <= delta, quadratic, linear))
    
    return loss

def build_salary_model(num_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(num_features,), sparse=True, name="sparse_features")

    x = tf.keras.layers.Dense(256, activation="relu",)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="salary_scaled")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="salary_estimator")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=huber_loss(delta=0.05),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model

def preview_salary_model(num_features: int) -> tf.keras.Model:
    model = build_salary_model(num_features)
    print("==== Arsitektur Salary Estimator =====\n")
    model.summary()
    return model


# 7. B. Fit Model
from pathlib import Path
import joblib

ARTIFACT_DIR = Path(__file__).resolve().parent / "exported_artifacts"

def train_salary_model(
    model: tf.keras.Model,
    X_train_sparse,
    y_train_scaled: np.ndarray,
    X_val_sparse,
    y_validation_scaled: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
) -> tf.keras.callbacks.History:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae",
        mode="min",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print("==== Training Salary Estimator =====\n")
    history = model.fit(
        X_train_sparse,
        y_train_scaled,
        validation_data=(X_val_sparse, y_validation_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    print("==== Training selesai =====\n")
    return history

def evaluate_salary_model(
    model: tf.keras.Model,
    X_val_sparse,
    y_val_scaled: np.ndarray,
    salary_scaler: MinMaxScaler,
) -> float:
    y_pred_scaled = model.predict(X_val_sparse, verbose=0).ravel()
    val_mae_scaled = np.mean(np.abs(y_val_scaled - y_pred_scaled))
    print("==== Evaluasi (skala 0–1) =====\n")
    print(f"Validation MAE (scaled): {val_mae_scaled:.6f}")
    print(f"Target .cursorrules:      MAE <= 0.02")

# Opsional: MAE dalam Rupiah (lebih mudah dibaca)
    y_val_idr = salary_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).ravel()
    y_pred_idr = salary_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    val_mae_idr = np.mean(np.abs(y_val_idr - y_pred_idr))
    print(f"Validation MAE (IDR):     Rp{val_mae_idr:,.0f}")
    return val_mae_scaled


def save_salary_artifacts(
    model: tf.keras.Model,
    ohe: OneHotEncoder,
    salary_scaler: MinMaxScaler,
    artifacts_dir: Path = ARTIFACT_DIR,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifacts_dir / "salary_estimator.keras")
    joblib.dump(ohe, artifacts_dir / "salary_ohe.pkl")
    joblib.dump(salary_scaler, artifacts_dir / "salary_scaler.pkl")
    print(f"\nArtefak disimpan di: {artifacts_dir}")


if __name__ == "__main__":
    df = load_salary_dataset()
    inspect_salary_dataset(df)    
    df = inspect_parsed_salary(df)  
    df_clean = clean_salary_dataset(df)

    X_train, X_validation, y_train, y_validation = split_train_validation(df_clean)
    X_train_sparse, X_val_sparse, ohe = encode_categorical_features(
        X_train, X_validation
    )

    y_train_scaled, y_validation_scaled, salary_scaler = scale_target_salary(
        y_train, y_validation
    )

    num_features = X_train_sparse.shape[1]
    model = build_salary_model(num_features)
    history = train_salary_model(
        model,
        X_train_sparse,
        y_train_scaled,
        X_val_sparse,
        y_validation_scaled,
    )
    val_mae = evaluate_salary_model(
        model,
        X_val_sparse,
        y_validation_scaled,
        salary_scaler,
    )
    if val_mae <= 0.02:
        print("\n✓ Kriteria MAE <= 0.02 terpenuhi (skala scaled).")
    else:
        print("\n✗ MAE belum memenuhi target — tuning diperlukan.")
    save_salary_artifacts(model, ohe, salary_scaler)

    
    