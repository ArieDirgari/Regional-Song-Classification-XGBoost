"""
processor.py
Pipeline ekstraksi fitur audio untuk halaman deteksi.
Dibangun ulang 1:1 dari notebook Feature_Extraction_2.ipynb.
"""

import io
import os
import tempfile

import numpy as np
import pandas as pd
import librosa
import streamlit as st
from pyAudioAnalysis import ShortTermFeatures


# =============================================================================
# IRSD COLUMN ORDER
# =============================================================================
IRSD_COLUMNS = [
    "ZCR_mean", "Energy_mean", "Energy_Entropy_mean",
    "Spectral_Centroid_mean", "Spectral_Spread_mean",
    "Spectral_Entropy_mean", "Spectral_Flux_mean",
    "Spectral_Rolloff_mean",
    "Mfcc1_mean", "Mfcc2_mean", "Mfcc3_mean", "Mfcc4_mean", "Mfcc5_mean",
    "Mfcc6_mean", "Mfcc7_mean", "Mfcc8_mean", "Mfcc9_mean", "Mfcc10_mean",
    "Mfcc11_mean", "Mfcc12_mean", "Mfcc13_mean",
    "Chroma1_mean", "Chroma2_mean", "Chroma3_mean", "Chroma4_mean",
    "Chroma5_mean", "Chroma6_mean", "Chroma7_mean", "Chroma8_mean",
    "Chroma9_mean", "Chroma10_mean", "Chroma11_mean", "Chroma12_mean",
    "Chroma_Deviation_mean",
    "ZCR_std", "Energy_std", "Energy_Entropy_std",
    "Spectral_Centroid_std", "Spectral_Spread_std",
    "Spectral_Entropy_std", "Spectral_Flux_std",
    "Spectral_Rolloff_std",
    "Mfcc1_std", "Mfcc2_std", "Mfcc3_std", "Mfcc4_std", "Mfcc5_std",
    "Mfcc6_std", "Mfcc7_std", "Mfcc8_std", "Mfcc9_std", "Mfcc10_std",
    "Mfcc11_std", "Mfcc12_std", "Mfcc13_std",
    "Chroma1_std", "Chroma2_std", "Chroma3_std", "Chroma4_std",
    "Chroma5_std", "Chroma6_std", "Chroma7_std", "Chroma8_std",
    "Chroma9_std", "Chroma10_std", "Chroma11_std", "Chroma12_std",
]


# =============================================================================
# STEP 1 — LOAD AUDIO
# Notebook: librosa.load(file_path, sr=22050, mono=True)
# MP3 harus ditulis ke temp file karena ffmpeg butuh seekable file.
# =============================================================================
def _load_audio(uploaded_file) -> tuple[np.ndarray, int] | tuple[None, None]:
    try:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

        if ext == "mp3":
            # ffmpeg tidak bisa baca dari pipe/BytesIO — tulis ke disk dulu
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                y, sr = librosa.load(tmp_path, sr=22050, mono=True)
            finally:
                os.unlink(tmp_path)
        else:
            # WAV aman langsung dari BytesIO
            y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050, mono=True)

        return y, sr

    except Exception as e:
        st.error(f"Gagal memuat audio: {e}")
        return None, None


# =============================================================================
# STEP 2 — PREPROCESSING
# Notebook:
#   y_trimmed, _ = librosa.effects.trim(y)
#   y_norm = y_trimmed / np.max(np.abs(y_trimmed))
# =============================================================================
def _preprocess(y: np.ndarray) -> np.ndarray:
    y_trimmed, _ = librosa.effects.trim(y)
    y_norm = y_trimmed / np.max(np.abs(y_trimmed))
    return y_norm


# =============================================================================
# STEP 3 — SEGMENTASI
# Notebook (blok yang dipakai untuk feature extraction):
#   num_segments = 20, duration = 10
#   segment_length = len(y_norm) // num_segments
#   max_samples = sr * duration
#   seg = y_norm[start:end][:max_samples]
# =============================================================================
def _segment(y_norm: np.ndarray, sr: int,
             num_segments: int = 20,
             duration: int = 10) -> list[np.ndarray]:
    segment_length = len(y_norm) // num_segments
    max_samples = sr * duration
    segments = []

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        seg = y_norm[start:end]
        seg = seg[:max_samples]
        if len(seg) > 0:
            segments.append(seg)

    return segments


# =============================================================================
# STEP 4 — FEATURE EXTRACTION PER SEGMEN
# Notebook:
#   win = int(0.050 * sr)
#   step = int(0.025 * sr)
#   features, feature_names = ShortTermFeatures.feature_extraction(seg, sr, win, step)
#   mean = np.mean(features, axis=1)
#   std  = np.std(features,  axis=1)
# =============================================================================
def _extract_features(segments: list[np.ndarray],
                      sr: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    win  = int(0.050 * sr)
    step = int(0.025 * sr)

    all_mean = []
    all_std  = []
    feature_names = []

    for seg in segments:
        features, names = ShortTermFeatures.feature_extraction(seg, sr, win, step)
        all_mean.append(np.mean(features, axis=1))
        all_std.append(np.std(features,  axis=1))
        feature_names = names   # sama untuk setiap segmen

    # Aggregasi: rata-rata mean dan std dari semua segmen
    mean_final = np.mean(all_mean, axis=0)
    std_final  = np.mean(all_std,  axis=0)

    return mean_final, std_final, feature_names


# =============================================================================
# STEP 5 — MAPPING KE IRSD COLUMNS
# Notebook: mapping nama fitur pyAudioAnalysis → nama kolom IRSD
# =============================================================================
def _map_to_irsd(mean_final: np.ndarray,
                 std_final: np.ndarray,
                 feature_names: list[str]) -> pd.DataFrame:
    feature_dict = {}

    for i, name in enumerate(feature_names):
        base = name.lower()

        if base == "zcr":
            key = "ZCR"
        elif base == "energy":
            key = "Energy"
        elif base == "energy_entropy":
            key = "Energy_Entropy"
        elif base == "spectral_centroid":
            key = "Spectral_Centroid"
        elif base == "spectral_spread":
            key = "Spectral_Spread"
        elif base == "spectral_entropy":
            key = "Spectral_Entropy"
        elif base == "spectral_flux":
            key = "Spectral_Flux"
        elif base == "spectral_rolloff":
            key = "Spectral_Rolloff"
        elif "mfcc" in base:
            idx = base.split("_")[1]
            key = f"Mfcc{idx}"
        elif "chroma" in base and "std" not in base:
            idx = base.split("_")[1]
            key = f"Chroma{idx}"
        elif base == "chroma_std":
            key = "Chroma_Deviation"
        else:
            continue

        feature_dict[f"{key}_mean"] = mean_final[i]
        feature_dict[f"{key}_std"]  = std_final[i]

    final_features = {col: feature_dict.get(col, 0) for col in IRSD_COLUMNS}
    return pd.DataFrame([final_features])


# =============================================================================
# PUBLIC API — dipanggil dari halaman_deteksi
# =============================================================================
def process_audio(uploaded_file) -> pd.DataFrame | None:
    """
    Jalankan seluruh pipeline ekstraksi fitur dari file audio yang diupload.
    Mengembalikan DataFrame 1 baris x 67 kolom (IRSD_COLUMNS),
    atau None jika terjadi error.
    """
    # 1. Load
    y, sr = _load_audio(uploaded_file)
    if y is None:
        return None

    # 2. Preprocessing: trim silence → normalize
    y_norm = _preprocess(y)

    # 3. Segmentasi
    segments = _segment(y_norm, sr, num_segments=20, duration=10)
    if not segments:
        st.error("Segmentasi gagal: tidak ada segmen yang dihasilkan.")
        return None

    # 4. Feature extraction + aggregasi
    mean_final, std_final, feature_names = _extract_features(segments, sr)

    # 5. Mapping ke IRSD columns
    df_features = _map_to_irsd(mean_final, std_final, feature_names)

    return df_features.astype(float)