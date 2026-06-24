import joblib
import numpy as np  # Ditambahkan untuk membantu pemrosesan array probabilitas

class MusicClassifier:
    def __init__(self, model_path, encoder_path):
        self.model = joblib.load(model_path)
        self.le = joblib.load(encoder_path)

    def predict(self, df_features):
        try:
            # 1. Prediksi Label Numerik & Konversi ke Nama Provinsi
            prediction_numeric = self.model.predict(df_features)
            region_name = self.le.inverse_transform(prediction_numeric)[0]

            # 2. Hitung Confidence Score menggunakan predict_proba
            # Karena input df_features adalah 1 baris, hasil predict_proba berbentuk [[prob_1, prob_2, ...]]
            probabilities = self.model.predict_proba(df_features)
            confidence_score = float(probabilities[0].max())  # Mengambil nilai probabilitas tertinggi

            return region_name, confidence_score

        except Exception as e:
            print(f"Error prediksi: {e}")
            return None, 0.0