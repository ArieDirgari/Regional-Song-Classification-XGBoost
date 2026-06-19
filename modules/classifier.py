import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use('Agg')
from pyAudioAnalysis import ShortTermFeatures

import joblib
import os

class MusicClassifier:
    def __init__(self, model_path, encoder_path):

        self.model = joblib.load(model_path)
        self.le = joblib.load(encoder_path)

    def predict(self, df_features):
        try:
            prediction_numeric = self.model.predict(df_features)

            region_name = self.le.inverse_transform(
                prediction_numeric
            )[0]

            return region_name

        except Exception as e:
            print(f"Error prediksi: {e}")
            return None
