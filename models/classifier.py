# Machine learning classifier logic
import os
import joblib
from typing import Dict


class SemanticClassifier:
    """
    Semantic classifier for detecting implicit prompt injection attempts.
    This class loads a trained ML model and exposes a prediction interface.
    """

    def __init__(
        self,
        model_path: str = "models/model.pkl",
        vectorizer_path: str = "models/vectorizer.pkl"
    ):
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.model = None
            self.vectorizer = None

    def is_ready(self) -> bool:
        """
        Check whether the classifier has a loaded model and vectorizer.
        """
        return self.model is not None and self.vectorizer is not None

    def predict_proba(self, text: str) -> Dict:
        """
        Predict the probability of prompt injection.
        Returns a dictionary with probability and readiness state.
        """
        if not self.is_ready():
            return {
                "ready": False,
                "probability": 0.0
            }

        features = self.vectorizer.transform([text])
        probability = self.model.predict_proba(features)[0][1]

        return {
            "ready": True,
            "probability": float(probability)
        }
