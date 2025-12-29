# Machine learning classifier logic will go here
from typing import Dict


class SemanticClassifier:
    """
    Semantic classifier for detecting implicit prompt injection attempts.
    This class wraps a trained ML model and exposes a prediction interface.
    """

    def __init__(self, model=None, vectorizer=None):
        """
        Initialize the classifier.
        Model and vectorizer are expected to be pre-trained.
        """
        self.model = model
        self.vectorizer = vectorizer

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
