"""
predictor.py
Loads the trained model and vectorizer, and exposes prediction utilities
for both single-review and batch predictions.
"""

import os
import joblib
import pandas as pd

from backend.preprocessor import clean_text, clean_batch, validate_review_text


class PredictorError(Exception):
    """Raised when prediction cannot be performed."""
    pass


class SentimentPredictor:
    """Wraps the trained Logistic Regression model and TF-IDF vectorizer."""

    def __init__(self, model_path: str, vectorizer_path: str):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None

    def load(self):
        """Loads the model and vectorizer from disk."""
        if not os.path.exists(self.model_path):
            raise PredictorError(f"Model file not found at {self.model_path}. Please train the model first.")
        if not os.path.exists(self.vectorizer_path):
            raise PredictorError(f"Vectorizer file not found at {self.vectorizer_path}. Please train the model first.")

        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        except Exception as exc:
            raise PredictorError(f"Failed to load model artifacts: {exc}") from exc

        return self

    def is_ready(self) -> bool:
        """Returns True if model and vectorizer are loaded and ready."""
        return self.model is not None and self.vectorizer is not None

    def predict_single(self, review_text: str) -> dict:
        """
        Predicts sentiment for a single review.

        Args:
            review_text: Raw review string entered by the user.

        Returns:
            dict with keys: review, sentiment, confidence (0-100 float)

        Raises:
            PredictorError if model isn't loaded or input is invalid.
        """
        if not self.is_ready():
            raise PredictorError("Model is not loaded. Cannot make predictions.")

        is_valid, message = validate_review_text(review_text)
        if not is_valid:
            raise PredictorError(message)

        cleaned = clean_text(review_text)
        vec = self.vectorizer.transform([cleaned])

        prediction = self.model.predict(vec)[0]
        probabilities = self.model.predict_proba(vec)[0]
        class_index = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_index]) * 100

        return {
            "review": review_text,
            "sentiment": prediction,
            "confidence": round(confidence, 2),
        }

    def predict_batch(self, reviews: list) -> pd.DataFrame:
        """
        Predicts sentiment for a list of reviews.

        Args:
            reviews: List of raw review strings.

        Returns:
            DataFrame with columns: review, sentiment, confidence
        """
        if not self.is_ready():
            raise PredictorError("Model is not loaded. Cannot make predictions.")

        if reviews is None or len(reviews) == 0:
            raise PredictorError("No reviews found to predict on.")

        # Replace empty/invalid entries with a placeholder so indices stay aligned
        safe_reviews = []
        valid_flags = []
        for r in reviews:
            valid, _ = validate_review_text(r if isinstance(r, str) else "")
            valid_flags.append(valid)
            safe_reviews.append(r if valid else "")

        cleaned = clean_batch(safe_reviews)
        vec = self.vectorizer.transform(cleaned)

        predictions = self.model.predict(vec)
        probabilities = self.model.predict_proba(vec)

        results = []
        for i, review in enumerate(reviews):
            if not valid_flags[i]:
                results.append({"review": review, "sentiment": "Invalid", "confidence": 0.0})
                continue
            pred = predictions[i]
            class_index = list(self.model.classes_).index(pred)
            confidence = float(probabilities[i][class_index]) * 100
            results.append({
                "review": review,
                "sentiment": pred,
                "confidence": round(confidence, 2),
            })

        return pd.DataFrame(results)
