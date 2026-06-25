"""
trainer.py
Trains the TF-IDF + Logistic Regression sentiment classification model
and saves the trained artifacts to disk using joblib.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from backend.preprocessor import clean_batch
from backend.dataset_generator import ensure_dataset


class ModelTrainer:
    """Encapsulates training of the TF-IDF + Logistic Regression pipeline."""

    def __init__(self, data_path: str, model_path: str, vectorizer_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.metrics_ = {}

    def train(self) -> dict:
        """
        Trains the model end-to-end:
            1. Loads (or generates) the dataset.
            2. Cleans the review text.
            3. Vectorizes text using TF-IDF.
            4. Trains a Logistic Regression classifier.
            5. Evaluates on a held-out test split.
            6. Saves model + vectorizer to disk via joblib.

        Returns:
            Dictionary of evaluation metrics.
        """
        df = ensure_dataset(self.data_path)
        df = df.dropna(subset=["review", "sentiment"]).reset_index(drop=True)

        cleaned_reviews = clean_batch(df["review"].tolist())
        labels = df["sentiment"].tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_reviews,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label="Positive", zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label="Positive", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="Positive", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Negative"])

        self.metrics_ = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "labels": ["Positive", "Negative"],
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)
        joblib.dump(self.metrics_, self._metrics_path())

        return self.metrics_

    def _metrics_path(self) -> str:
        model_dir = os.path.dirname(self.model_path)
        return os.path.join(model_dir, "metrics.pkl")

    def load_metrics(self) -> dict:
        """Loads previously saved evaluation metrics, if available."""
        metrics_path = self._metrics_path()
        if os.path.exists(metrics_path):
            return joblib.load(metrics_path)
        return {}


def models_exist(model_path: str, vectorizer_path: str) -> bool:
    """Checks whether both the model and vectorizer files exist on disk."""
    return os.path.exists(model_path) and os.path.exists(vectorizer_path)


def train_if_needed(data_path: str, model_path: str, vectorizer_path: str) -> dict:
    """
    Trains the model only if model files are missing.
    Returns metrics dict (freshly trained or loaded from disk).
    """
    trainer = ModelTrainer(data_path, model_path, vectorizer_path)
    if not models_exist(model_path, vectorizer_path):
        return trainer.train()
    existing_metrics = trainer.load_metrics()
    if existing_metrics:
        return existing_metrics
    return trainer.train()
