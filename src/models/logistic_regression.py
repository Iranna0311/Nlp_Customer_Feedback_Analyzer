"""Logistic regression wrapper for sentiment classification."""

import os
import joblib
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        """
        Wrapper class for Scikit-Learn's Logistic Regression Classifier.
        """
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    def train(self, X_train, y_train):
        """Trains the model on the engineering features."""
        print("Training Logistic Regression Model...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Predicts class tags (0 or 1)."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predicts class probabilities."""
        return self.model.predict_proba(X)

    def save_model(self, filepath: str):
        """Saves model weights to a file path."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Logistic Regression model saved to: {filepath}")

    def load_model(self, filepath: str):
        """Loads model weights from a file path."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        self.model = joblib.load(filepath)
        print(f"Logistic Regression model loaded from: {filepath}")