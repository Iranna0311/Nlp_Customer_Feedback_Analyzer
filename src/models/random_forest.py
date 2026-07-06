"""Random forest wrapper for sentiment classification."""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = None):
        """
        Wrapper class for Scikit-Learn's Random Forest Classifier.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    def train(self, X_train, y_train):
        print("Training Random Forest Model...")
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Random Forest model saved to: {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        self.model = joblib.load(filepath)
        print(f"Random Forest model loaded from: {filepath}")