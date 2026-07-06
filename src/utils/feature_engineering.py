import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class FeedbackFeatureEngineer:
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """Transform cleaned text strings into TF-IDF features."""
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit_transform(self, df: pd.DataFrame, text_column: str = 'cleaned_text'):
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in dataframe. Run preprocessing first.")

        return self.vectorizer.fit_transform(df[text_column].astype(str))

    def transform(self, df: pd.DataFrame, text_column: str = 'cleaned_text'):
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in dataframe.")

        return self.vectorizer.transform(df[text_column].astype(str))

    def save_vectorizer(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"Vectorizer successfully saved to: {filepath}")

    def load_vectorizer(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No vectorizer file found at {filepath}")

        self.vectorizer = joblib.load(filepath)
        print(f"Vectorizer successfully loaded from: {filepath}")