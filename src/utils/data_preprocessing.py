"""
data_preprocessing.py
---------------------
Step 2: Data Collection & Preprocessing
Handles loading, cleaning, and preparing raw customer feedback data.

Usage:
    from src.utils.data_preprocessing import DataPreprocessor
    dp = DataPreprocessor()
    df = dp.load_data("data/raw/feedback.csv")
    df_clean = dp.run_pipeline(df)
    dp.save(df_clean, "data/processed/feedback_clean.csv")
"""

import re
import string
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── optional: install with  pip install nltk
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LABEL_MAP = {"positive": 1, "negative": 0, "neutral": 2}

BASIC_STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "what", "which", "who", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "the", "a", "an", "and",
    "but", "or", "for", "nor", "so", "yet", "both", "either", "neither",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "in", "on", "at", "by", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "to", "of", "off", "over", "under"
}


# ─────────────────────────────────────────────
# DataPreprocessor class
# ─────────────────────────────────────────────

class DataPreprocessor:
    """
    End-to-end preprocessing pipeline for customer feedback text.

    Pipeline stages
    ---------------
    1. load_data        – read CSV / JSON / Excel
    2. explore          – quick EDA summary
    3. drop_duplicates  – remove exact duplicate rows
    4. handle_missing   – fill or drop NaN values
    5. clean_text       – lowercase, remove noise
    6. tokenize         – split into word tokens
    7. remove_stopwords – filter common words
    8. stem_or_lemmatize – reduce words to root form
    9. encode_labels    – convert string labels → integers
    10. save            – write cleaned data to disk
    """

    def __init__(self, use_lemmatizer: bool = True, language: str = "english"):
        self.use_lemmatizer = use_lemmatizer
        self.language = language

        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words(language))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        else:
            log.warning("NLTK not found – falling back to basic stopword list.")
            self.stop_words = BASIC_STOPWORDS
            self.stemmer = None
            self.lemmatizer = None

    # ── 1. Load ──────────────────────────────

    def load_data(self, path: str) -> pd.DataFrame:
        """Load a CSV, JSON, or Excel file into a DataFrame."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        loaders = {
            ".csv": pd.read_csv,
            ".json": pd.read_json,
            ".xlsx": pd.read_excel,
            ".xls": pd.read_excel,
        }
        loader = loaders.get(p.suffix.lower())
        if loader is None:
            raise ValueError(f"Unsupported file type: {p.suffix}")

        df = loader(path)
        log.info(f"Loaded {len(df):,} rows from {path}")
        return df

    # ── 2. EDA summary ───────────────────────

    def explore(self, df: pd.DataFrame, text_col: str = "text") -> dict:
        """Print a quick exploratory summary and return stats dict."""
        stats = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
        }
        if text_col in df.columns:
            lengths = df[text_col].dropna().astype(str).str.len()
            stats["text_length"] = {
                "mean": round(lengths.mean(), 1),
                "median": round(lengths.median(), 1),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
            }
        if "label" in df.columns:
            stats["label_distribution"] = df["label"].value_counts().to_dict()

        log.info("=== EDA Summary ===")
        for k, v in stats.items():
            log.info(f"  {k}: {v}")
        return stats

    # ── 3. Drop duplicates ───────────────────

    def drop_duplicates(self, df: pd.DataFrame,
                        subset: list = None) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=subset).reset_index(drop=True)
        log.info(f"Dropped {before - len(df)} duplicate rows.")
        return df

    # ── 4. Handle missing values ─────────────

    def handle_missing(self, df: pd.DataFrame,
                       text_col: str = "text",
                       strategy: str = "drop") -> pd.DataFrame:
        """
        strategy: 'drop' removes rows with missing text,
                  'fill' replaces NaN with empty string.
        """
        before = len(df)
        if strategy == "drop":
            df = df.dropna(subset=[text_col]).reset_index(drop=True)
        elif strategy == "fill":
            df[text_col] = df[text_col].fillna("")
        log.info(f"handle_missing ({strategy}): {before - len(df)} rows removed.")
        return df

    # ── 5. Clean text ────────────────────────

    def clean_text(self, text: str) -> str:
        """
        Applies all cleaning steps to a single string:
        - lowercase
        - remove URLs
        - remove HTML tags
        - remove emojis / non-ASCII
        - remove punctuation & digits
        - collapse whitespace
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)          # URLs
        text = re.sub(r"<[^>]+>", "", text)                  # HTML tags
        text = re.sub(r"[^\x00-\x7F]+", " ", text)           # non-ASCII / emoji
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"\d+", "", text)                       # digits
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_column(self, df: pd.DataFrame,
                     text_col: str = "text") -> pd.DataFrame:
        df = df.copy()
        df[text_col] = df[text_col].apply(self.clean_text)
        log.info("Text cleaning applied.")
        return df

    # ── 6. Tokenize ──────────────────────────

    def tokenize(self, text: str) -> list:
        """Split cleaned text into a list of word tokens."""
        if NLTK_AVAILABLE:
            return word_tokenize(text)
        return text.split()

    def tokenize_column(self, df: pd.DataFrame,
                        text_col: str = "text") -> pd.DataFrame:
        df = df.copy()
        df["tokens"] = df[text_col].apply(self.tokenize)
        log.info("Tokenization done.")
        return df

    # ── 7. Remove stopwords ──────────────────

    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def remove_stopwords_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tokens"] = df["tokens"].apply(self.remove_stopwords)
        log.info("Stopwords removed.")
        return df

    # ── 8. Stem / Lemmatize ──────────────────

    def normalize_token(self, token: str) -> str:
        if NLTK_AVAILABLE:
            if self.use_lemmatizer:
                return self.lemmatizer.lemmatize(token)
            return self.stemmer.stem(token)
        # Simple suffix-stripping fallback
        for suffix in ("ing", "tion", "ly", "ed", "ness"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                return token[: -len(suffix)]
        return token

    def normalize_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tokens"] = df["tokens"].apply(
            lambda toks: [self.normalize_token(t) for t in toks]
        )
        df["clean_text"] = df["tokens"].apply(lambda toks: " ".join(toks))
        log.info("Stemming/Lemmatization done.")
        return df

    # ── 9. Encode labels ─────────────────────

    def encode_labels(self, df: pd.DataFrame,
                      label_col: str = "label") -> pd.DataFrame:
        """Convert string labels → integer codes."""
        df = df.copy()
        if label_col not in df.columns:
            log.warning(f"Column '{label_col}' not found – skipping encoding.")
            return df
        df[label_col] = df[label_col].str.lower().str.strip()
        df["label_encoded"] = df[label_col].map(LABEL_MAP)
        unmapped = df["label_encoded"].isnull().sum()
        if unmapped:
            log.warning(f"{unmapped} rows have unrecognised labels → set to NaN.")
        log.info("Label encoding done.")
        return df

    # ── 10. Save ─────────────────────────────

    def save(self, df: pd.DataFrame, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"Saved {len(df):,} rows to {path}")

    # ── Full pipeline ────────────────────────

    def run_pipeline(self, df: pd.DataFrame,
                     text_col: str = "text",
                     label_col: str = "label",
                     missing_strategy: str = "drop") -> pd.DataFrame:
        """
        Run all preprocessing steps in order.
        Returns the fully preprocessed DataFrame.
        """
        log.info("=== Starting preprocessing pipeline ===")
        self.explore(df, text_col)
        df = self.drop_duplicates(df, subset=[text_col])
        df = self.handle_missing(df, text_col, missing_strategy)
        df = self.clean_column(df, text_col)
        df = self.tokenize_column(df, text_col)
        df = self.remove_stopwords_column(df)
        df = self.normalize_column(df)
        df = self.encode_labels(df, label_col)
        log.info(f"=== Pipeline complete: {len(df):,} rows ready ===")
        return df