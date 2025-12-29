from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from src.preprocess import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "fake_or_real_news.csv"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at: {csv_path}\n"
            f"Put your CSV there, or pass a path like:\n"
            f"  python -m src.train --data path/to/file.csv"
        )

    df = pd.read_csv(csv_path)

    # Expected columns based on your notebook: title/text/label (after dropping others)
    # We'll be robust: we only need "text" and "label".
    required = {"text", "label"}
    missing = required - set(df.columns.str.lower())

    # try case-insensitive mapping if needed
    colmap = {c.lower(): c for c in df.columns}
    if missing:
        raise ValueError(
            f"Dataset must include columns {required}. "
            f"Found columns: {list(df.columns)}"
        )

    text_col = colmap["text"]
    label_col = colmap["label"]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text", "label"]).drop_duplicates()

    return df


def encode_labels(df: pd.DataFrame) -> Tuple[pd.Series, dict]:
    """
    Convert labels to 0/1 with a stable mapping.
    We’ll map: FAKE -> 0, REAL -> 1 (common & intuitive).
    """
    label_map = {"FAKE": 0, "REAL": 1}
    y = df["label"].astype(str).str.upper().map(label_map)

    if y.isna().any():
        bad = df.loc[y.isna(), "label"].unique().tolist()
        raise ValueError(
            f"Unexpected label values found: {bad}. "
            f"Expected only: {list(label_map.keys())}"
        )

    return y.astype(int), label_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to CSV dataset")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--max_features", type=int, default=50000, help="Max TF-IDF vocab size")
    parser.add_argument("--ngram_max", type=int, default=2, help="Max n-gram size (1=unigram, 2=uni+bi)")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    df = load_data(data_path)
    y, label_map = encode_labels(df)

    # Preprocess text (reusing your pipeline from preprocess.py)
    X_text = df["text"].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # TF-IDF (like your notebook), with optional n-grams
    vectorizer = TfidfVectorizer(
        stop_words=None,  # stopwords already handled in clean_text()
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model: Linear SVM (fast + strong for TF-IDF)
    model = LinearSVC()

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # Evaluation
    print("\n=== Evaluation ===")
    print("F1 (macro):", round(f1_score(y_test, y_pred, average="macro"), 4))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    # Save artifacts
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    model_path = MODELS_DIR / "model.pkl"
    labelmap_path = MODELS_DIR / "label_map.json"

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    with open(labelmap_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print("\n✅ Saved artifacts:")
    print(f"- {vectorizer_path}")
    print(f"- {model_path}")
    print(f"- {labelmap_path}")


if __name__ == "__main__":
    main()
