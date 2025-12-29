from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib

from src.preprocess import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_artifacts(models_dir: Path = MODELS_DIR):
    """
    Load vectorizer, model, and label map from disk.
    """
    vectorizer_path = models_dir / "vectorizer.pkl"
    model_path = models_dir / "model.pkl"
    labelmap_path = models_dir / "label_map.json"

    missing = [p for p in [vectorizer_path, model_path, labelmap_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Train first:\n"
            "  python -m src.train --data data/fake_or_real_news.csv\n\n"
            f"Missing files: {', '.join(str(p) for p in missing)}"
        )

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)

    with open(labelmap_path, "r", encoding="utf-8") as f:
        label_map: Dict[str, int] = json.load(f)

    # Invert label map to go from number -> label
    inv_label_map = {int(v): k for k, v in label_map.items()}

    return vectorizer, model, label_map, inv_label_map


def predict_text(text: str) -> Dict[str, object]:
    """
    Predict REAL/FAKE for a single input text.

    Returns a dict you can reuse in Streamlit:
      {
        "label": "FAKE" or "REAL",
        "label_id": 0/1,
        "decision_score": float or None
      }
    """
    vectorizer, model, _, inv_label_map = load_artifacts()

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    label_id = int(model.predict(X)[0])
    label = inv_label_map.get(label_id, str(label_id))

    # LinearSVC doesn't provide predict_proba; it provides a decision function.
    decision_score = None
    if hasattr(model, "decision_function"):
        # for binary classification, decision_function returns shape (1,)
        decision_score = float(model.decision_function(X)[0])

    return {
        "label": label,
        "label_id": label_id,
        "decision_score": decision_score,
    }


def _read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Raw text to classify")
    parser.add_argument("--file", type=str, default=None, help="Path to a .txt file to classify")
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit(
            "Provide --text or --file.\n"
            "Example:\n"
            '  python -m src.predict --text "Some news text..."\n'
            "  python -m src.predict --file data/sample.txt"
        )

    if args.file:
        text = _read_text_file(Path(args.file))
    else:
        text = args.text or ""

    result = predict_text(text)

    print("\n=== Prediction ===")
    print("Label:", result["label"])
    if result["decision_score"] is not None:
        print("Decision score (LinearSVC):", round(result["decision_score"], 4))
        print("(Higher usually means more toward the positive class.)")


if __name__ == "__main__":
    main()
