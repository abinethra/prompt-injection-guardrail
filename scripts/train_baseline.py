from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_SEED = 42


def load_split(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run scripts/prepare_data.py first.")
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def build_pipeline() -> Pipeline:
    word_char_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        analyzer="char_wb",
        max_features=30000,
        sublinear_tf=True,
    )
    # Combine word n-grams and char_wb n-grams using a FeatureUnion
    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    analyzer="word",
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            ("char_wb", word_char_vectorizer),
        ]
    )

    classifier = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    return Pipeline(
        steps=[
            ("features", features),
            ("clf", classifier),
        ]
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train.csv")
    val_df = load_split("val.csv")

    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["label"])

    val_probs = pipeline.predict_proba(val_df["text"])[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_f1 = f1_score(val_df["label"], val_preds, zero_division=0)

    print("=== Validation Set Performance (threshold=0.50) ===")
    print(f"Validation F1: {val_f1:.4f}")
    print(classification_report(val_df["label"], val_preds, target_names=["benign", "injection"], zero_division=0))

    model_path = MODELS_DIR / "tfidf_logreg_baseline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved baseline model to: {model_path}")


if __name__ == "__main__":
    main()