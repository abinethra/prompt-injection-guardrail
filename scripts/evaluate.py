from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"


def compute_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # FPR = FP / (FP + TN), where positive class is 1 and negative (benign) class is 0
    negatives = y_true == 0
    fp = int(np.sum((y_pred == 1) & negatives))
    tn = int(np.sum((y_pred == 0) & negatives))
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def evaluate_test_split(threshold: float = 0.5) -> Dict[str, object]:
    test_path = DATA_DIR / "test.csv"
    model_path = MODELS_DIR / "tfidf_logreg_baseline.joblib"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}. Run scripts/prepare_data.py first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}. Run scripts/train_baseline.py first.")

    test_df = pd.read_csv(test_path)
    texts: List[str] = test_df["text"].fillna("").astype(str).tolist()
    y_true = test_df["label"].astype(int).to_numpy()

    model = joblib.load(model_path)

    latencies_ms: List[float] = []
    probs_list: List[float] = []

    for text in texts:
        start = time.perf_counter()
        prob = float(model.predict_proba([text])[0][1])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        probs_list.append(prob)

    probs = np.array(probs_list, dtype=float)
    y_pred = (probs >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = compute_false_positive_rate(y_true, y_pred)

    lat_arr = np.array(latencies_ms, dtype=float)
    p50_ms = float(np.percentile(lat_arr, 50))
    p95_ms = float(np.percentile(lat_arr, 95))
    mean_ms = float(np.mean(lat_arr))

    return {
        "model": "tfidf_logreg_baseline",
        "threshold": threshold,
        "num_samples": int(len(texts)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "latency_ms": {
            "p50": round(p50_ms, 4),
            "p95": round(p95_ms, 4),
            "mean": round(mean_ms, 4),
        },
    }


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_test_split(threshold=0.5)

    out_path = METRICS_DIR / "baseline.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== Test Split Baseline Evaluation ===")
    print(f"Samples:              {results['num_samples']}")
    print(f"Precision:            {results['precision']:.4f}")
    print(f"Recall:               {results['recall']:.4f}")
    print(f"F1 Score:             {results['f1']:.4f}")
    print(f"False Positive Rate:  {results['false_positive_rate']:.4f}")
    print(f"Latency P50:          {results['latency_ms']['p50']:.4f} ms")
    print(f"Latency P95:          {results['latency_ms']['p95']:.4f} ms")
    print(f"Saved metrics to:     {out_path}")


if __name__ == "__main__":
    main()