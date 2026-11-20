from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.heuristics import analyze_heuristics

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformer"
METRICS_DIR = ROOT_DIR / "metrics"

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def compute_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negatives = y_true == 0
    fp = int(np.sum((y_pred == 1) & negatives))
    tn = int(np.sum((y_pred == 0) & negatives))
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def summarize_latency(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms, dtype=float)
    return {
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "mean": round(float(np.mean(arr)), 4),
    }


def metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": round(compute_false_positive_rate(y_true, y_pred), 4),
    }


def score_heuristics_all(texts: List[str]) -> tuple[np.ndarray, List[float]]:
    probs: List[float] = []
    latencies: List[float] = []
    for text in texts:
        t0 = time.perf_counter()
        res = analyze_heuristics(text)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        probs.append(float(res.score))
    return np.array(probs, dtype=float), latencies


def score_baseline_all(model, texts: List[str]) -> tuple[np.ndarray, List[float]]:
    if len(texts) > 0:
        _ = model.predict_proba([texts[0]])
    probs: List[float] = []
    latencies: List[float] = []
    for text in texts:
        t0 = time.perf_counter()
        p = float(model.predict_proba([text])[0][1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        probs.append(p)
    return np.array(probs, dtype=float), latencies


def score_transformer_all(
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    texts: List[str],
    max_length: int = 128,
) -> tuple[np.ndarray, List[float]]:
    model.eval()
    if len(texts) > 0:
        warm = tokenizer(texts[0], truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            _ = model(input_ids=warm["input_ids"].to(device), attention_mask=warm["attention_mask"].to(device))

    probs: List[float] = []
    latencies: List[float] = []
    for text in texts:
        enc = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            p = float(torch.softmax(logits, dim=-1)[0][1].item())
        latencies.append((time.perf_counter() - t0) * 1000.0)
        probs.append(p)
    return np.array(probs, dtype=float), latencies


def main() -> None:
    test_path = DATA_DIR / "test.csv"
    baseline_path = MODELS_DIR / "tfidf_logreg_baseline.joblib"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}. Run scripts/prepare_data.py first.")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline model: {baseline_path}. Run scripts/train_baseline.py first.")
    if not TRANSFORMER_DIR.exists():
        raise FileNotFoundError(f"Missing transformer model: {TRANSFORMER_DIR}. Run scripts/train_transformer.py first.")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_path)
    texts = test_df["text"].fillna("").astype(str).tolist()
    y_true = test_df["label"].astype(int).to_numpy()

    baseline_model = joblib.load(baseline_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR).to(device)

    h_probs, h_lat = score_heuristics_all(texts)
    b_probs, b_lat = score_baseline_all(baseline_model, texts)
    t_probs, t_lat = score_transformer_all(tokenizer, transformer_model, device, texts, max_length=128)

    combined_probs = np.clip(0.25 * h_probs + 0.30 * b_probs + 0.45 * t_probs, 0.0, 1.0)
    combined_lat = [h + b + t for h, b, t in zip(h_lat, b_lat, t_lat)]

    # Keep backwards-compatible baseline.json at threshold=0.50
    baseline_only_summary = {
        "model": "tfidf_logreg_baseline",
        "num_samples": int(len(texts)),
        **metrics_at_threshold(y_true, b_probs, 0.50),
        "latency_ms": summarize_latency(b_lat),
    }
    with (METRICS_DIR / "baseline.json").open("w", encoding="utf-8") as f:
        json.dump(baseline_only_summary, f, indent=2)

    model_scores_map = {
        "baseline_tfidf_logreg": (b_probs, b_lat),
        "transformer_distilbert": (t_probs, t_lat),
        "combined_guardrail": (combined_probs, combined_lat),
    }

    comparison_models: Dict[str, object] = {}
    threshold_sweep: Dict[str, List[Dict[str, float]]] = {}

    for name, (probs, lats) in model_scores_map.items():
        comparison_models[name] = {
            "num_samples": int(len(texts)),
            **metrics_at_threshold(y_true, probs, 0.50),
            "latency_ms": summarize_latency(lats),
        }
        threshold_sweep[name] = [metrics_at_threshold(y_true, probs, th) for th in THRESHOLDS]

    full_report = {
        "num_samples": int(len(texts)),
        "default_threshold": 0.50,
        "weights": {
            "heuristics": 0.25,
            "baseline_tfidf_logreg": 0.30,
            "transformer_distilbert": 0.45,
        },
        "models": comparison_models,
        "threshold_sweep": threshold_sweep,
    }

    comparison_path = METRICS_DIR / "comparison.json"
    sweep_path = METRICS_DIR / "threshold_sweep.json"

    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    with sweep_path.open("w", encoding="utf-8") as f:
        json.dump(threshold_sweep, f, indent=2)

    print("=== Model Comparison @ threshold=0.50 ===")
    print(f"{'Model':<24} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7} {'P50(ms)':>9} {'P95(ms)':>9}")
    for name, m in comparison_models.items():
        print(
            f"{name:<24} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} "
            f"{m['false_positive_rate']:>7.4f} {m['latency_ms']['p50']:>9.2f} {m['latency_ms']['p95']:>9.2f}"
        )

    print("\n=== Combined Guardrail Threshold Sweep (Recall vs FPR) ===")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8}")
    for row in threshold_sweep["combined_guardrail"]:
        print(
            f"{row['threshold']:>10.2f} {row['precision']:>10.4f} {row['recall']:>8.4f} "
            f"{row['f1']:>8.4f} {row['false_positive_rate']:>8.4f}"
        )

    print(f"\nSaved metrics to: {comparison_path} and {sweep_path}")


if __name__ == "__main__":
    main()