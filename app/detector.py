from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.heuristics import analyze_heuristics

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = ROOT_DIR / "models" / "tfidf_logreg_baseline.joblib"
DEFAULT_TRANSFORMER_DIR = ROOT_DIR / "models" / "transformer"


@dataclass
class DetectionResult:
    prompt: str
    score: float
    flagged: bool
    threshold: float
    signals: List[str] = field(default_factory=list)
    heuristic_score: float = 0.0
    baseline_score: float = 0.0
    transformer_score: float = 0.0


class GuardrailDetector:
    """Combines heuristic rules, TF-IDF + Logistic Regression, and DistilBERT into a single score."""

    def __init__(
        self,
        baseline_path: Path = DEFAULT_BASELINE_PATH,
        transformer_dir: Path = DEFAULT_TRANSFORMER_DIR,
        threshold: float = 0.5,
        heuristic_weight: float = 0.25,
        baseline_weight: float = 0.30,
        transformer_weight: float = 0.45,
        max_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self.max_length = max_length

        weight_sum = heuristic_weight + baseline_weight + transformer_weight
        if weight_sum <= 0:
            raise ValueError("Component weights must sum to a positive value.")
        self.heuristic_weight = heuristic_weight / weight_sum
        self.baseline_weight = baseline_weight / weight_sum
        self.transformer_weight = transformer_weight / weight_sum

        if not baseline_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found at {baseline_path}. Run scripts/train_baseline.py first."
            )
        if not transformer_dir.exists():
            raise FileNotFoundError(
                f"Transformer directory not found at {transformer_dir}. Run scripts/train_transformer.py first."
            )

        self.baseline_model = joblib.load(baseline_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
        self.transformer_model.to(self.device)
        self.transformer_model.eval()

    def predict_heuristic(self, prompt: str) -> tuple[float, List[str]]:
        res = analyze_heuristics(prompt)
        return float(res.score), list(res.signals)

    def predict_baseline(self, prompt: str) -> float:
        prob = float(self.baseline_model.predict_proba([prompt])[0][1])
        return min(1.0, max(0.0, prob))

    def predict_transformer(self, prompt: str) -> float:
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            injection_prob = float(probs[0][1].item())

        return min(1.0, max(0.0, injection_prob))

    def score_prompt(self, prompt: str, threshold: Optional[float] = None) -> DetectionResult:
        active_threshold = self.threshold if threshold is None else threshold
        signals: List[str] = []

        h_score, h_signals = self.predict_heuristic(prompt)
        signals.extend(h_signals)

        b_score = self.predict_baseline(prompt)
        if b_score >= 0.5:
            signals.append("tfidf_logreg_high_risk")

        t_score = self.predict_transformer(prompt)
        if t_score >= 0.5:
            signals.append("transformer_high_risk")

        combined = (
            self.heuristic_weight * h_score
            + self.baseline_weight * b_score
            + self.transformer_weight * t_score
        )
        combined_score = round(min(1.0, max(0.0, combined)), 4)

        return DetectionResult(
            prompt=prompt,
            score=combined_score,
            flagged=combined_score >= active_threshold,
            threshold=active_threshold,
            signals=signals,
            heuristic_score=round(h_score, 4),
            baseline_score=round(b_score, 4),
            transformer_score=round(t_score, 4),
        )