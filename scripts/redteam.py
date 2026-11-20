from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.detector import GuardrailDetector
from app.main import BLOCK_THRESHOLD, FLAG_THRESHOLD

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REDTEAM_CSV = ROOT_DIR / "data" / "redteam.csv"
DEFAULT_OUT_JSON = ROOT_DIR / "metrics" / "redteam.json"


def classify_pred(score: float, flag_threshold: float = FLAG_THRESHOLD, block_threshold: float = BLOCK_THRESHOLD) -> int:
    # Any prompt at or above flag_threshold is treated as flagged/blocked (positive = injection)
    return 1 if score >= flag_threshold else 0


def run_redteam(
    csv_path: Path = DEFAULT_REDTEAM_CSV,
    out_json: Path = DEFAULT_OUT_JSON,
    flag_threshold: float = FLAG_THRESHOLD,
    block_threshold: float = BLOCK_THRESHOLD,
    label_prefix: str = "",
) -> Dict[str, object]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing red-team dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"text", "expected_label", "category"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"redteam.csv must contain columns {required_cols}, got {list(df.columns)}")

    detector = GuardrailDetector(threshold=flag_threshold)

    bypasses: List[Dict[str, object]] = []
    false_blocks: List[Dict[str, object]] = []
    category_stats: Dict[str, Dict[str, int]] = {}

    tp = fp = tn = fn = 0

    for idx, row in df.iterrows():
        text = str(row["text"])
        expected = int(row["expected_label"])
        category = str(row["category"])

        res = detector.score_prompt(text, threshold=flag_threshold)
        pred = classify_pred(res.score, flag_threshold=flag_threshold, block_threshold=block_threshold)

        cat_entry = category_stats.setdefault(
            category, {"total": 0, "expected_pos": 0, "expected_neg": 0, "bypasses": 0, "false_blocks": 0}
        )
        cat_entry["total"] += 1
        if expected == 1:
            cat_entry["expected_pos"] += 1
            if pred == 1:
                tp += 1
            else:
                fn += 1
                cat_entry["bypasses"] += 1
                bypasses.append(
                    {
                        "row": int(idx),
                        "category": category,
                        "score": res.score,
                        "signals": res.signals,
                        "prompt_preview": text[:110],
                    }
                )
        else:
            cat_entry["expected_neg"] += 1
            if pred == 0:
                tn += 1
            else:
                fp += 1
                cat_entry["false_blocks"] += 1
                false_blocks.append(
                    {
                        "row": int(idx),
                        "category": category,
                        "score": res.score,
                        "signals": res.signals,
                        "prompt_preview": text[:110],
                    }
                )

    total_attacks = tp + fn
    total_benign = tn + fp
    bypass_rate = round(fn / total_attacks, 4) if total_attacks else 0.0
    false_block_rate = round(fp / total_benign, 4) if total_benign else 0.0
    detection_recall = round(tp / total_attacks, 4) if total_attacks else 0.0

    summary = {
        "flag_threshold": flag_threshold,
        "block_threshold": block_threshold,
        "total_cases": int(len(df)),
        "total_attacks": total_attacks,
        "total_benign": total_benign,
        "detected_attacks": tp,
        "bypasses_count": fn,
        "false_blocks_count": fp,
        "detection_recall": detection_recall,
        "bypass_rate": bypass_rate,
        "false_block_rate": false_block_rate,
        "by_category": category_stats,
        "bypasses": bypasses,
        "false_blocks": false_blocks,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    prefix = f"[{label_prefix}] " if label_prefix else ""
    print(f"=== {prefix}Red-Team Summary (flag_threshold={flag_threshold:.2f}) ===")
    print(
        f"Total={len(df)} | Attacks={total_attacks} (Detected={tp}, Bypasses={fn}, Recall={detection_recall:.4f}, BypassRate={bypass_rate:.4f}) | "
        f"Benign={total_benign} (FalseBlocks={fp}, FalseBlockRate={false_block_rate:.4f})"
    )

    print("\n--- Bypasses (Expected Injection=1, Missed) ---")
    if not bypasses:
        print("None!")
    else:
        print(f"{'Row':>4} {'Category':<28} {'Score':>6} {'Signals':<35} {'Prompt'}")
        for b in bypasses:
            print(
                f"{b['row']:>4} {b['category']:<28} {b['score']:>6.4f} "
                f"{','.join(b['signals'])[:35]:<35} {b['prompt_preview']}"
            )

    print("\n--- False Blocks (Expected Benign=0, Flagged/Blocked) ---")
    if not false_blocks:
        print("None!")
    else:
        print(f"{'Row':>4} {'Category':<28} {'Score':>6} {'Signals':<35} {'Prompt'}")
        for fb in false_blocks:
            print(
                f"{fb['row']:>4} {fb['category']:<28} {fb['score']:>6.4f} "
                f"{','.join(fb['signals'])[:35]:<35} {fb['prompt_preview']}"
            )

    print(f"\nSaved red-team report to: {out_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run red-team evaluation against GuardrailDetector.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_REDTEAM_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--flag-threshold", type=float, default=FLAG_THRESHOLD)
    parser.add_argument("--block-threshold", type=float, default=BLOCK_THRESHOLD)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    run_redteam(
        csv_path=args.csv,
        out_json=args.out,
        flag_threshold=args.flag_threshold,
        block_threshold=args.block_threshold,
        label_prefix=args.label,
    )


if __name__ == "__main__":
    main()