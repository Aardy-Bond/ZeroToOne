#!/usr/bin/env python3
"""
Train the narrative quality prior.

Fits a TF-IDF + Ridge regressor on lars1234/story_writing_benchmark, which
contains LLM-generated short stories with per-dimension craft ratings. The
artifact it writes lets the retention engine attach a story-quality estimate
to each scene.

This predicts rated story quality. It does not predict retention, and the
dataset contains no listener behaviour of any kind.

Usage:
    python scripts/train_quality_proxy.py
    python scripts/train_quality_proxy.py --max-samples 4000 --label overall_score
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = ARTIFACT_DIR / "quality_prior.joblib"
METADATA_PATH = ARTIFACT_DIR / "quality_prior.meta.json"

DATASET = "lars1234/story_writing_benchmark"
CONFIG = "average"
FEATURE_VERSION = "qp-1"

TEXT_COLUMN = "story_text"
DEFAULT_LABEL = "overall_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_LABEL, help="Target column to regress on.")
    parser.add_argument(
        "--max-samples", type=int, default=6000,
        help="Cap on training rows. Keeps a CPU-only fit under a couple of minutes.",
    )
    parser.add_argument(
        "--language", default="en",
        help="Filter to one language, or 'all'. The dataset also has es and de.",
    )
    parser.add_argument("--max-features", type=int, default=40000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import numpy as np
        from datasets import load_dataset
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        import joblib
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print("Install with: pip install scikit-learn joblib datasets", file=sys.stderr)
        return 1

    print(f"Loading {DATASET} (config={CONFIG})...")
    try:
        dataset = load_dataset(DATASET, CONFIG, split="train")
    except Exception as exc:
        print(f"Failed to download dataset: {exc}", file=sys.stderr)
        print("The retention engine runs without this artifact; the quality", file=sys.stderr)
        print("term is simply omitted from the forecast.", file=sys.stderr)
        return 1

    frame = dataset.to_pandas()
    print(f"  {len(frame)} rows, {len(frame.columns)} columns")

    if args.language != "all" and "language" in frame.columns:
        frame = frame[frame["language"] == args.language]
        print(f"  {len(frame)} rows after filtering to language={args.language}")

    if args.label not in frame.columns:
        print(f"Label '{args.label}' not in dataset. Available: "
              f"{[c for c in frame.columns if c.startswith('q') or 'score' in c]}",
              file=sys.stderr)
        return 1

    frame = frame[[TEXT_COLUMN, args.label]].dropna()
    frame = frame[frame[TEXT_COLUMN].str.split().str.len() >= 50]
    print(f"  {len(frame)} rows after dropping nulls and very short stories")

    if len(frame) < 200:
        print("Too few usable rows to train a meaningful model.", file=sys.stderr)
        return 1

    if len(frame) > args.max_samples:
        frame = frame.sample(n=args.max_samples, random_state=args.seed)
        print(f"  sampled down to {len(frame)} rows")

    texts = frame[TEXT_COLUMN].tolist()
    labels = frame[args.label].astype(float).to_numpy()

    label_min = float(np.min(labels))
    label_max = float(np.max(labels))
    print(f"  label '{args.label}' spans {label_min:.2f} to {label_max:.2f} "
          f"(mean {labels.mean():.2f})")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed
    )

    # Word and character n-grams together: word features capture vocabulary
    # and phrasing, character features pick up rhythm and punctuation habits
    # that survive tokenisation.
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, 2),
                    min_df=3,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("ridge", Ridge(alpha=1.0, random_state=args.seed)),
        ]
    )

    print(f"Fitting on {len(x_train)} stories...")
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    r2 = float(r2_score(y_test, predictions))
    mae = float(mean_absolute_error(y_test, predictions))
    baseline_mae = float(mean_absolute_error(y_test, np.full_like(y_test, y_train.mean())))

    print(f"  R^2  {r2:.4f}")
    print(f"  MAE  {mae:.4f}  (predict-the-mean baseline {baseline_mae:.4f})")

    if r2 <= 0:
        print()
        print("The fit explains no variance beyond the mean. Refusing to write an", file=sys.stderr)
        print("artifact that would add noise to the forecast. The engine will", file=sys.stderr)
        print("report the quality prior as unavailable.", file=sys.stderr)
        return 1

    top_contributors = _top_terms(pipeline)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)

    metadata = {
        "model_version": f"{FEATURE_VERSION}-{datetime.now(timezone.utc):%Y%m%d}",
        "feature_version": FEATURE_VERSION,
        "dataset": DATASET,
        "dataset_config": CONFIG,
        "dataset_license": "MIT",
        "label": args.label,
        "language_filter": args.language,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(x_train),
        "test_count": len(x_test),
        "label_min": label_min,
        "label_max": label_max,
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "baseline_mae": round(baseline_mae, 4),
        "estimator": "TfidfVectorizer(1,2) + Ridge(alpha=1.0)",
        "top_contributors": top_contributors,
        "caveat": (
            "Predicts rated story quality on LLM-written short fiction. "
            "Not a retention label. Contains no listener behaviour."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"Wrote {ARTIFACT_PATH}")
    print(f"Wrote {METADATA_PATH}")
    return 0


def _top_terms(pipeline, count: int = 10) -> list[str]:
    """Highest positively weighted n-grams, for display as evidence."""
    try:
        vectorizer = pipeline.named_steps["tfidf"]
        ridge = pipeline.named_steps["ridge"]
        names = vectorizer.get_feature_names_out()
        coefficients = ridge.coef_
        ranked = sorted(zip(names, coefficients), key=lambda pair: pair[1], reverse=True)
        return [term for term, _ in ranked[:count]]
    except Exception:
        return []


if __name__ == "__main__":
    raise SystemExit(main())
