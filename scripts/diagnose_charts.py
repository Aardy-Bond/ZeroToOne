"""
What do the EKG and the survival curve actually show?

Runs a real forecast and prints the numbers behind both charts, so the question
"does this represent anything" is answered with the spread of the data rather
than an opinion about the design.
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from retention_engine.orchestrator import run_forecast
from retention_engine.target_cohort import DEFAULT_COHORT

load_dotenv()

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "night_ward_next_scene.txt"


def bar(value: float, width: int = 28) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def main() -> int:
    text = SAMPLE.read_text()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Forecasting {SAMPLE.name} ({len(text.split())} words)...\n")
    forecast = run_forecast(text, cohort=DEFAULT_COHORT, client=client, use_llm_deep_dive=False)

    scenes = forecast.scenes
    series = {
        "Tension": [a.dna.tension for a in scenes],
        "Emotional intensity": [a.emotion.arousal for a in scenes],
        "Event movement": [a.dna.event_movement for a in scenes],
        "Quality prior": [a.quality.score for a in scenes],
        "Cliffhanger": [a.dna.cliffhanger_strength for a in scenes],
        "Exposition": [a.dna.exposition_ratio for a in scenes],
    }

    print(f"{len(scenes)} scenes\n")
    print("=" * 78)
    print("THE EKG: six series, all drawn on one 0-1 axis")
    print("=" * 78)
    print(f"{'series':22} {'min':>5} {'max':>5} {'range':>6} {'sd':>5}   values")
    for name, values in series.items():
        spread = max(values) - min(values)
        sd = statistics.pstdev(values)
        shown = " ".join(f"{v:.2f}" for v in values)
        print(
            f"{name:22} {min(values):5.2f} {max(values):5.2f} "
            f"{spread:6.2f} {sd:5.2f}   {shown}"
        )

    print("\nHow much of the 0-1 axis each series actually occupies:")
    for name, values in series.items():
        lo, hi = min(values), max(values)
        pad = "·" * int(lo * 40)
        band = "█" * max(1, int((hi - lo) * 40))
        print(f"  {name:22} |{pad}{band}".ljust(68) + f"| {lo:.2f}-{hi:.2f}")

    print("\n" + "=" * 78)
    print("THE SURVIVAL CURVE")
    print("=" * 78)
    curve = forecast.primary_curve.curve
    print(f"{'scene':>6}  {'survival':>8}  {'hazard':>7}  shape")
    print(f"{'start':>6}  {curve[0]:8.1f}  {'-':>7}  {bar(1.0)}")
    for point, sf in zip(curve[1:], forecast.primary_curve.scenes):
        print(
            f"{sf.scene_index:>6}  {point:8.1f}  {sf.hazard:7.3f}  {bar(point / 100)}"
        )

    deltas = [curve[i] - curve[i - 1] for i in range(1, len(curve))]
    print(f"\n  monotonically decreasing: {all(d <= 0 for d in deltas)}")
    print(f"  total drop: {curve[0] - curve[-1]:.1f} points")
    print(f"  biggest single drop: scene {deltas.index(min(deltas)) + 1}, {min(deltas):.1f}")

    print("\n  Counterfactual cohorts (do they disagree?)")
    print(f"    {forecast.primary_curve.cohort_label:38} ends at "
          f"{forecast.primary_curve.final_survival:5.1f}   (primary)")
    for c in forecast.counterfactual_curves:
        print(f"    {c.cohort_label:38} ends at {c.final_survival:5.1f}")

    print("\n" + "=" * 78)
    print("HAZARD CONTRIBUTIONS: what actually drives the curve")
    print("=" * 78)
    totals: dict[str, float] = {}
    for sf in forecast.primary_curve.scenes:
        for c in sf.contributions:
            totals[c.factor] = totals.get(c.factor, 0) + abs(c.delta)
    for label, value in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {label:38} {value:6.3f}")

    dead = [label for label, value in totals.items() if value < 0.001]
    if dead:
        print(f"\n  never moved: {', '.join(dead)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
