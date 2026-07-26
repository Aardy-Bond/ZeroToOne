"""
Render the forecast charts to PNG so they can be looked at outside the app.

Captures the Plotly figures the dashboard would draw by intercepting
`st.plotly_chart`, which avoids splitting the render functions in two just to
make them inspectable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from retention_engine.orchestrator import run_forecast
from retention_engine.target_cohort import DEFAULT_COHORT

load_dotenv()

OUT = Path(__file__).resolve().parents[1] / "output" / "charts"
SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "night_ward_next_scene.txt"


def main() -> int:
    captured: list = []
    st.plotly_chart = lambda figure, **kwargs: captured.append(figure)

    from dashboard import forecast_view

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    forecast = run_forecast(
        SAMPLE.read_text(),
        cohort=DEFAULT_COHORT,
        client=client,
        use_llm_deep_dive=False,
    )

    OUT.mkdir(parents=True, exist_ok=True)

    forecast_view.render_survival_curves(forecast)
    forecast_view.render_narrative_ekg(forecast)

    names = ["survival.png", "ekg.png"]
    for figure, name in zip(captured, names):
        path = OUT / name
        figure.write_image(str(path), width=1100, scale=2)
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
