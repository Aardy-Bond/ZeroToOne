"""
Compress a ForecastResult into the brief the producer agent reads.

Writers never fill this by hand — it is assembled from the read-through
session state so casting and cues stay grounded in survival, EKG-ish
signals, and cliffhanger scores.
"""

from __future__ import annotations

from retention_engine.engagement_forecast import risk_band
from retention_engine.schemas import ForecastResult

from .schemas import SpeakableLine

TOP_RISK_SCENES = 4
CONTRIBUTIONS_PER_SCENE = 3


def pack_forecast_context(
    forecast: ForecastResult,
    lines: list[SpeakableLine],
    *,
    top_risk: int = TOP_RISK_SCENES,
) -> str:
    """Plain-text agent brief: scores first, then numbered lines."""
    cliff = forecast.cliffhanger
    cohort = forecast.cohort
    sections: list[str] = [
        "RETENTION FORECAST (simulated pre-release — not live listener data)",
        f"Cohort: {cohort.label} · genre={cohort.genre_affinity} · "
        f"pace={cohort.pace_preference} · mode={cohort.listening_mode}",
        f"Overall survival proxy: {forecast.overall_survival:.1f} / 100",
        f"Unlock Pull Index: {cliff.unlock_pull_index:.1f} / 100",
        f"Cliffhanger types: {', '.join(cliff.types) or 'none'}",
        f"Cliffhanger note: {cliff.recommendation}",
        "",
        "EKG SNAPSHOT (per scene)",
    ]

    hazard_by = {s.scene_index: s for s in forecast.primary_curve.scenes}
    for analysis in forecast.scenes:
        sf = hazard_by.get(analysis.scene.index)
        hazard = sf.hazard if sf else 0.0
        survival = sf.survival if sf else 0.0
        sections.append(
            f"  scene {analysis.scene.index} [{analysis.scene.heading}] "
            f"tempo={analysis.dna.scene_tempo:.2f} "
            f"arousal={analysis.emotion.arousal:.2f} "
            f"tension={analysis.dna.tension:.2f} "
            f"exposition={analysis.dna.exposition_ratio:.2f} "
            f"hazard={hazard:.3f} ({risk_band(hazard)}) "
            f"survival={survival:.1f} "
            f"emotion={analysis.dna.dominant_emotion}"
        )

    sections.append("")
    sections.append("TOP RISK SCENES")
    for scene_fc in forecast.risk_ranking[:top_risk]:
        factors = scene_fc.top_risk_factors[:CONTRIBUTIONS_PER_SCENE]
        detail = "; ".join(f"{c.factor} (+{c.delta:.3f}: {c.detail})" for c in factors)
        sections.append(
            f"  scene {scene_fc.scene_index}: hazard={scene_fc.hazard:.3f}, "
            f"survival={scene_fc.survival:.1f}"
            + (f" — {detail}" if detail else "")
        )

    if forecast.risk_explanations:
        sections.append("")
        sections.append("RISK NOTES")
        for note in forecast.risk_explanations[:top_risk]:
            sections.append(
                f"  scene {note.scene_index}: {note.why_risky} "
                f"Fix: {note.surgical_fix}"
            )

    characters = sorted({line.character for line in lines})
    sections.append("")
    sections.append(f"CAST LIST (use only these names): {', '.join(characters)}")
    sections.append("")
    sections.append("NUMBERED SPEAKABLE LINES")
    for line in lines:
        preview = line.text if len(line.text) <= 220 else line.text[:217] + "..."
        sections.append(
            f"[{line.index}|scene {line.scene_index}|{line.character}] {preview}"
        )

    sections.append("")
    sections.append(
        "Produce a ProducerPlan: casting for every cast member, one SceneDirection "
        "per scene that appears, selective LineCues (about one per 2–3 lines; "
        "always cover cold open, reveals, high-hazard scenes, final hook), "
        "sound cues on important beats, and a compact MarketingBrief. "
        "Ground every choice in the scores above. Do not invent characters."
    )
    return "\n".join(sections)
