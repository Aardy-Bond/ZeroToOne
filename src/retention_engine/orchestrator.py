"""
Engagement Survival Forecast orchestrator.

Runs the four evidence layers in order, assembles the survival curves, and
produces the before/after comparison. This is the only module the dashboard
and the CLI need to import.
"""

from __future__ import annotations

import logging

from openai import OpenAI, OpenAIError

from .cliffhanger import analyse_ending
from .emotion_signals import analyse_emotions
from .engagement_forecast import build_curve, episode_quality_prior, risk_band
from .prompts import (
    RISK_DEEP_DIVE_SYSTEM,
    RISK_DEEP_DIVE_USER,
    format_cohort_block,
    format_evidence_block,
)
from .quality_proxy import get_quality_proxy
from .scene_features import build_client, measure_scenes, split_scenes
from .schemas import (
    ComparisonResult,
    CohortProfile,
    ForecastResult,
    MetricDelta,
    RiskExplanation,
    SceneAnalysis,
)
from .structural_features import (
    apply_sequence_signals,
    compute_scene_signals,
    describe_evidence,
)
from .target_cohort import DEFAULT_COHORT, counterfactual_cohorts

logger = logging.getLogger(__name__)

DEEP_DIVE_MODEL = "gpt-4o"
DEEP_DIVE_TEMPERATURE = 0.4
MAX_DEEP_DIVES = 3

# Reproducibility floors, measured rather than guessed. Re-running the forecast
# on a byte-identical episode still moves these numbers, because the Scene DNA
# sensor is not deterministic at temperature 0 and a seed does not fix it. Five
# repeat runs on samples/wexler_street_continuation.txt gave an Unlock Pull
# Index range of 37.4-45.8 (sd 3.25) and a survival range of 40.5-43.7. A
# before/after delta smaller than these bands says nothing about the revision.
UPI_NOISE_BAND = 8.5
SURVIVAL_NOISE_BAND = 3.5


class ForecastError(Exception):
    """Raised when a forecast cannot be produced."""


def analyse_scenes(
    script_text: str,
    *,
    client: OpenAI | None = None,
    use_emotion_classifier: bool = True,
) -> list[SceneAnalysis]:
    """Run all four evidence layers over a continuation."""
    scenes = split_scenes(script_text)
    logger.info("Split continuation into %d scene(s).", len(scenes))

    dna_list = measure_scenes(scenes, client=client)

    signals = [compute_scene_signals(scene) for scene in scenes]
    signals = apply_sequence_signals(signals, dna_list)

    emotions = analyse_emotions(
        [scene.text for scene in scenes],
        dna_list,
        use_classifier=use_emotion_classifier,
    )

    proxy = get_quality_proxy()
    qualities = [proxy.score(scene.text) for scene in scenes]

    return [
        SceneAnalysis(scene=scene, dna=dna, structural=signal, emotion=emotion, quality=quality)
        for scene, dna, signal, emotion, quality in zip(
            scenes, dna_list, signals, emotions, qualities
        )
    ]


def explain_risks(
    analyses: list[SceneAnalysis],
    primary_curve,
    cohort: CohortProfile,
    counterfactuals: list[CohortProfile],
    *,
    client: OpenAI | None = None,
    use_llm: bool = True,
    max_explanations: int = MAX_DEEP_DIVES,
) -> list[RiskExplanation]:
    """
    Explain the riskiest scenes.

    Deterministic evidence is assembled first and always returned. The LLM
    deep-dive runs for at most the top few risk zones and degrades to the
    deterministic version if it fails, so a rate limit costs detail rather
    than the whole panel.
    """
    ranked = sorted(primary_curve.scenes, key=lambda s: s.hazard, reverse=True)
    flagged = [s for s in ranked if risk_band(s.hazard) != "low"][:max_explanations]
    if not flagged:
        flagged = ranked[:1]

    explanations: list[RiskExplanation] = []

    for forecast_scene in flagged:
        analysis = next(
            (a for a in analyses if a.scene.index == forecast_scene.scene_index), None
        )
        if analysis is None:
            continue

        baseline = _deterministic_explanation(analysis, forecast_scene, cohort)

        if use_llm:
            enriched = _llm_explanation(
                analysis, forecast_scene, cohort, counterfactuals,
                total_scenes=len(analyses), client=client,
            )
            explanations.append(enriched or baseline)
        else:
            explanations.append(baseline)

    explanations.sort(key=lambda e: e.scene_index)
    return explanations


def _deterministic_explanation(
    analysis: SceneAnalysis, forecast_scene, cohort: CohortProfile
) -> RiskExplanation:
    evidence = describe_evidence(analysis.structural, analysis.dna)
    contributors = forecast_scene.top_risk_factors[:3]

    why = " ".join(evidence[:3]) if evidence else (
        f"Hazard {forecast_scene.hazard:.2f} is driven mainly by "
        + ", ".join(c.factor.replace("_", " ") for c in contributors)
        + "."
    )

    return RiskExplanation(
        scene_index=analysis.scene.index,
        why_risky=why,
        cohort_expectation=(
            f"{cohort.label} prefers a {cohort.pace_preference} pace with "
            f"{cohort.complexity_tolerance} complexity tolerance and came for "
            f"{cohort.emotional_preference}. Likely contributors here: "
            + ", ".join(f"{c.factor.replace('_', ' ')} (+{c.delta:.3f})" for c in contributors)
            + "."
        ),
        surgical_fix=(
            "Target the single largest contributor above rather than reworking the "
            "scene wholesale."
        ),
        trade_off="Run the counterfactual curves after editing to see which cohorts moved.",
        source="deterministic",
    )


def _llm_explanation(
    analysis: SceneAnalysis,
    forecast_scene,
    cohort: CohortProfile,
    counterfactuals: list[CohortProfile],
    *,
    total_scenes: int,
    client: OpenAI | None,
) -> RiskExplanation | None:
    try:
        openai_client = client or build_client()
    except Exception as exc:
        logger.warning("Deep dive skipped, no client: %s", exc)
        return None

    user = RISK_DEEP_DIVE_USER.format(
        cohort_block=format_cohort_block(cohort),
        counterfactual_block="\n".join(f"- {c.label}" for c in counterfactuals),
        scene_index=analysis.scene.index,
        evidence_block=format_evidence_block(analysis, forecast_scene),
        total_scenes=total_scenes,
        survival=forecast_scene.survival,
        scene_text=analysis.scene.text,
    )

    try:
        completion = openai_client.chat.completions.parse(
            model=DEEP_DIVE_MODEL,
            temperature=DEEP_DIVE_TEMPERATURE,
            messages=[
                {"role": "system", "content": RISK_DEEP_DIVE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format=RiskExplanation,
        )
    except OpenAIError as exc:
        logger.warning("Deep dive failed for scene %d: %s", analysis.scene.index, exc)
        return None

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return None

    # The model is told the scene number but the forecast owns the truth.
    parsed.scene_index = analysis.scene.index
    parsed.source = "llm_deep_dive"
    return parsed


def run_forecast(
    script_text: str,
    *,
    cohort: CohortProfile | None = None,
    client: OpenAI | None = None,
    use_llm_deep_dive: bool = True,
    use_emotion_classifier: bool = True,
    analyses: list[SceneAnalysis] | None = None,
) -> ForecastResult:
    """
    Produce a complete Engagement Survival Forecast.

    Pass `analyses` to re-score an already measured draft against a different
    cohort without paying for the sensor pass again.
    """
    if not (script_text or "").strip():
        raise ValueError("script_text must not be empty.")

    target = cohort or DEFAULT_COHORT
    scene_analyses = analyses or analyse_scenes(
        script_text, client=client, use_emotion_classifier=use_emotion_classifier
    )
    if not scene_analyses:
        raise ForecastError("No scenes could be extracted from the continuation.")

    primary = build_curve(scene_analyses, target)
    counterfactuals = counterfactual_cohorts(target)
    counterfactual_curves = [build_curve(scene_analyses, c) for c in counterfactuals]

    return ForecastResult(
        scenes=scene_analyses,
        primary_curve=primary,
        counterfactual_curves=counterfactual_curves,
        cliffhanger=analyse_ending(scene_analyses),
        risk_explanations=explain_risks(
            scene_analyses, primary, target, counterfactuals,
            client=client, use_llm=use_llm_deep_dive,
        ),
        quality_prior=episode_quality_prior(scene_analyses),
        cohort=target,
    )


def compare_forecasts(baseline: ForecastResult, revised: ForecastResult) -> ComparisonResult:
    """
    Diff two forecasts into an honest before/after card.

    Reports movement in both directions. A revision that lifts the primary
    cohort while hurting a counterfactual one shows both facts.
    """
    metrics = [
        MetricDelta(
            label="Overall engagement survival proxy",
            before=round(baseline.overall_survival, 1),
            after=round(revised.overall_survival, 1),
            noise_band=SURVIVAL_NOISE_BAND,
        ),
        MetricDelta(
            label=f"Primary cohort ({revised.cohort.label})",
            before=round(baseline.primary_curve.final_survival, 1),
            after=round(revised.primary_curve.final_survival, 1),
            noise_band=SURVIVAL_NOISE_BAND,
        ),
        MetricDelta(
            label="Unlock Pull Index",
            before=baseline.cliffhanger.unlock_pull_index,
            after=revised.cliffhanger.unlock_pull_index,
            noise_band=UPI_NOISE_BAND,
        ),
        MetricDelta(
            label="Hook strength",
            before=baseline.cliffhanger.hook_strength,
            after=revised.cliffhanger.hook_strength,
        ),
    ]

    if baseline.quality_prior.available and revised.quality_prior.available:
        metrics.append(
            MetricDelta(
                label="Narrative quality prior",
                before=round(baseline.quality_prior.score, 3),
                after=round(revised.quality_prior.score, 3),
            )
        )

    for before_curve, after_curve in zip(
        baseline.counterfactual_curves, revised.counterfactual_curves
    ):
        metrics.append(
            MetricDelta(
                label=after_curve.cohort_label,
                before=round(before_curve.final_survival, 1),
                after=round(after_curve.final_survival, 1),
                noise_band=SURVIVAL_NOISE_BAND,
            )
        )

    movement = _risk_movement(baseline, revised)
    evidence = _evidence_lines(baseline, revised)

    survival = metrics[0]
    if not survival.significant:
        headline = (
            f"No measurable change: relative engagement survival "
            f"{survival.delta:+.1f} points, inside the ±{SURVIVAL_NOISE_BAND:.0f}-point "
            f"range this sensor produces when re-measuring an unchanged episode."
        )
    elif survival.delta > 0:
        headline = (
            f"Forecast improvement: relative engagement survival "
            f"{survival.delta:+.1f} points."
        )
    else:
        headline = (
            f"Forecast regression: relative engagement survival "
            f"{survival.delta:+.1f} points."
        )

    return ComparisonResult(
        metrics=metrics,
        risk_scene_movement=movement,
        primary_evidence=evidence or ["No individual signal moved materially."],
        headline=headline,
    )


def _risk_movement(baseline: ForecastResult, revised: ForecastResult) -> list[str]:
    before = {s.scene_index: s.hazard for s in baseline.primary_curve.scenes}
    after = {s.scene_index: s.hazard for s in revised.primary_curve.scenes}

    lines: list[str] = []
    for index in sorted(set(before) | set(after)):
        if index not in before:
            lines.append(f"Scene {index}: new scene, hazard {after[index]:.3f}.")
            continue
        if index not in after:
            lines.append(f"Scene {index}: removed from the revision.")
            continue

        change = after[index] - before[index]
        if abs(change) < 0.01:
            continue
        direction = "fell" if change < 0 else "rose"
        lines.append(
            f"Scene {index}: hazard {direction} {abs(change):.3f} "
            f"({risk_band(before[index])} to {risk_band(after[index])})."
        )

    return lines or ["No scene changed risk band."]


def _evidence_lines(baseline: ForecastResult, revised: ForecastResult) -> list[str]:
    """Name the structural signals that actually moved."""
    lines: list[str] = []

    def worst(result: ForecastResult, attribute: str) -> int:
        return max((getattr(a.structural, attribute) for a in result.scenes), default=0)

    for attribute, label in (
        ("consecutive_exposition_scenes", "exposition fatigue"),
        ("payoff_debt", "payoff debt"),
        ("consecutive_low_conflict_scenes", "low-conflict streak"),
        ("pov_repeat_streak", "single-POV streak"),
    ):
        before, after = worst(baseline, attribute), worst(revised, attribute)
        if after < before:
            lines.append(f"Lower {label} ({before} to {after}).")
        elif after > before:
            lines.append(f"Higher {label} ({before} to {after}).")

    hook_delta = revised.cliffhanger.hook_strength - baseline.cliffhanger.hook_strength
    if abs(hook_delta) >= 2:
        lines.append(
            f"{'Stronger' if hook_delta > 0 else 'Weaker'} end hook "
            f"({baseline.cliffhanger.hook_strength:.0f} to "
            f"{revised.cliffhanger.hook_strength:.0f})."
        )

    before_complexity = max((a.dna.complexity for a in baseline.scenes), default=0.0)
    after_complexity = max((a.dna.complexity for a in revised.scenes), default=0.0)
    if after_complexity <= before_complexity + 0.02:
        lines.append("No increased complexity risk.")
    else:
        lines.append(
            f"Complexity rose ({before_complexity:.2f} to {after_complexity:.2f})."
        )

    return lines
