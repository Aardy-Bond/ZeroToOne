"""
The Engagement Survival Proxy.

This is a transparent, hand-specified hazard model, not a trained survival
model. Every term is a named constant multiplied by a measured signal, and
every term that fires is returned as a HazardContribution so the UI can show
the writer the arithmetic. Nothing here is fitted to listener data, because no
listener data exists for it to be fitted to.

Survival decays multiplicatively from 100: each scene keeps (1 - hazard) of
whatever attention reached it. That shape is a modelling choice, chosen
because attention lost early cannot be re-lost later.
"""

from __future__ import annotations

from .schemas import (
    CohortCurve,
    CohortProfile,
    HazardContribution,
    QualityPrior,
    SceneAnalysis,
    SceneForecast,
)
from .target_cohort import CohortVector, to_vector

# Every scene carries some risk simply by asking for another minute of
# attention. Without this the curve would stay flat on a merely inoffensive
# episode, which would be misleading.
BASELINE_HAZARD = 0.025

W_PACE_MISMATCH = 0.13
W_COMPLEXITY_OVERSHOOT = 0.22
W_EXPOSITION_OVERSHOOT = 0.30
W_EXPOSITION_FATIGUE = 0.055        # per consecutive exposition-heavy scene beyond the first
W_LOW_EVENT_MOVEMENT = 0.16
W_EMOTIONAL_FLATNESS = 0.14
W_QUALITY_PENALTY = 0.18
W_PAYOFF_DEBT = 0.030               # per unresolved question beyond the grace allowance
W_POV_FATIGUE = 0.035               # per scene of an unbroken POV streak beyond the allowance
W_CONTENT_OVERSHOOT = 0.20

W_COHORT_ALIGNMENT = 0.15
W_CLIFFHANGER_LIFT = 0.12
W_PROGRESSION_LIFT = 0.07

# A few open threads are how serialized fiction works; debt only becomes a
# drag once it outpaces what the listener can hold.
PAYOFF_DEBT_GRACE = 2
POV_STREAK_GRACE = 2

EMOTIONAL_FLATNESS_FLOOR = 0.35
LOW_EVENT_FLOOR = 0.35

# Clamped so no single scene can wipe out the curve or push it upward.
HAZARD_FLOOR = 0.005
HAZARD_CEILING = 0.45

RISK_ELEVATED = 0.14
RISK_HIGH = 0.24

# A relative proxy should never render as absolute zero. Over a long enough
# run of bad scenes the multiplicative decay underflows to 0.00, which reads
# as "every listener left" — a claim this model is in no position to make.
SURVIVAL_FLOOR = 0.1


def _contribution(factor: str, delta: float, detail: str) -> HazardContribution:
    return HazardContribution(factor=factor, delta=round(delta, 4), detail=detail)


def scene_hazard(
    analysis: SceneAnalysis,
    vector: CohortVector,
    *,
    is_final_scene: bool = False,
) -> tuple[float, list[HazardContribution]]:
    """
    Compute one scene's hazard and the named terms that produced it.

    Returns the clamped hazard plus every contribution that actually fired.
    Terms that evaluate to zero are omitted, so the contribution list reads as
    "here is what went wrong here" rather than a wall of zeroes.
    """
    dna = analysis.dna
    structural = analysis.structural
    contributions: list[HazardContribution] = [
        _contribution("baseline", BASELINE_HAZARD, "Cost of asking for another minute of attention.")
    ]
    hazard = BASELINE_HAZARD

    # --- Risk-raising terms ------------------------------------------------

    pace_gap = vector.pace_mismatch(dna)
    if pace_gap > 0.12:
        delta = W_PACE_MISMATCH * pace_gap
        hazard += delta
        contributions.append(
            _contribution(
                "pace_mismatch", delta,
                f"Scene tempo {dna.scene_tempo:.2f} against a target of "
                f"{vector.pace_target:.2f} for this cohort.",
            )
        )

    complexity_gap = vector.complexity_overshoot(dna)
    if complexity_gap > 0:
        delta = W_COMPLEXITY_OVERSHOOT * complexity_gap
        hazard += delta
        contributions.append(
            _contribution(
                "complexity_overshoot", delta,
                f"Complexity {dna.complexity:.2f} exceeds this cohort's ceiling of "
                f"{vector.complexity_ceiling:.2f}.",
            )
        )

    exposition_gap = vector.exposition_overshoot(dna)
    if exposition_gap > 0:
        delta = W_EXPOSITION_OVERSHOOT * exposition_gap
        hazard += delta
        contributions.append(
            _contribution(
                "exposition_overshoot", delta,
                f"Exposition {dna.exposition_ratio:.0%} against a tolerance of "
                f"{vector.exposition_tolerance:.0%}.",
            )
        )

    fatigue_scenes = max(0, structural.consecutive_exposition_scenes - 1)
    if fatigue_scenes:
        delta = W_EXPOSITION_FATIGUE * fatigue_scenes
        hazard += delta
        contributions.append(
            _contribution(
                "exposition_fatigue", delta,
                f"{structural.consecutive_exposition_scenes} consecutive exposition-heavy scenes.",
            )
        )

    if dna.event_movement < LOW_EVENT_FLOOR:
        shortfall = LOW_EVENT_FLOOR - dna.event_movement
        delta = W_LOW_EVENT_MOVEMENT * (shortfall / LOW_EVENT_FLOOR)
        if not dna.conflict_present:
            delta *= 1.4
        hazard += delta
        contributions.append(
            _contribution(
                "low_event_movement", delta,
                f"Event movement {dna.event_movement:.2f}"
                + ("; no active conflict." if not dna.conflict_present else "."),
            )
        )

    if dna.emotional_intensity < EMOTIONAL_FLATNESS_FLOOR:
        shortfall = EMOTIONAL_FLATNESS_FLOOR - dna.emotional_intensity
        delta = W_EMOTIONAL_FLATNESS * (shortfall / EMOTIONAL_FLATNESS_FLOOR)
        hazard += delta
        contributions.append(
            _contribution(
                "emotional_flatness", delta,
                f"Peak emotional intensity {dna.emotional_intensity:.2f}.",
            )
        )

    # Only penalise on a trained artifact. An untrained proxy returns 0.5 and
    # available=False, and guessing from that would invent a signal.
    if analysis.quality.available and analysis.quality.score < 0.5:
        delta = W_QUALITY_PENALTY * (0.5 - analysis.quality.score) * 2
        hazard += delta
        contributions.append(
            _contribution(
                "quality_prior_penalty", delta,
                f"Narrative quality prior {analysis.quality.score:.2f} "
                "(story-quality proxy, not retention).",
            )
        )

    debt = max(0, structural.payoff_debt - PAYOFF_DEBT_GRACE)
    if debt:
        delta = W_PAYOFF_DEBT * debt
        hazard += delta
        contributions.append(
            _contribution(
                "payoff_debt", delta,
                f"{structural.payoff_debt} questions open, "
                f"{PAYOFF_DEBT_GRACE} tolerated before debt starts costing.",
            )
        )

    pov_excess = max(0, structural.pov_repeat_streak - POV_STREAK_GRACE)
    if pov_excess:
        lead = structural.speaking_characters[0] if structural.speaking_characters else "one voice"
        delta = W_POV_FATIGUE * pov_excess
        hazard += delta
        contributions.append(
            _contribution(
                "pov_fatigue", delta,
                f"{lead} has led {structural.pov_repeat_streak} consecutive scenes.",
            )
        )

    content_gap = vector.content_overshoot(dna)
    if content_gap > 0:
        delta = W_CONTENT_OVERSHOOT * content_gap
        hazard += delta
        contributions.append(
            _contribution(
                "content_boundary_overshoot", delta,
                f"Violence/gore/dark themes exceed this cohort's stated boundary "
                f"by {content_gap:.2f}.",
            )
        )

    # --- Risk-lowering terms -----------------------------------------------

    alignment = vector.emotional_alignment(dna)
    if alignment > 0.25:
        delta = -W_COHORT_ALIGNMENT * alignment
        hazard += delta
        contributions.append(
            _contribution(
                "cohort_alignment", delta,
                f"Delivers the {vector.emotion_axis.replace('_', ' ')} this cohort came for "
                f"({alignment:.2f}).",
            )
        )

    # A hook only earns its lift when something is actually at stake behind
    # it. A cliffhanger with no stakes is a stylistic tic, not a pull.
    if dna.cliffhanger_strength > 0.35 and dna.stakes_level > 0.3:
        earned = dna.cliffhanger_strength * (0.5 + 0.5 * dna.stakes_level)
        delta = -W_CLIFFHANGER_LIFT * earned
        if is_final_scene:
            delta *= 1.5
        hazard += delta
        contributions.append(
            _contribution(
                "earned_cliffhanger", delta,
                f"Hook {dna.cliffhanger_strength:.2f} backed by stakes {dna.stakes_level:.2f}"
                + (" at the episode end." if is_final_scene else "."),
            )
        )

    progression = 0.0
    if dna.character_development_present:
        progression += 0.5
    if dna.new_information_revealed:
        progression += 0.3
    if dna.mystery_questions_answered > 0:
        progression += 0.2 * min(2, dna.mystery_questions_answered)
    if progression > 0:
        delta = -W_PROGRESSION_LIFT * progression
        hazard += delta
        contributions.append(
            _contribution(
                "meaningful_progression", delta,
                "Character change, new information, or a resolved question.",
            )
        )

    return max(HAZARD_FLOOR, min(HAZARD_CEILING, hazard)), contributions


def build_curve(
    analyses: list[SceneAnalysis],
    cohort: CohortProfile,
) -> CohortCurve:
    """Run the hazard model across an episode and decay survival from 100."""
    vector = to_vector(cohort)
    scenes: list[SceneForecast] = []
    survival = 100.0
    last = len(analyses) - 1

    for position, analysis in enumerate(analyses):
        hazard, contributions = scene_hazard(
            analysis, vector, is_final_scene=(position == last)
        )
        survival = max(SURVIVAL_FLOOR, survival * (1.0 - hazard))
        scenes.append(
            SceneForecast(
                scene_index=analysis.scene.index,
                hazard=round(hazard, 4),
                survival=round(survival, 2),
                contributions=contributions,
            )
        )

    return CohortCurve(
        cohort_label=cohort.label,
        is_counterfactual=cohort.is_counterfactual,
        scenes=scenes,
    )


def risk_band(hazard: float) -> str:
    """Bucket a hazard for display."""
    if hazard >= RISK_HIGH:
        return "high"
    if hazard >= RISK_ELEVATED:
        return "elevated"
    return "low"


def episode_quality_prior(analyses: list[SceneAnalysis]) -> QualityPrior:
    """Average the per-scene quality priors into one episode-level number."""
    available = [a.quality for a in analyses if a.quality.available]
    if not available:
        unavailable = analyses[0].quality if analyses else QualityPrior(available=False)
        return QualityPrior(
            available=False,
            score=0.5,
            message=unavailable.message or "Quality-prior artifact not trained.",
        )

    mean = sum(q.score for q in available) / len(available)
    return QualityPrior(
        available=True,
        score=round(mean, 4),
        message=f"Mean across {len(available)} scene(s).",
        model_version=available[0].model_version,
        top_contributors=available[0].top_contributors,
    )
