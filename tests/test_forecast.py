"""Cohort vectors, the hazard proxy, the Cliffhanger Lab, and before/after."""

from __future__ import annotations

import pytest
from conftest import make_dna, make_scene

from retention_engine.cliffhanger import analyse_ending, classify_ending
from retention_engine.engagement_forecast import (
    HAZARD_CEILING,
    build_curve,
    risk_band,
    scene_hazard,
)
from retention_engine.orchestrator import compare_forecasts
from retention_engine.schemas import (
    CohortProfile,
    EmotionSignals,
    ForecastResult,
    MetricDelta,
    QualityPrior,
    SceneAnalysis,
)
from retention_engine.structural_features import apply_sequence_signals, compute_scene_signals
from retention_engine.target_cohort import DEFAULT_COHORT, counterfactual_cohorts, to_vector


def build_analyses(dna_list, texts=None) -> list[SceneAnalysis]:
    """Assemble SceneAnalysis objects without touching any API."""
    texts = texts or [f"Scene {i} body text goes here." for i in range(1, len(dna_list) + 1)]
    scenes = [make_scene(i, text=t) for i, t in enumerate(texts, 1)]
    signals = apply_sequence_signals([compute_scene_signals(s) for s in scenes], dna_list)

    return [
        SceneAnalysis(
            scene=scene,
            dna=dna,
            structural=signal,
            emotion=EmotionSignals(
                source="scene_dna_fallback",
                arousal=dna.emotional_intensity,
                warmth=dna.warmth,
                negative_load=dna.dread,
                tension=dna.tension,
            ),
            quality=QualityPrior(available=False, score=0.5, message="test"),
        )
        for scene, dna, signal in zip(scenes, dna_list, signals)
    ]


def build_forecast(dna_list, cohort=None, texts=None) -> ForecastResult:
    analyses = build_analyses(dna_list, texts)
    target = cohort or DEFAULT_COHORT
    return ForecastResult(
        scenes=analyses,
        primary_curve=build_curve(analyses, target),
        counterfactual_curves=[build_curve(analyses, c) for c in counterfactual_cohorts(target)],
        cliffhanger=analyse_ending(analyses),
        risk_explanations=[],
        quality_prior=QualityPrior(available=False),
        cohort=target,
    )


class TestCohortVectors:
    def test_pace_preference_moves_the_target(self):
        slow = to_vector(DEFAULT_COHORT.model_copy(update={"pace_preference": "slow"}))
        fast = to_vector(DEFAULT_COHORT.model_copy(update={"pace_preference": "fast"}))
        assert slow.pace_target < fast.pace_target

    def test_fast_cohorts_tolerate_less_exposition(self):
        slow = to_vector(DEFAULT_COHORT.model_copy(update={"pace_preference": "slow"}))
        fast = to_vector(DEFAULT_COHORT.model_copy(update={"pace_preference": "fast"}))
        assert fast.exposition_tolerance < slow.exposition_tolerance

    def test_complexity_overshoot_only_counts_above_the_ceiling(self):
        vector = to_vector(DEFAULT_COHORT.model_copy(update={"complexity_tolerance": "low"}))
        assert vector.complexity_overshoot(make_dna(complexity=0.2)) == 0.0
        assert vector.complexity_overshoot(make_dna(complexity=0.9)) > 0.0

    def test_emotional_alignment_reads_the_preferred_axis(self):
        dread_cohort = to_vector(DEFAULT_COHORT.model_copy(update={"emotional_preference": "dread"}))
        romance_cohort = to_vector(DEFAULT_COHORT.model_copy(update={"emotional_preference": "romance"}))
        dna = make_dna(dread=0.9, romance=0.05)
        assert dread_cohort.emotional_alignment(dna) == 0.9
        assert romance_cohort.emotional_alignment(dna) == 0.05

    def test_content_overshoot_uses_the_worst_axis(self):
        vector = to_vector(DEFAULT_COHORT.model_copy(update={"content_boundary": "low"}))
        assert vector.content_overshoot(make_dna(gore_level=0.9, violence_intensity=0.0)) > 0

    def test_three_labelled_counterfactuals_are_produced(self):
        alternatives = counterfactual_cohorts(DEFAULT_COHORT)
        assert len(alternatives) == 3
        assert all(c.is_counterfactual for c in alternatives)
        assert all(c.label.startswith("What-if") for c in alternatives)


class TestHazardProxy:
    def test_hazard_is_deterministic(self):
        analyses = build_analyses([make_dna()])
        vector = to_vector(DEFAULT_COHORT)
        first, _ = scene_hazard(analyses[0], vector)
        second, _ = scene_hazard(analyses[0], vector)
        assert first == second

    def test_exposition_dump_scores_riskier_than_an_active_scene(self):
        vector = to_vector(DEFAULT_COHORT)
        dump = build_analyses(
            [make_dna(exposition_ratio=0.85, action_density=0.0, scene_tempo=0.15,
                      tension=0.05, dread=0.0, conflict_present=False, stakes_level=0.0)]
        )[0]
        active = build_analyses(
            [make_dna(exposition_ratio=0.05, action_density=0.8, scene_tempo=0.8,
                      tension=0.8, dread=0.7, conflict_present=True, stakes_level=0.8)]
        )[0]

        assert scene_hazard(dump, vector)[0] > scene_hazard(active, vector)[0]

    def test_every_firing_term_is_reported(self):
        analysis = build_analyses(
            [make_dna(exposition_ratio=0.9, complexity=0.95, conflict_present=False,
                      action_density=0.0, scene_tempo=0.1, tension=0.0, dread=0.0,
                      romance=0.0, anger=0.0, sadness=0.0, wonder=0.0, stakes_level=0.0)]
        )[0]
        _, contributions = scene_hazard(analysis, to_vector(DEFAULT_COHORT))

        factors = {c.factor for c in contributions}
        assert "exposition_overshoot" in factors
        assert "complexity_overshoot" in factors
        assert "low_event_movement" in factors
        assert "emotional_flatness" in factors
        assert all(c.detail for c in contributions)

    def test_untrained_quality_prior_adds_no_term(self):
        analysis = build_analyses([make_dna()])[0]
        assert analysis.quality.available is False
        _, contributions = scene_hazard(analysis, to_vector(DEFAULT_COHORT))
        assert not any(c.factor == "quality_prior_penalty" for c in contributions)

    def test_trained_low_quality_adds_a_penalty(self):
        analysis = build_analyses([make_dna()])[0]
        analysis.quality = QualityPrior(available=True, score=0.1, message="trained")
        _, contributions = scene_hazard(analysis, to_vector(DEFAULT_COHORT))
        assert any(c.factor == "quality_prior_penalty" for c in contributions)

    def test_unearned_cliffhanger_gets_no_lift(self):
        analysis = build_analyses([make_dna(cliffhanger_strength=0.9, stakes_level=0.0)])[0]
        _, contributions = scene_hazard(analysis, to_vector(DEFAULT_COHORT))
        assert not any(c.factor == "earned_cliffhanger" for c in contributions)

    def test_earned_cliffhanger_lowers_hazard(self):
        analysis = build_analyses([make_dna(cliffhanger_strength=0.9, stakes_level=0.9)])[0]
        _, contributions = scene_hazard(analysis, to_vector(DEFAULT_COHORT))
        lift = next(c for c in contributions if c.factor == "earned_cliffhanger")
        assert lift.delta < 0

    def test_hazard_is_clamped(self):
        catastrophic = build_analyses(
            [make_dna(exposition_ratio=1.0, complexity=1.0, conflict_present=False,
                      action_density=0.0, scene_tempo=0.0, tension=0.0, dread=0.0,
                      romance=0.0, anger=0.0, sadness=0.0, wonder=0.0, stakes_level=0.0,
                      gore_level=1.0, violence_intensity=1.0)]
        )[0]
        hazard, _ = scene_hazard(catastrophic, to_vector(DEFAULT_COHORT))
        assert hazard <= HAZARD_CEILING

    def test_risk_bands(self):
        assert risk_band(0.05) == "low"
        assert risk_band(0.18) == "elevated"
        assert risk_band(0.35) == "high"


class TestSurvivalCurve:
    def test_curve_starts_at_100_and_decays(self):
        curve = build_curve(build_analyses([make_dna()] * 4), DEFAULT_COHORT)
        values = curve.curve
        assert values[0] == 100.0
        assert all(later <= earlier for earlier, later in zip(values, values[1:]))

    def test_survival_never_goes_negative(self):
        weak = make_dna(exposition_ratio=1.0, conflict_present=False, action_density=0.0,
                        scene_tempo=0.0, tension=0.0, dread=0.0, romance=0.0, anger=0.0,
                        sadness=0.0, wonder=0.0, stakes_level=0.0, complexity=1.0)
        curve = build_curve(build_analyses([weak] * 20), DEFAULT_COHORT)
        assert curve.final_survival > 0

    def test_cohorts_disagree_about_the_same_episode(self):
        exposition_heavy = [
            make_dna(exposition_ratio=0.8, scene_tempo=0.2, action_density=0.1)
        ] * 3
        analyses = build_analyses(exposition_heavy)

        fast = build_curve(
            analyses, DEFAULT_COHORT.model_copy(update={"pace_preference": "fast"})
        )
        slow = build_curve(
            analyses, DEFAULT_COHORT.model_copy(update={"pace_preference": "slow"})
        )
        assert fast.final_survival < slow.final_survival


class TestCliffhangerLab:
    def test_resolved_ending_is_classified_weak(self):
        analyses = build_analyses([make_dna(cliffhanger_strength=0.05)])
        assert classify_ending(analyses[-1]) == ["weak_resolved"]

    def test_disappearance_marker_is_detected(self):
        analyses = build_analyses(
            [make_dna(cliffhanger_strength=0.8)],
            texts=["They opened the door. When they looked inside, Aanya was gone."],
        )
        assert "disappearance" in classify_ending(analyses[-1])

    def test_countdown_marker_is_detected(self):
        analyses = build_analyses(
            [make_dna(cliffhanger_strength=0.8)],
            texts=["He set the phone down. You have twenty-four hours, she said."],
        )
        assert "countdown" in classify_ending(analyses[-1])

    def test_false_resolution_needs_a_reversal_to_count(self):
        text = "She smiled in relief. It was over."
        calm = build_analyses([make_dna(cliffhanger_strength=0.6, surprise_factor=0.1)], [text])
        twist = build_analyses([make_dna(cliffhanger_strength=0.6, surprise_factor=0.9)], [text])

        assert "false_resolution" not in classify_ending(calm[-1])
        assert "false_resolution" in classify_ending(twist[-1])

    def test_strong_ending_beats_weak_ending_on_unlock_pull(self):
        weak = build_forecast(
            [make_dna(), make_dna(cliffhanger_strength=0.05, stakes_level=0.1,
                                  surprise_factor=0.05, mystery_questions_opened=0,
                                  mystery_questions_answered=2)]
        )
        strong = build_forecast(
            [make_dna(), make_dna(cliffhanger_strength=0.95, stakes_level=0.95,
                                  surprise_factor=0.85, mystery_questions_opened=3,
                                  character_development_present=True)]
        )
        assert strong.cliffhanger.unlock_pull_index > weak.cliffhanger.unlock_pull_index

    def test_payoff_debt_drags_the_index_down(self):
        ending = make_dna(cliffhanger_strength=0.9, stakes_level=0.8, surprise_factor=0.7)
        clean = build_forecast([make_dna(mystery_questions_opened=0), ending])
        indebted = build_forecast(
            [make_dna(mystery_questions_opened=9, mystery_questions_answered=0), ending]
        )

        assert indebted.cliffhanger.payoff_debt_penalty > clean.cliffhanger.payoff_debt_penalty
        assert indebted.cliffhanger.unlock_pull_index < clean.cliffhanger.unlock_pull_index

    def test_index_and_hook_stay_in_range(self):
        forecast = build_forecast([make_dna()] * 3)
        assert 0 <= forecast.cliffhanger.unlock_pull_index <= 100
        assert 0 <= forecast.cliffhanger.hook_strength <= 100

    def test_recommendation_never_claims_conversion(self):
        for dna in (make_dna(cliffhanger_strength=0.05), make_dna(cliffhanger_strength=0.95)):
            text = build_forecast([make_dna(), dna]).cliffhanger.recommendation.lower()
            assert "conversion" not in text
            assert "%" not in text
            assert "coin" not in text

    def test_empty_analyses_rejected(self):
        with pytest.raises(ValueError):
            analyse_ending([])


class TestBeforeAfterComparison:
    def test_improvement_is_detected_and_labelled(self):
        weak = [make_dna(exposition_ratio=0.9, conflict_present=False, action_density=0.0,
                         scene_tempo=0.1, tension=0.0, dread=0.0)] * 3
        strong = [make_dna(exposition_ratio=0.1, conflict_present=True, action_density=0.7,
                           scene_tempo=0.6, tension=0.8, dread=0.7)] * 3

        result = compare_forecasts(build_forecast(weak), build_forecast(strong))

        assert result.metrics[0].improved
        assert "+" in result.headline
        assert "improvement" in result.headline.lower()

    def test_regression_is_reported_honestly(self):
        weak = [make_dna(exposition_ratio=0.9, conflict_present=False, action_density=0.0,
                         scene_tempo=0.1, tension=0.0, dread=0.0)] * 3
        strong = [make_dna(exposition_ratio=0.1, conflict_present=True, action_density=0.7,
                           scene_tempo=0.6, tension=0.8, dread=0.7)] * 3

        result = compare_forecasts(build_forecast(strong), build_forecast(weak))
        assert not result.metrics[0].improved
        assert "regression" in result.headline.lower()

    def test_identical_drafts_report_no_measurable_change(self):
        """
        Re-running an unchanged episode must not be sold as an improvement.

        The Scene DNA sensor is not reproducible at temperature 0, so a real
        re-run drifts a few points on its own. The card has to attribute that
        to measurement rather than to the writer's edit.
        """
        dna = [make_dna()] * 3
        result = compare_forecasts(build_forecast(dna), build_forecast(dna))

        survival = result.metrics[0]
        assert survival.delta == 0
        assert not survival.significant
        assert not survival.improved
        assert "no measurable change" in result.headline.lower()

    def test_movement_inside_the_noise_band_is_not_called_an_improvement(self):
        drifted = MetricDelta(
            label="Unlock Pull Index", before=40.0, after=46.0, noise_band=8.5
        )
        assert drifted.delta == pytest.approx(6.0)
        assert not drifted.significant
        assert not drifted.improved, "a 6-point move inside an 8.5-point band proves nothing"

    def test_movement_beyond_the_noise_band_counts(self):
        real = MetricDelta(
            label="Unlock Pull Index", before=40.0, after=60.0, noise_band=8.5
        )
        assert real.significant
        assert real.improved

    def test_unlock_pull_index_carries_a_noise_band(self):
        result = compare_forecasts(
            build_forecast([make_dna()] * 2), build_forecast([make_dna()] * 2)
        )
        upi = next(m for m in result.metrics if m.label == "Unlock Pull Index")
        assert upi.noise_band > 0, "UPI drifts 8+ points between identical runs"

    def test_counterfactual_cohorts_appear_in_the_metrics(self):
        result = compare_forecasts(build_forecast([make_dna()] * 2), build_forecast([make_dna()] * 2))
        labels = [m.label for m in result.metrics]
        assert any(label.startswith("What-if") for label in labels)

    def test_exposition_improvement_is_named_as_evidence(self):
        before = build_forecast([make_dna(exposition_ratio=0.9)] * 3)
        after = build_forecast([make_dna(exposition_ratio=0.1)] * 3)

        result = compare_forecasts(before, after)
        assert any("exposition fatigue" in line for line in result.primary_evidence)

    def test_risk_movement_is_reported_per_scene(self):
        before = build_forecast([make_dna(exposition_ratio=0.95, conflict_present=False,
                                          action_density=0.0, scene_tempo=0.1)] * 2)
        after = build_forecast([make_dna(exposition_ratio=0.05, conflict_present=True,
                                         action_density=0.8, scene_tempo=0.6)] * 2)

        movement = compare_forecasts(before, after).risk_scene_movement
        assert any("fell" in line for line in movement)

    def test_unchanged_draft_reports_no_movement(self):
        forecast = build_forecast([make_dna()] * 3)
        result = compare_forecasts(forecast, forecast)
        assert "unchanged" in result.headline.lower()

    def test_disclaimer_travels_with_the_comparison(self):
        result = compare_forecasts(build_forecast([make_dna()]), build_forecast([make_dna()]))
        assert "not calibrated with live listener" in result.disclaimer
