"""
Integration tests that really use the network.

These are excluded from the default run because they cost money and time and
fail on a plane. Run them deliberately:

    python -m pytest -m integration                  # everything below
    python -m pytest -m "integration and not openai" # free, local model only

`-m integration` covers three classes of dependency, each skipped
independently when unavailable: the Hugging Face Hub, a locally cached
GoEmotions model, and the OpenAI API.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import make_dna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = PROJECT_ROOT / "samples" / "wexler_street_continuation.txt"

pytestmark = pytest.mark.integration


def _has_transformers() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    from retention_engine.emotion_signals import _load_classifier

    return _load_classifier() is not None


requires_goemotions = pytest.mark.skipif(
    not _has_transformers(), reason="GoEmotions model not installed or not cached"
)
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY", "").strip(), reason="OPENAI_API_KEY not set"
)


@pytest.fixture(scope="module")
def sample_text() -> str:
    return SAMPLE.read_text(encoding="utf-8")


@requires_goemotions
class TestGoEmotionsLive:
    """The real classifier, running on the real sample."""

    def test_classifier_path_is_selected(self, sample_text):
        from retention_engine.emotion_signals import analyse_emotions
        from retention_engine.scene_features import split_scenes

        scenes = split_scenes(sample_text)
        dna = [make_dna() for _ in scenes]

        signals = analyse_emotions([s.text for s in scenes], dna, use_classifier=True)
        assert all(s.source == "goemotions" for s in signals)

    def test_classifier_never_flattens_the_sensor_reading(self, sample_text):
        """
        The regression this guards against.

        GoEmotions returns near-zero on narrative prose. Treating that as
        evidence of calm collapsed the Narrative EKG to a flat line, so the
        classifier is only allowed to raise an axis.
        """
        from retention_engine.emotion_signals import analyse_emotions
        from retention_engine.scene_features import split_scenes

        scenes = split_scenes(sample_text)
        dna = [
            make_dna(tension=0.9, dread=0.8, action_density=0.7)
            for _ in scenes
        ]
        texts = [s.text for s in scenes]

        classified = analyse_emotions(texts, dna, use_classifier=True)
        fallback = analyse_emotions(texts, dna, use_classifier=False)

        for live, base in zip(classified, fallback):
            assert live.arousal >= base.arousal
            assert live.negative_load >= base.negative_load
            assert live.tension >= base.tension

    def test_arousal_retains_scene_to_scene_variation(self, sample_text):
        from retention_engine.emotion_signals import analyse_emotions
        from retention_engine.scene_features import split_scenes

        scenes = split_scenes(sample_text)
        # A quiet records-office scene against a basement confrontation.
        dna = [
            make_dna(tension=0.1, dread=0.0, action_density=0.05, scene_tempo=0.2)
            if i in (2, 3)
            else make_dna(tension=0.9, dread=0.85, action_density=0.6)
            for i in range(len(scenes))
        ]

        signals = analyse_emotions([s.text for s in scenes], dna, use_classifier=True)
        spread = max(s.arousal for s in signals) - min(s.arousal for s in signals)
        assert spread > 0.3, "classifier must not wash out the episode's shape"

    def test_neutral_is_excluded_from_top_emotions(self, sample_text):
        from retention_engine.emotion_signals import analyse_emotions
        from retention_engine.scene_features import split_scenes

        scenes = split_scenes(sample_text)
        signals = analyse_emotions(
            [s.text for s in scenes], [make_dna() for _ in scenes], use_classifier=True
        )
        assert all("neutral" not in s.top_emotions for s in signals)

    def test_source_note_states_the_domain_mismatch(self, sample_text):
        from retention_engine.emotion_signals import analyse_emotions, emotion_source_note
        from retention_engine.scene_features import split_scenes

        scenes = split_scenes(sample_text)
        signals = analyse_emotions(
            [s.text for s in scenes], [make_dna() for _ in scenes], use_classifier=True
        )
        note = emotion_source_note(signals)
        assert "Reddit" in note
        assert "not a retention dataset" in note


class TestHuggingFaceHub:
    """The dataset behind the quality prior is reachable and shaped as expected."""

    def test_quality_prior_dataset_has_the_columns_we_train_on(self):
        datasets = pytest.importorskip("datasets")

        loaded = datasets.load_dataset(
            "lars1234/story_writing_benchmark", "average", split="train[:50]"
        )
        assert "story_text" in loaded.column_names
        assert "overall_score" in loaded.column_names
        assert len(loaded) == 50

    def test_trained_artifact_matches_the_dataset_it_claims(self):
        from retention_engine.quality_proxy import get_quality_proxy

        proxy = get_quality_proxy()
        if not proxy.available:
            pytest.skip("quality prior not trained; run scripts/train_quality_proxy.py")

        meta = proxy.metadata
        assert meta["dataset"] == "lars1234/story_writing_benchmark"
        assert meta["label"] == "overall_score"
        assert meta["r2"] > 0
        assert meta["sample_count"] > 500
        assert "not a retention label" in meta["caveat"].lower()


@requires_openai
class TestOpenAILive:
    """Real structured-output calls. Costs a few cents."""

    def test_scene_dna_sensor_returns_a_valid_reading(self, sample_text):
        from retention_engine.scene_features import measure_scene, split_scenes

        scene = split_scenes(sample_text)[2]  # the records-office exposition dump
        dna = measure_scene(scene, 7, client=_client())

        assert 0.0 <= dna.exposition_ratio <= 1.0
        assert dna.exposition_ratio > 0.4, "the sensor should see a dialogue-heavy info dump"
        assert dna.active_character_count >= 2

    def test_sensor_separates_a_quiet_scene_from_a_tense_one(self, sample_text):
        from retention_engine.scene_features import measure_scene, split_scenes

        scenes = split_scenes(sample_text)
        client = _client()
        opening = measure_scene(scenes[0], 7, client=client)
        records = measure_scene(scenes[2], 7, client=client)

        assert opening.tension > records.tension
        assert records.exposition_ratio > opening.exposition_ratio

    def test_full_forecast_flags_the_deliberate_weak_middle(self, sample_text):
        from retention_engine.engagement_forecast import risk_band
        from retention_engine.orchestrator import run_forecast

        forecast = run_forecast(
            sample_text, use_llm_deep_dive=False, use_emotion_classifier=False
        )

        assert len(forecast.scenes) == 7
        risky = {
            s.scene_index
            for s in forecast.primary_curve.scenes
            if risk_band(s.hazard) != "low"
        }
        assert 3 in risky, "scene 3 is the planted exposition dump"
        assert 0 < forecast.overall_survival <= 100

    def test_counterfactual_cohorts_disagree_on_the_same_episode(self, sample_text):
        from retention_engine.orchestrator import run_forecast

        forecast = run_forecast(
            sample_text, use_llm_deep_dive=False, use_emotion_classifier=False
        )
        finals = [c.final_survival for c in forecast.counterfactual_curves]
        assert max(finals) - min(finals) > 5, "what-if cohorts should separate"

    def test_risk_deep_dive_avoids_causal_and_percentage_claims(self, sample_text):
        from retention_engine.orchestrator import run_forecast

        forecast = run_forecast(
            sample_text, use_llm_deep_dive=True, use_emotion_classifier=False
        )
        assert forecast.risk_explanations

        for explanation in forecast.risk_explanations:
            blob = " ".join(
                [
                    explanation.why_risky,
                    explanation.cohort_expectation,
                    explanation.surgical_fix,
                    explanation.trade_off,
                ]
            ).lower()
            assert "% of listeners" not in blob
            assert "pocket fm" not in blob
            assert "coin" not in blob
            assert explanation.surgical_fix.strip()


@requires_openai
class TestOpenQuestionExtraction:
    """
    That an unanswered question survives extraction at all.

    This cannot be covered offline, because the fake client returns whatever
    the test hands it and would prove only that the plumbing works. The
    behaviour under test is the model's, and it was wrong: across the whole
    six-part Kestrel fixture — a mystery, with a question asked aloud and
    underlined — not one open question was ever recorded, so payoff debt sat
    at zero and the dangling-thread check could never fire.

    Rewording the shared extraction prompt moved the behaviour without ever
    making it dependable, which is why open questions now get a call of their
    own. Asking for constraints and for withheld information in one breath is
    asking for opposite things.
    """

    def test_a_question_asked_and_not_answered_is_recorded(self):
        from projects.facts import extract_open_questions

        part = (
            Path(__file__).resolve().parents[1]
            / "samples" / "kestrel" / "part_02.txt"
        )
        questions = extract_open_questions(
            part.read_text(encoding="utf-8"),
            position=1,
            branch_name="Main timeline",
            client=_client(),
        )
        assert questions, "no open question recorded from a part that asks one outright"
        assert all(f.kind == "open_question" for f in questions)
        assert any("sign" in f.claim.lower() for f in questions), (
            f"found questions, but not the manifest one: {[f.claim for f in questions]}"
        )


def _client():
    from retention_engine.scene_features import build_client

    return build_client()
