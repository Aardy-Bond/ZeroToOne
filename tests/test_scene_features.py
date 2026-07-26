"""Scene segmentation and strict Scene DNA parsing."""

from __future__ import annotations

import pytest
from conftest import FakeOpenAI, make_dna, make_scene
from openai import OpenAIError

from retention_engine.scene_features import (
    SceneFeatureError,
    measure_scene,
    measure_scenes,
    split_scenes,
)
from retention_engine.schemas import SceneDNA


class TestSplitScenes:
    def test_splits_on_slug_lines(self, screenplay):
        scenes = split_scenes(screenplay)
        assert len(scenes) == 3
        assert [s.index for s in scenes] == [1, 2, 3]
        assert scenes[0].heading.startswith("INT. BASEMENT")
        assert all(not s.inferred for s in scenes)

    def test_scene_text_includes_its_heading(self, screenplay):
        scenes = split_scenes(screenplay)
        assert scenes[1].text.startswith("INT. KITCHEN")
        # A scene must not leak the next scene's content.
        assert "HALLWAY" not in scenes[1].text

    def test_prose_without_headings_is_inferred(self, prose):
        scenes = split_scenes(prose)
        assert len(scenes) >= 2
        assert all(s.inferred for s in scenes)
        assert all(s.heading.startswith("INFERRED SEGMENT") for s in scenes)

    def test_prose_split_never_loses_words(self, prose):
        scenes = split_scenes(prose)
        recombined = sum(s.word_count for s in scenes)
        assert recombined == len(prose.split())

    def test_short_tail_folds_into_previous_scene(self):
        long_paragraph = "word " * 240
        text = long_paragraph + "\n\nA short closing line."
        scenes = split_scenes(text)
        assert len(scenes) == 1
        assert "short closing line" in scenes[-1].text

    def test_substantial_cold_open_becomes_its_own_scene(self):
        cold_open = "A voice in the dark says the thing nobody wants to hear. " * 12
        text = f"{cold_open}\n\nINT. STUDIO - DAY\n\nThe engineer rewinds the tape.\n"

        scenes = split_scenes(text)
        assert scenes[0].heading == "COLD OPEN"
        assert "voice in the dark" in scenes[0].text

    def test_short_preamble_is_folded_in_rather_than_dropped(self):
        """Text before the first slug line must never leave the analysis."""
        text = (
            "A voice in the dark says the thing nobody wants to hear.\n\n"
            "INT. STUDIO - DAY\n\nThe engineer rewinds the tape.\n"
        )
        scenes = split_scenes(text)

        assert len(scenes) == 1
        assert "voice in the dark" in scenes[0].text
        assert sum(s.word_count for s in scenes) == len(text.split())

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            split_scenes("   ")

    def test_single_heading_over_a_very_long_body_is_subdivided(self):
        body = "\n\n".join(["Something happens here in this paragraph. " * 12] * 10)
        scenes = split_scenes(f"INT. ROOM - NIGHT\n\n{body}")
        assert len(scenes) >= 2


class TestSceneDNAValidation:
    def test_out_of_range_values_are_clamped_not_rejected(self):
        dna = make_dna(tension=1.8, exposition_ratio=-0.4)
        assert dna.tension == 1.0
        assert dna.exposition_ratio == 0.0

    def test_negative_counts_are_clamped(self):
        dna = make_dna(mystery_questions_opened=-3, active_character_count=-1)
        assert dna.mystery_questions_opened == 0
        assert dna.active_character_count == 0

    def test_extra_keys_are_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SceneDNA(**{**make_dna().model_dump(), "verdict": "this scene is bad"})

    def test_invalid_enum_is_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_dna(genre_alignment="documentary")

    def test_emotional_intensity_is_the_peak_axis(self):
        dna = make_dna(tension=0.2, dread=0.85, anger=0.3)
        assert dna.emotional_intensity == 0.85

    def test_event_movement_combines_action_tempo_and_stakes(self):
        quiet = make_dna(action_density=0.0, scene_tempo=0.1, stakes_level=0.0)
        loud = make_dna(action_density=0.9, scene_tempo=0.9, stakes_level=0.9)
        assert quiet.event_movement < loud.event_movement


class TestMeasureScenes:
    def test_measures_every_scene_and_preserves_order(self):
        scenes = [make_scene(i, f"Scene {i} body text.") for i in (1, 2, 3)]
        expected = [make_dna(tension=t) for t in (0.1, 0.5, 0.9)]
        client = FakeOpenAI(expected)

        result = measure_scenes(scenes, client=client)

        assert [d.tension for d in result] == [0.1, 0.5, 0.9]
        assert len(client.calls) == 3

    def test_sends_calibration_anchors_to_the_sensor(self):
        client = FakeOpenAI([make_dna()])
        measure_scene(make_scene(), 1, client=client)

        system = client.calls[0]["messages"][0]["content"]
        assert "CALIBRATION ANCHORS" in system
        assert "defusing a bomb with thirty seconds remaining" in system
        # The sensor must not be told to form opinions.
        assert "do not review" in system.lower()

    def test_uses_zero_temperature_for_repeatability(self):
        client = FakeOpenAI([make_dna()])
        measure_scene(make_scene(), 1, client=client)
        assert client.calls[0]["temperature"] == 0.0

    def test_one_failed_scene_fails_the_batch(self):
        scenes = [make_scene(1), make_scene(2)]
        client = FakeOpenAI([make_dna(), OpenAIError("rate limited")])

        with pytest.raises(SceneFeatureError, match="scene 2"):
            measure_scenes(scenes, client=client)

    def test_refusal_is_surfaced(self):
        from conftest import FakeParseResponse

        class Refusing:
            chat = None

            def __init__(self):
                self.chat = type(
                    "C", (), {"completions": type(
                        "P", (), {"parse": lambda _self, **kw: FakeParseResponse(None, "refused")}
                    )()}
                )()

        with pytest.raises(SceneFeatureError, match="refused"):
            measure_scene(make_scene(), 1, client=Refusing())

    def test_empty_scene_list_rejected(self):
        with pytest.raises(ValueError):
            measure_scenes([], client=FakeOpenAI([]))
