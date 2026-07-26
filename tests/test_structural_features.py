"""Deterministic structural signals — no model in the loop."""

from __future__ import annotations

from conftest import make_dna, make_scene

from retention_engine.structural_features import (
    apply_sequence_signals,
    compute_scene_signals,
    describe_evidence,
    extract_speakers,
)


class TestPerSceneSignals:
    def test_counts_words_and_sentences(self):
        scene = make_scene(text="One thing happened. Then another thing happened too.")
        signals = compute_scene_signals(scene)
        assert signals.word_count == 8
        assert signals.sentence_count == 2

    def test_extracts_screenplay_speakers(self):
        text = (
            "INT. ROOM - NIGHT\n\nShe waits.\n\n"
            "PRIYA\nHello?\n\nDEV (CONT'D)\nI'm here.\n"
        )
        assert extract_speakers(text) == ["PRIYA", "DEV"]

    def test_parenthetical_suffixes_collapse_to_one_character(self):
        text = "PRIYA\nOne.\n\nPRIYA (CONT'D)\nTwo.\n\nPRIYA (V.O.)\nThree.\n"
        signals = compute_scene_signals(make_scene(text=text))
        assert signals.speaking_characters == ["PRIYA"]
        assert signals.pov_dominance == 1.0

    def test_scene_headings_are_not_mistaken_for_speakers(self):
        text = "INT. BASEMENT - NIGHT\n\nDust settles.\n\nDEV\nHello.\n"
        assert extract_speakers(text) == ["DEV"]

    def test_urgency_punctuation_density(self):
        calm = compute_scene_signals(make_scene(text="He walked home slowly and then slept."))
        urgent = compute_scene_signals(make_scene(text="Run! Now! Why? Move--"))
        assert urgent.urgency_punctuation_density > calm.urgency_punctuation_density

    def test_time_pressure_terms_are_counted(self):
        signals = compute_scene_signals(
            make_scene(text="You have twelve hours. After midnight the deadline passes.")
        )
        assert signals.time_pressure_hits >= 3

    def test_hook_marker_detected_at_scene_end(self):
        hooked = compute_scene_signals(
            make_scene(text="She turned around slowly.\n\nHe was already behind her.")
        )
        settled = compute_scene_signals(
            make_scene(text="She turned around.\n\nThe room was empty and she felt fine.")
        )
        assert hooked.hook_marker_present
        assert not settled.hook_marker_present

    def test_long_sentences_raise_mean_length(self):
        short = compute_scene_signals(make_scene(text="He ran. She hid. It waited."))
        long = compute_scene_signals(
            make_scene(
                text=(
                    "The deed changed hands eleven times before the consortium that "
                    "held it through the twenties dissolved in 1931 following a "
                    "dispute over the eastern easement surveyed incorrectly in 1908."
                )
            )
        )
        assert long.mean_sentence_length > short.mean_sentence_length * 3


class TestSequenceSignals:
    def test_exposition_streak_accumulates_and_resets(self):
        scenes = [make_scene(i) for i in range(1, 5)]
        dna = [
            make_dna(exposition_ratio=0.7),
            make_dna(exposition_ratio=0.6),
            make_dna(exposition_ratio=0.1),
            make_dna(exposition_ratio=0.8),
        ]
        signals = apply_sequence_signals([compute_scene_signals(s) for s in scenes], dna)

        assert [s.consecutive_exposition_scenes for s in signals] == [1, 2, 0, 1]

    def test_payoff_debt_accumulates_across_scenes(self):
        scenes = [make_scene(i) for i in range(1, 4)]
        dna = [
            make_dna(mystery_questions_opened=2, mystery_questions_answered=0),
            make_dna(mystery_questions_opened=2, mystery_questions_answered=0),
            make_dna(mystery_questions_opened=0, mystery_questions_answered=1),
        ]
        signals = apply_sequence_signals([compute_scene_signals(s) for s in scenes], dna)

        assert [s.payoff_debt for s in signals] == [2, 4, 3]

    def test_payoff_debt_never_goes_negative(self):
        scenes = [make_scene(1)]
        dna = [make_dna(mystery_questions_opened=0, mystery_questions_answered=5)]
        signals = apply_sequence_signals([compute_scene_signals(s) for s in scenes], dna)
        assert signals[0].payoff_debt == 0

    def test_low_conflict_streak_tracks_consecutive_quiet_scenes(self):
        scenes = [make_scene(i) for i in range(1, 4)]
        dna = [
            make_dna(conflict_present=False, action_density=0.0, scene_tempo=0.1, stakes_level=0.0),
            make_dna(conflict_present=False, action_density=0.0, scene_tempo=0.1, stakes_level=0.0),
            make_dna(conflict_present=True, action_density=0.8, scene_tempo=0.8, stakes_level=0.8),
        ]
        signals = apply_sequence_signals([compute_scene_signals(s) for s in scenes], dna)
        assert [s.consecutive_low_conflict_scenes for s in signals] == [1, 2, 0]

    def test_pov_streak_counts_same_lead_across_scenes(self):
        text = "PRIYA\nStill me.\n"
        scenes = [make_scene(i, text=text) for i in range(1, 4)]
        signals = apply_sequence_signals(
            [compute_scene_signals(s) for s in scenes], [make_dna() for _ in scenes]
        )
        assert [s.pov_repeat_streak for s in signals] == [1, 2, 3]

    def test_pov_streak_resets_on_a_new_lead(self):
        scenes = [
            make_scene(1, text="PRIYA\nOne.\n"),
            make_scene(2, text="PRIYA\nTwo.\n"),
            make_scene(3, text="DEV\nThree.\n"),
        ]
        signals = apply_sequence_signals(
            [compute_scene_signals(s) for s in scenes], [make_dna() for _ in scenes]
        )
        assert [s.pov_repeat_streak for s in signals] == [1, 2, 1]

    def test_mismatched_lengths_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            apply_sequence_signals([compute_scene_signals(make_scene())], [])


class TestEvidenceDescriptions:
    def test_clean_scene_produces_no_warnings(self):
        signal = compute_scene_signals(
            make_scene(text="She moved fast. He followed. The door slammed behind them.")
        )
        dna = make_dna(
            exposition_ratio=0.1, conflict_present=True, complexity=0.3,
            tension=0.8, cliffhanger_strength=0.6,
        )
        assert describe_evidence(signal, dna) == []

    def test_exposition_fatigue_is_quoted_with_numbers(self):
        signal = compute_scene_signals(make_scene(text="He explained the history at length."))
        signal.consecutive_exposition_scenes = 3
        notes = describe_evidence(signal, make_dna(exposition_ratio=0.7))
        assert any("3 consecutive scenes" in n for n in notes)
        assert any("70%" in n for n in notes)

    def test_missing_conflict_is_named(self):
        signal = compute_scene_signals(make_scene(text="They sat quietly for a while."))
        notes = describe_evidence(signal, make_dna(conflict_present=False))
        assert any("No active conflict" in n for n in notes)
