"""Directing sheet generation and audio manifest compatibility."""

from __future__ import annotations

import json

import pytest
from conftest import FakeOpenAI, FakeSpeech, make_dna
from test_forecast import build_forecast

from audio_engine.synthesizer import (
    DIRECTED_TTS_MODEL,
    TTS_MODEL,
    AudioSynthesizer,
    build_directed_cue_sheet,
    lay_out_timeline,
    parse_script,
)
from retention_engine.directing_sheet import (
    assign_scene_indices,
    build_directing_sheet,
    select_preview_scenes,
)
from writers_room.schemas import FoleyTrigger

PREVIEW_SCRIPT = (
    "INT. BASEMENT - NIGHT\n\n"
    "The bulb dies.\n\n"
    "PRIYA\nTwo.\n\n"
    "NARRATOR (V.O.)\nSomething breathes. It is not a child.\n\n"
    "DEV\nPriya. What did you find down here.\n"
)


def make_synth(tmp_path, speech: FakeSpeech | None = None) -> AudioSynthesizer:
    client = FakeOpenAI([])
    client.audio.speech = speech or FakeSpeech()
    return AudioSynthesizer(output_dir=tmp_path, client=client)


class TestDirectingSheet:
    def test_one_entry_per_chunk(self):
        forecast = build_forecast([make_dna()] * 2)
        chunks = parse_script(PREVIEW_SCRIPT)
        indices = [1] * len(chunks)

        sheet = build_directing_sheet(chunks, forecast, indices)
        assert len(sheet) == len(chunks)
        assert [e.chunk_index for e in sheet] == [c.index for c in chunks]

    def test_carries_every_required_field(self):
        forecast = build_forecast([make_dna()] * 2)
        chunks = parse_script(PREVIEW_SCRIPT)
        entry = build_directing_sheet(chunks, forecast, [1] * len(chunks))[0]

        payload = entry.model_dump()
        for key in (
            "narrative_role", "engagement_risk", "dominant_emotion", "delivery_tempo",
            "target_speed", "pause_before_reveal_ms", "instruction", "foley",
            "hook_role", "survival_proxy", "unlock_pull_index",
        ):
            assert key in payload

    def test_dread_reads_slower_than_action(self):
        chunks = parse_script(PREVIEW_SCRIPT)

        dread = build_forecast([make_dna(dread=0.95, action_density=0.0, scene_tempo=0.2)])
        action = build_forecast([make_dna(dread=0.0, action_density=0.95, scene_tempo=0.9)])

        slow = build_directing_sheet(chunks, dread, [1] * len(chunks))[0]
        fast = build_directing_sheet(chunks, action, [1] * len(chunks))[0]
        assert slow.target_speed < fast.target_speed

    def test_speed_stays_inside_a_performable_band(self):
        extremes = [
            make_dna(action_density=1.0, scene_tempo=1.0, tension=1.0, dread=0.0),
            make_dna(action_density=0.0, scene_tempo=0.0, tension=0.0, dread=1.0,
                     exposition_ratio=1.0),
        ]
        chunks = parse_script(PREVIEW_SCRIPT)
        for dna in extremes:
            entry = build_directing_sheet(chunks, build_forecast([dna]), [1] * len(chunks))[0]
            assert 0.88 <= entry.target_speed <= 1.15

    def test_final_scene_is_marked_as_the_episode_hook(self):
        forecast = build_forecast(
            [make_dna(), make_dna(cliffhanger_strength=0.9, stakes_level=0.9)]
        )
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [2] * len(chunks))

        assert sheet[0].narrative_role == "episode_end_hook"
        assert sheet[0].hook_role
        assert sheet[0].pause_before_reveal_ms >= 750

    def test_resolved_ending_is_not_called_a_hook(self):
        forecast = build_forecast(
            [make_dna(), make_dna(cliffhanger_strength=0.02, mystery_questions_answered=3)]
        )
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [2] * len(chunks))
        assert sheet[0].narrative_role == "episode_end_resolution"

    def test_risk_band_reaches_the_instruction(self):
        risky = build_forecast(
            [make_dna(exposition_ratio=0.95, conflict_present=False, action_density=0.0,
                      scene_tempo=0.1, tension=0.0, dread=0.0, romance=0.0, anger=0.0,
                      sadness=0.0, wonder=0.0, stakes_level=0.0, complexity=0.95)]
        )
        chunks = parse_script(PREVIEW_SCRIPT)
        entry = build_directing_sheet(chunks, risky, [1] * len(chunks))[0]

        assert entry.engagement_risk in ("elevated", "high")
        assert "risk" in entry.instruction.lower()

    def test_narration_chunks_get_narrator_direction(self):
        forecast = build_forecast([make_dna()])
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [1] * len(chunks))

        narrator = next(
            e for e, c in zip(sheet, chunks) if c.kind == "narration"
        )
        assert "narrator voice" in narrator.instruction.lower()

    def test_mismatched_index_length_rejected(self):
        forecast = build_forecast([make_dna()])
        chunks = parse_script(PREVIEW_SCRIPT)
        with pytest.raises(ValueError):
            build_directing_sheet(chunks, forecast, [1])

    def test_scene_assignment_maps_chunks_to_their_scene(self, screenplay):
        from retention_engine.scene_features import split_scenes

        scenes = split_scenes(screenplay)
        chunks = parse_script(screenplay)
        indices = assign_scene_indices(chunks, scenes, parse_script)

        assert len(indices) == len(chunks)
        assert indices == sorted(indices)
        assert set(indices) <= {s.index for s in scenes}


class TestPreviewSelection:
    def test_preview_prefers_the_ending(self):
        forecast = build_forecast([make_dna()] * 6)
        selected = select_preview_scenes(forecast, target_seconds=90)
        assert selected[-1].scene.index == 6

    def test_preview_respects_the_time_budget(self):
        long_text = " ".join(["word"] * 400)
        forecast = build_forecast([make_dna()] * 5, texts=[long_text] * 5)
        selected = select_preview_scenes(forecast, target_seconds=90, words_per_second=2.5)
        assert len(selected) < 5

    def test_budget_counts_spoken_words_not_action_lines(self):
        """Action lines are never voiced, so budgeting on them under-fills the preview."""
        scene_text = (
            "INT. ROOM - NIGHT\n\n"
            + " ".join(["action"] * 120)
            + "\n\nDEV\n"
            + " ".join(["spoken"] * 30)
            + "\n"
        )
        forecast = build_forecast([make_dna()] * 6, texts=[scene_text] * 6)

        by_total = select_preview_scenes(forecast)
        by_spoken = select_preview_scenes(
            forecast,
            spoken_word_count=lambda s: sum(c.word_count for c in parse_script(s.text)),
        )

        assert len(by_spoken) > len(by_total)

    def test_preview_reaches_the_minimum_runtime_when_material_allows(self):
        scene_text = "INT. ROOM - NIGHT\n\nDEV\n" + " ".join(["spoken"] * 40) + "\n"
        forecast = build_forecast([make_dna()] * 8, texts=[scene_text] * 8)

        selected = select_preview_scenes(
            forecast,
            spoken_word_count=lambda s: sum(c.word_count for c in parse_script(s.text)),
        )
        chunks = parse_script("\n\n".join(a.scene.text for a in selected))
        assert 60 <= lay_out_timeline(chunks) <= 90

    def test_minimum_runtime_wins_over_the_ceiling(self):
        """Scenes are indivisible; running slightly long beats cutting the setup."""
        scene_text = "INT. ROOM - NIGHT\n\nDEV\n" + " ".join(["spoken"] * 110) + "\n"
        forecast = build_forecast([make_dna()] * 4, texts=[scene_text] * 4)

        selected = select_preview_scenes(
            forecast,
            spoken_word_count=lambda s: sum(c.word_count for c in parse_script(s.text)),
        )
        chunks = parse_script("\n\n".join(a.scene.text for a in selected))
        assert lay_out_timeline(chunks) >= 60

    def test_scenes_stay_in_order(self):
        forecast = build_forecast([make_dna()] * 4)
        selected = select_preview_scenes(forecast)
        assert [a.scene.index for a in selected] == sorted(a.scene.index for a in selected)


class TestDirectedSynthesis:
    def test_preview_manifest_embeds_the_directing_sheet(self, tmp_path):
        forecast = build_forecast([make_dna()] * 2)
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [2] * len(chunks))

        manifest = make_synth(tmp_path).synthesize_directed_preview(
            PREVIEW_SCRIPT, [], sheet, generate_audio=False
        )

        assert manifest["export_type"] == "retention_directed_preview"
        assert len(manifest["retention_directing_sheet"]) == len(chunks)
        assert manifest["retention_directing_sheet"][0]["narrative_role"]

    def test_instructions_and_speed_reach_the_tts_call(self, tmp_path):
        speech = FakeSpeech()
        forecast = build_forecast([make_dna()] * 2)
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [2] * len(chunks))

        make_synth(tmp_path, speech).synthesize_directed_preview(
            PREVIEW_SCRIPT, [], sheet, generate_audio=True
        )

        assert speech.calls
        first = speech.calls[0]
        assert first["model"] == DIRECTED_TTS_MODEL
        assert first["instructions"]
        assert 0.88 <= first["speed"] <= 1.15

    def test_rejected_instructions_fall_back_to_plain_tts(self, tmp_path):
        speech = FakeSpeech(reject_kwargs=("instructions",))
        forecast = build_forecast([make_dna()] * 2)
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [2] * len(chunks))

        manifest = make_synth(tmp_path, speech).synthesize_directed_preview(
            PREVIEW_SCRIPT, [], sheet, generate_audio=True
        )

        # Every chunk still rendered, via the undirected retry.
        assert all(c["audio_file"] for c in manifest["chunks"])
        assert all(call["model"] == TTS_MODEL for call in speech.calls)

    def test_preview_writes_to_its_own_directory(self, tmp_path):
        forecast = build_forecast([make_dna()])
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [1] * len(chunks))

        manifest = make_synth(tmp_path).synthesize_directed_preview(
            PREVIEW_SCRIPT, [], sheet, generate_audio=True
        )

        assert manifest["audio_dir"].endswith("preview_audio")
        assert (tmp_path / "preview_manifest.json").exists()
        assert (tmp_path / "preview_cue_sheet.txt").exists()

    def test_forecast_summary_is_attached(self, tmp_path):
        forecast = build_forecast([make_dna()])
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [1] * len(chunks))

        manifest = make_synth(tmp_path).synthesize_directed_preview(
            PREVIEW_SCRIPT, [], sheet, generate_audio=False,
            forecast_summary={"unlock_pull_index": 82.0},
        )
        assert manifest["forecast_summary"]["unlock_pull_index"] == 82.0

    def test_empty_preview_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            make_synth(tmp_path).synthesize_directed_preview("   ", [], [])

    def test_cue_sheet_carries_direction_and_disclaimer(self):
        forecast = build_forecast([make_dna()] * 2)
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [2] * len(chunks))

        text = build_directed_cue_sheet(chunks, sheet)
        assert "RETENTION-DIRECTED PREVIEW CUE SHEET" in text
        assert "not calibrated with listener data" in text
        assert "DELIVERY:" in text


class TestBackwardCompatibility:
    """The original production export must be untouched by the preview work."""

    def test_existing_synthesize_still_produces_its_manifest(self, tmp_path):
        manifest = make_synth(tmp_path).synthesize(
            PREVIEW_SCRIPT,
            [FoleyTrigger(timestamp="00:02", sound_effect="filament tick")],
            generate_audio=False,
        )

        for key in (
            "project", "generated_at", "tts_model", "audio_generated", "audio_dir",
            "total_runtime_seconds", "total_runtime", "casting", "chunk_count",
            "foley_cue_count", "chunks", "unassigned_foley",
        ):
            assert key in manifest

        assert "retention_directing_sheet" not in manifest
        assert (tmp_path / "production_manifest.json").exists()
        assert (tmp_path / "cue_sheet.txt").exists()

    def test_existing_synthesize_uses_the_original_model_and_directory(self, tmp_path):
        speech = FakeSpeech()
        make_synth(tmp_path, speech).synthesize(PREVIEW_SCRIPT, [], generate_audio=True)

        assert all(call["model"] == TTS_MODEL for call in speech.calls)
        assert all("instructions" not in call for call in speech.calls)
        assert (tmp_path / "scratch_audio").exists()

    def test_preview_does_not_overwrite_the_production_manifest(self, tmp_path):
        synth = make_synth(tmp_path)
        synth.synthesize(PREVIEW_SCRIPT, [], generate_audio=False)
        original = json.loads((tmp_path / "production_manifest.json").read_text())

        forecast = build_forecast([make_dna()])
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [1] * len(chunks))
        synth.synthesize_directed_preview(PREVIEW_SCRIPT, [], sheet, generate_audio=False)

        after = json.loads((tmp_path / "production_manifest.json").read_text())
        assert after == original

    def test_foley_still_pins_to_chunks_in_the_preview(self, tmp_path):
        forecast = build_forecast([make_dna()])
        chunks = parse_script(PREVIEW_SCRIPT)
        sheet = build_directing_sheet(chunks, forecast, [1] * len(chunks))

        manifest = make_synth(tmp_path).synthesize_directed_preview(
            PREVIEW_SCRIPT,
            [FoleyTrigger(timestamp="00:00", sound_effect="filament tick")],
            sheet,
            generate_audio=False,
        )
        placed = [cue for chunk in manifest["chunks"] for cue in chunk["foley"]]
        assert placed and placed[0]["sound_effect"] == "filament tick"
