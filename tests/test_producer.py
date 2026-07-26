"""AI Producer — context pack, alignment, loudness→instruction, disclaimer."""

from __future__ import annotations

from conftest import FakeOpenAI, make_dna
from test_forecast import build_forecast

from producer.agent import run_producer
from producer.context import pack_forecast_context
from producer.schemas import (
    PRODUCER_DISCLAIMER,
    CastingChoice,
    LineCue,
    MarketingBrief,
    ProducerPlan,
    SceneDirection,
    SoundCue,
    SpeakableLine,
)
from producer.script_align import align_plan_to_lines, extract_speakable_lines
from producer.to_directing_sheet import (
    loudness_instruction,
    plan_to_directing_sheet,
)

SCREENPLAY = (
    "INT. BASEMENT - NIGHT\n\n"
    "PRIYA\nTwo.\n\n"
    "NARRATOR (V.O.)\nSomething breathes. It is not a child.\n\n"
    "DEV\nPriya. What did you find down here.\n"
)

PROSE = (
    "The cellar door stuck halfway. Priya counted the steps.\n\n"
    "Something breathed that was not a child.\n\n"
    "She whispered the number again: two.\n"
)


def _sample_plan(line_count: int = 3) -> ProducerPlan:
    return ProducerPlan(
        strategy="Slow the dread, then land the hook hard.",
        casting=[
            CastingChoice(character="PRIYA", voice="nova", rationale="Young, clear."),
            CastingChoice(character="NARRATOR", voice="onyx", rationale="Steady guide."),
            CastingChoice(character="DEV", voice="echo", rationale="Tense contrast."),
        ],
        scenes=[
            SceneDirection(
                scene_index=1,
                tempo="measured",
                loudness="intimate",
                energy="low",
                emotional_color="controlled dread",
                note="Keep it close.",
            )
        ],
        line_cues=[
            LineCue(
                line_index=1,
                tempo="measured",
                loudness="whisper",
                pause_before_ms=500,
                instruction="Almost nothing above a breath.",
            ),
            LineCue(line_index=99, instruction="orphan cue"),
        ],
        sound_cues=[
            SoundCue(line_index=2, effect="low room tone"),
            SoundCue(line_index=99, effect="ghost"),
        ],
        marketing=MarketingBrief(
            logline="A number in the dark.",
            hook_bullets=["She counts.", "Something answers."],
            target_listener="Thriller commuters",
            title_treatment="TWO",
            disclaimer="made up",
        ),
    )


class TestExtractLines:
    def test_screenplay_lines(self):
        forecast = build_forecast([make_dna()])
        lines = extract_speakable_lines(SCREENPLAY, forecast)
        assert len(lines) == 3
        assert lines[0].character == "PRIYA"
        assert "breathes" in lines[1].text.lower()

    def test_prose_falls_back_to_narrator(self):
        forecast = build_forecast([make_dna(), make_dna()])
        lines = extract_speakable_lines(PROSE, forecast)
        assert len(lines) >= 2
        assert all(line.character == "NARRATOR" for line in lines)


class TestContext:
    def test_pack_includes_survival_and_numbered_lines(self):
        forecast = build_forecast(
            [make_dna(tension=0.9, dread=0.8), make_dna(cliffhanger_strength=0.9)]
        )
        lines = extract_speakable_lines(SCREENPLAY, forecast)
        brief = pack_forecast_context(forecast, lines)
        assert "Overall survival proxy" in brief
        assert "Unlock Pull Index" in brief
        assert "EKG SNAPSHOT" in brief
        assert "[1|" in brief
        assert "not live" in brief.lower() or "Simulated" in brief


class TestAlign:
    def test_drops_orphan_indices_and_fixes_disclaimer(self):
        forecast = build_forecast([make_dna()])
        lines = extract_speakable_lines(SCREENPLAY, forecast)
        plan = align_plan_to_lines(_sample_plan(), lines)
        assert all(c.line_index <= len(lines) for c in plan.line_cues)
        assert all(c.line_index <= len(lines) for c in plan.sound_cues)
        assert 99 not in {c.line_index for c in plan.line_cues}


class TestLoudnessMapping:
    def test_whisper_phrase_in_instruction(self):
        phrase = loudness_instruction("whisper")
        assert "whisper" in phrase.lower()

    def test_sheet_embeds_loudness_in_instruction(self):
        forecast = build_forecast([make_dna(dread=0.9)])
        lines = extract_speakable_lines(SCREENPLAY, forecast)
        plan = align_plan_to_lines(_sample_plan(), lines)
        sheet = plan_to_directing_sheet(plan, lines, forecast)
        first = next(e for e in sheet if e.chunk_index == 1)
        assert "whisper" in first.instruction.lower()
        assert first.target_speed < 1.0


class TestAgentDisclaimer:
    def test_run_producer_forces_honesty_disclaimer(self):
        forecast = build_forecast([make_dna()])
        plan = _sample_plan()
        # Strip casting names that won't match if parse differs — agent path
        # will re-cast. Feed FakeOpenAI the plan with a weak disclaimer.
        client = FakeOpenAI([plan])
        result, lines = run_producer(SCREENPLAY, forecast, client=client)
        assert lines
        assert "not calibrated" in result.disclaimer.lower()
        assert "not calibrated" in result.marketing.disclaimer.lower()
        assert PRODUCER_DISCLAIMER.split("—")[0].strip()[:20] in result.disclaimer
