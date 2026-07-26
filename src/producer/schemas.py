"""
Schemas for the AI Producer plan.

The producer turns a retention forecast into casting, selective line cues,
sound design, and a short marketing brief. Strict structured output — every
field the model returns is declared here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from retention_engine.schemas import FORECAST_DISCLAIMER

LoudnessBand = Literal["whisper", "intimate", "full", "projected"]
TempoBand = Literal["measured", "conversational", "driving", "urgent"]
EnergyBand = Literal["low", "medium", "high"]

# Voices available to gpt-4o-mini-tts / tts-1-hd in this project.
AllowedVoice = Literal["onyx", "nova", "echo", "shimmer", "fable", "alloy"]

PRODUCER_DISCLAIMER = FORECAST_DISCLAIMER


class CastingChoice(BaseModel):
    """One character cast to a TTS voice."""

    model_config = ConfigDict(extra="forbid")

    character: str
    voice: AllowedVoice
    rationale: str = Field(description="One short sentence explaining the cast.")


class SceneDirection(BaseModel):
    """Default delivery for a whole scene; line cues only override when needed."""

    model_config = ConfigDict(extra="forbid")

    scene_index: int
    tempo: TempoBand = "conversational"
    loudness: LoudnessBand = "full"
    energy: EnergyBand = "medium"
    emotional_color: str = Field(
        default="grounded",
        description="Short color note, e.g. controlled dread, warm close.",
    )
    note: str = Field(default="", description="One sentence scene-level direction.")


class LineCue(BaseModel):
    """
    Selective override for one speakable line.

    Only emit when the line should deviate from the scene default — roughly
    one cue per two to three lines, always covering cold open, reveals,
    high-risk scenes, and the final hook.
    """

    model_config = ConfigDict(extra="forbid")

    line_index: int = Field(description="1-based index into the numbered speakable lines.")
    tempo: TempoBand | None = None
    loudness: LoudnessBand | None = None
    pause_before_ms: int = Field(default=0, ge=0, le=2000)
    emphasis: str = Field(default="", description="Word or short phrase to land, if any.")
    breath_hold: bool = False
    instruction: str = Field(
        default="",
        description="Plain performance note an actor or TTS model can follow.",
    )


class SoundCue(BaseModel):
    """A sound effect attached to a speakable line."""

    model_config = ConfigDict(extra="forbid")

    line_index: int = Field(description="1-based speakable line the cue attaches to.")
    effect: str = Field(description="Short SFX label, e.g. distant door, heartbeat swell.")
    placement: Literal["under", "before", "after"] = "under"


class MarketingBrief(BaseModel):
    """Compact episode marketing notes — not live retention claims."""

    model_config = ConfigDict(extra="forbid")

    logline: str
    hook_bullets: list[str] = Field(default_factory=list, max_length=5)
    target_listener: str = ""
    title_treatment: str = Field(
        default="",
        description="Suggested episode title or title card treatment.",
    )
    disclaimer: str = PRODUCER_DISCLAIMER


class ProducerPlan(BaseModel):
    """Full retention-directed production plan for one draft."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(
        description="One sentence production strategy grounded in the forecast."
    )
    casting: list[CastingChoice] = Field(default_factory=list)
    scenes: list[SceneDirection] = Field(default_factory=list)
    line_cues: list[LineCue] = Field(default_factory=list)
    sound_cues: list[SoundCue] = Field(default_factory=list)
    marketing: MarketingBrief
    disclaimer: str = PRODUCER_DISCLAIMER


class SpeakableLine(BaseModel):
    """One numbered speakable unit aligned to the draft for cue attachment."""

    model_config = ConfigDict(extra="forbid")

    index: int
    character: str
    text: str
    kind: Literal["dialogue", "narration"] = "dialogue"
    scene_index: int = 1
