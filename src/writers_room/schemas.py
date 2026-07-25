"""
Structured output schemas for the Project Anubhuti Writers Room.

These models are passed directly to OpenAI structured outputs, so every field
must be required and every object must forbid extra keys.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FoleyTrigger(BaseModel):
    """A single sound cue anchored to a moment in the scene."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(
        description="Where the cue lands, as MM:SS or a scene beat reference."
    )
    sound_effect: str = Field(
        description="The specific sound to lay in, e.g. 'wet floorboard creak'."
    )


class SceneCritique(BaseModel):
    """Combined notes from the four-expert panel for one scene."""

    model_config = ConfigDict(extra="forbid")

    director_notes: str = Field(
        description="Visuals, blocking, camera angles, and pacing."
    )
    editor_notes: str = Field(
        description="Dialogue cuts, filler removal, and tightening."
    )
    psychologist_notes: str = Field(
        description="Emotional stakes, character tension, and motivation."
    )
    continuity_critique: str = Field(
        description=(
            "Any contradictions between the scene and the established canon "
            "supplied in the canon warnings. State 'No canon violations "
            "detected.' when the scene is consistent."
        )
    )
    foley_triggers: list[FoleyTrigger] = Field(
        description="Sound design cues mapped to specific moments."
    )
