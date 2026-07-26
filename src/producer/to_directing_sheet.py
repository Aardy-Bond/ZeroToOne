"""
Map a ProducerPlan onto DirectingSheetEntry rows the TTS path already understands.

Loudness becomes instruction language — OpenAI TTS has no gain fader.
"""

from __future__ import annotations

from retention_engine.engagement_forecast import risk_band
from retention_engine.schemas import DirectingSheetEntry, ForecastResult

from .schemas import (
    LineCue,
    LoudnessBand,
    ProducerPlan,
    SceneDirection,
    SpeakableLine,
    TempoBand,
)

LOUDNESS_PHRASE: dict[LoudnessBand, str] = {
    "whisper": "Almost a whisper — intimate and close to the mic",
    "intimate": "Quiet and close, as if speaking to one person",
    "full": "Natural speaking volume",
    "projected": "Project with clear force — fill the space without shouting",
}

TEMPO_SPEED: dict[TempoBand, float] = {
    "measured": 0.90,
    "conversational": 1.00,
    "driving": 1.08,
    "urgent": 1.14,
}


def loudness_instruction(band: LoudnessBand | str | None) -> str:
    if not band:
        return ""
    return LOUDNESS_PHRASE.get(band, LOUDNESS_PHRASE["full"])  # type: ignore[arg-type]


def plan_to_directing_sheet(
    plan: ProducerPlan,
    lines: list[SpeakableLine],
    forecast: ForecastResult,
) -> list[DirectingSheetEntry]:
    """One DirectingSheetEntry per speakable line."""
    scene_dir = {s.scene_index: s for s in plan.scenes}
    cues = {c.line_index: c for c in plan.line_cues}
    sounds: dict[int, list[str]] = {}
    for cue in plan.sound_cues:
        sounds.setdefault(cue.line_index, []).append(cue.effect)

    survival_by = {s.scene_index: s.survival for s in forecast.primary_curve.scenes}
    hazard_by = {s.scene_index: s.hazard for s in forecast.primary_curve.scenes}
    ordered = [a.scene.index for a in forecast.scenes]
    last_index = ordered[-1] if ordered else None

    entries: list[DirectingSheetEntry] = []
    for i, line in enumerate(lines):
        scene = scene_dir.get(line.scene_index) or _default_scene(line.scene_index)
        cue = cues.get(line.index)
        analysis = forecast.scene_by_index(line.scene_index)
        emotion = (
            analysis.dna.dominant_emotion
            if analysis is not None
            else scene.emotional_color
        )
        tempo = (cue.tempo if cue and cue.tempo else scene.tempo)
        loudness = (cue.loudness if cue and cue.loudness else scene.loudness)
        speed = TEMPO_SPEED.get(tempo, 1.0)
        pause = cue.pause_before_ms if cue else (250 if i == 0 else 200)
        if line.scene_index == last_index and i == len(lines) - 1:
            pause = max(pause, 750)

        instruction = _compose_instruction(scene, cue, loudness, line)
        band = risk_band(hazard_by.get(line.scene_index, 0.0))
        role = _role(line, i, len(lines), line.scene_index == last_index)

        entries.append(
            DirectingSheetEntry(
                chunk_index=line.index,
                scene=line.scene_index,
                character=line.character,
                narrative_role=role,
                engagement_risk=band,  # type: ignore[arg-type]
                dominant_emotion=str(emotion),
                delivery_tempo=tempo,
                target_speed=speed,
                pause_before_reveal_ms=pause,
                instruction=instruction,
                foley=sounds.get(line.index, []),
                hook_role=(
                    "+".join(forecast.cliffhanger.types)
                    if line.scene_index == last_index
                    else ""
                ),
                survival_proxy=survival_by.get(line.scene_index, 0.0),
                unlock_pull_index=forecast.cliffhanger.unlock_pull_index,
            )
        )
    return entries


def casting_overrides(plan: ProducerPlan) -> dict[str, str]:
    return {c.character: c.voice for c in plan.casting}


def _default_scene(scene_index: int) -> SceneDirection:
    return SceneDirection(scene_index=scene_index)


def _compose_instruction(
    scene: SceneDirection,
    cue: LineCue | None,
    loudness: LoudnessBand | str,
    line: SpeakableLine,
) -> str:
    parts: list[str] = []
    color = (cue.instruction if cue and cue.instruction else scene.note) or scene.emotional_color
    if color:
        parts.append(color.rstrip("."))
    loud = loudness_instruction(loudness)
    if loud and loudness != "full":
        parts.append(loud.rstrip("."))
    if cue and cue.emphasis:
        parts.append(f"Land the words “{cue.emphasis}” clearly")
    if cue and cue.breath_hold:
        parts.append("Take a breath hold before the key phrase")
    if line.kind == "narration":
        parts.append("Narration — clear and present, not theatrical")
    if not parts:
        parts.append("Grounded and natural")
    text = ". ".join(parts)
    return text[0].upper() + text[1:] + ("." if not text.endswith(".") else "")


def _role(line: SpeakableLine, position: int, total: int, is_final_scene: bool) -> str:
    if position == 0:
        return "cold_open_hook"
    if is_final_scene and position == total - 1:
        return "episode_end_hook"
    if line.kind == "narration":
        return "narration"
    return "development"
