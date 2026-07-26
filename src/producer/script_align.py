"""
Turn a draft into numbered speakable lines the producer can cue against.

Screenplay dialogue uses the audio engine's parser. Prose drafts (no character
cues) fall back to paragraph-as-narration so the producer still has lines to
direct — matching how writers often paste story text into the desk.
"""

from __future__ import annotations

import re

from audio_engine.synthesizer import NARRATOR_KEY, ScriptChunk, parse_script
from retention_engine.schemas import ForecastResult, Scene

from .schemas import LineCue, ProducerPlan, SpeakableLine

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")


def extract_speakable_lines(
    draft: str,
    forecast: ForecastResult | None = None,
) -> list[SpeakableLine]:
    """Numbered speakable units with best-effort scene indices."""
    draft = (draft or "").strip()
    if not draft:
        return []

    chunks = parse_script(draft)
    if not chunks:
        chunks = _prose_as_narration(draft)

    scene_indices = _assign_scenes(chunks, draft, forecast)
    return [
        SpeakableLine(
            index=chunk.index,
            character=chunk.character,
            text=chunk.text,
            kind="narration" if chunk.kind == "narration" else "dialogue",
            scene_index=scene_indices[i],
        )
        for i, chunk in enumerate(chunks)
    ]


def chunks_from_lines(lines: list[SpeakableLine]) -> list[ScriptChunk]:
    """Rebuild ScriptChunk list for TTS / directing-sheet mapping."""
    return [
        ScriptChunk(
            index=line.index,
            kind=line.kind,
            character=line.character,
            text=line.text,
        )
        for line in lines
    ]


def align_plan_to_lines(plan: ProducerPlan, lines: list[SpeakableLine]) -> ProducerPlan:
    """
    Drop cues that point at missing lines; clamp scene directions to known scenes.

    The model sometimes invents indices. Alignment keeps the UI and TTS honest.
    """
    if not lines:
        return plan.model_copy(
            update={"line_cues": [], "sound_cues": [], "scenes": []}
        )

    valid = {line.index for line in lines}
    scenes_present = {line.scene_index for line in lines}
    characters = {line.character.upper() for line in lines}

    casting = [
        c for c in plan.casting if c.character.strip().upper() in characters
        or c.character.strip().upper().replace(" ", "")
        in {n.replace(" ", "") for n in characters}
    ]
    # Normalise character names to the draft spelling.
    name_by_upper = {line.character.upper(): line.character for line in lines}
    normalised_casting = []
    seen: set[str] = set()
    for choice in casting:
        key = choice.character.strip().upper()
        canonical = name_by_upper.get(key)
        if canonical is None:
            for draft_name, original in name_by_upper.items():
                if draft_name.replace(" ", "") == key.replace(" ", ""):
                    canonical = original
                    break
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        normalised_casting.append(choice.model_copy(update={"character": canonical}))

    # Ensure every speaking character has a cast entry (filled later if empty).
    scenes = [s for s in plan.scenes if s.scene_index in scenes_present]
    line_cues = _dedupe_line_cues(
        [c for c in plan.line_cues if c.line_index in valid]
    )
    sound_cues = [c for c in plan.sound_cues if c.line_index in valid]

    return plan.model_copy(
        update={
            "casting": normalised_casting,
            "scenes": scenes,
            "line_cues": line_cues,
            "sound_cues": sound_cues,
        }
    )


def _dedupe_line_cues(cues: list[LineCue]) -> list[LineCue]:
    best: dict[int, LineCue] = {}
    for cue in cues:
        prior = best.get(cue.line_index)
        if prior is None or len(cue.instruction) >= len(prior.instruction):
            best[cue.line_index] = cue
    return [best[i] for i in sorted(best)]


def _prose_as_narration(draft: str) -> list[ScriptChunk]:
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(draft) if p.strip()]
    if not paragraphs:
        # Single block with newlines — treat non-empty lines as units.
        paragraphs = [ln.strip() for ln in draft.splitlines() if ln.strip()]
    chunks: list[ScriptChunk] = []
    for paragraph in paragraphs:
        # Skip obvious slug lines so they do not get voiced.
        if re.match(r"^(INT\.|EXT\.|INT\./EXT\.|I/E\.)", paragraph, re.I):
            continue
        chunks.append(
            ScriptChunk(
                index=len(chunks) + 1,
                kind="narration",
                character=NARRATOR_KEY,
                text=re.sub(r"\s+", " ", paragraph).strip(),
            )
        )
    return chunks


def _assign_scenes(
    chunks: list[ScriptChunk],
    draft: str,
    forecast: ForecastResult | None,
) -> list[int]:
    if not chunks:
        return []
    if forecast is None or not forecast.scenes:
        return [1] * len(chunks)

    scenes = forecast.scenes
    # Prefer mapping by parsing each scene's spoken content when formats match.
    from retention_engine.directing_sheet import assign_scene_indices

    mapped = assign_scene_indices(
        chunks,
        [a.scene for a in scenes],
        parse_fn=lambda text: parse_script(text) or _prose_as_narration(text),
    )
    if mapped and len(mapped) == len(chunks):
        return mapped

    # Fall back: proportionally distribute lines across scenes by word share.
    return _proportional_scene_map(chunks, [a.scene for a in scenes])


def _proportional_scene_map(
    chunks: list[ScriptChunk],
    scenes: list[Scene],
) -> list[int]:
    if not scenes:
        return [1] * len(chunks)
    weights = [max(s.word_count, 1) for s in scenes]
    weight_sum = sum(weights) or 1
    quotas = [
        max(1, round(len(chunks) * (w / weight_sum))) for w in weights
    ]
    # Adjust so quotas sum to len(chunks).
    while sum(quotas) > len(chunks) and any(q > 1 for q in quotas):
        for i in range(len(quotas)):
            if sum(quotas) <= len(chunks):
                break
            if quotas[i] > 1:
                quotas[i] -= 1
    while sum(quotas) < len(chunks):
        quotas[-1] += 1

    mapping: list[int] = []
    for scene, quota in zip(scenes, quotas):
        mapping.extend([scene.index] * quota)
    return mapping[: len(chunks)]
