"""
Retention directing sheet.

Turns a forecast into per-chunk performance direction for the voice session:
how fast to read, where to hold a pause, what the beat is doing, and which
foley belongs underneath it. The point is that the audio preview is directed
by the same evidence that produced the forecast, rather than being read flat.
"""

from __future__ import annotations

from .engagement_forecast import risk_band
from .schemas import CliffhangerReport, DirectingSheetEntry, ForecastResult, SceneAnalysis

# gpt-4o-mini-tts honours a speed hint alongside the instruction text. Kept
# inside a narrow band because anything past roughly 1.15 starts clipping
# consonants and stops sounding like performance.
SPEED_MIN = 0.88
SPEED_MAX = 1.15
SPEED_BASE = 1.0

PAUSE_STANDARD_MS = 250
PAUSE_REVEAL_MS = 750
PAUSE_FINAL_HOOK_MS = 900

# Share of a preview's wall-clock runtime taken by inter-chunk silence rather
# than speech, at the synthesizer's default pause length.
PAUSE_OVERHEAD = 0.12

TEMPO_LABELS = (
    (0.95, "measured"),
    (1.02, "conversational"),
    (1.08, "driving"),
    (SPEED_MAX, "urgent"),
)


def _tempo_label(speed: float) -> str:
    for ceiling, label in TEMPO_LABELS:
        if speed <= ceiling:
            return label
    return "urgent"


def _target_speed(analysis: SceneAnalysis, is_final: bool) -> float:
    """
    Derive a speech rate from what the scene actually contains.

    Dread reads slower and action reads faster; the episode's final beat pulls
    back deliberately so the hook lands rather than rushing past.
    """
    dna = analysis.dna
    speed = SPEED_BASE

    speed += 0.10 * (dna.action_density - 0.4)
    speed += 0.06 * (dna.scene_tempo - 0.5)
    speed -= 0.08 * dna.dread
    speed -= 0.05 * dna.exposition_ratio
    speed += 0.04 * dna.tension

    if is_final:
        speed -= 0.03

    return round(max(SPEED_MIN, min(SPEED_MAX, speed)), 3)


def _narrative_role(
    analysis: SceneAnalysis, position: int, total: int, cliffhanger: CliffhangerReport
) -> str:
    if position == total - 1:
        if "weak_resolved" in cliffhanger.types:
            return "episode_end_resolution"
        return "episode_end_hook"
    if position == 0:
        return "cold_open_hook" if analysis.dna.cliffhanger_strength > 0.4 else "setup"
    if analysis.dna.exposition_ratio >= 0.45:
        return "exposition_bridge"
    if analysis.dna.surprise_factor >= 0.5 or analysis.dna.new_information_revealed:
        return "reveal"
    if analysis.dna.conflict_present and analysis.dna.stakes_level > 0.5:
        return "escalation"
    return "development"


def _instruction(
    analysis: SceneAnalysis,
    role: str,
    band: str,
    is_narration: bool,
    speed: float,
) -> str:
    """Plain performance direction an actor or TTS model can follow."""
    dna = analysis.dna
    parts: list[str] = []

    if dna.dread > 0.5:
        parts.append("Controlled dread. Keep the voice low and unhurried")
    elif dna.tension > 0.5:
        parts.append("Held tension. Tight, contained delivery")
    elif dna.warmth > 0.5 or dna.romance > 0.4:
        parts.append("Warm and close, as if speaking to one person")
    elif dna.action_density > 0.5:
        parts.append("Forward momentum. Clipped and physical")
    elif dna.sadness > 0.4:
        parts.append("Quiet weight. Let the pauses carry it")
    else:
        parts.append("Grounded and natural")

    if role == "exposition_bridge":
        parts.append(
            "this passage carries information, so land the specifics and keep it moving "
            "rather than reciting"
        )
    elif role == "episode_end_hook":
        parts.append("hold a clear pause before the final revelation and stop cleanly, without a fade")
    elif role == "reveal":
        parts.append("plant the new fact deliberately, then let it sit")
    elif role == "escalation":
        parts.append("build steadily, do not peak early")

    if band == "high":
        parts.append("this beat is a flagged engagement risk, so keep the energy up and cut any drift")
    elif band == "elevated":
        parts.append("keep the pace honest here")

    if is_narration:
        parts.append("narrator voice: present and observational, never theatrical")

    parts.append(f"target rate about {speed:.2f}x")
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."


def _foley_for_chunk(chunk_foley: list[dict], analysis: SceneAnalysis) -> list[str]:
    """Cues already pinned to this chunk, else a suggestion from the scene's texture."""
    explicit = [str(cue.get("sound_effect", "")).strip() for cue in chunk_foley]
    explicit = [cue for cue in explicit if cue]
    if explicit:
        return explicit

    dna = analysis.dna
    suggestions: list[str] = []
    if dna.dread > 0.5:
        suggestions.append("low room tone, barely audible")
    if dna.action_density > 0.5:
        suggestions.append("movement and impact close to the mic")
    if dna.setting_change:
        suggestions.append("ambience shift marking the new space")
    return suggestions[:2]


def assign_scene_indices(chunks, scenes, parse_fn) -> list[int]:
    """
    Map each spoken chunk to the scene it came from.

    Scenes are re-parsed individually and chunks handed out in order, which
    works because parsing is deterministic. If the counts disagree with a
    whole-script parse, everything falls back to the first scene rather than
    silently misattributing direction.
    """
    per_scene = [len(parse_fn(scene.text)) for scene in scenes]
    if sum(per_scene) != len(chunks):
        return [scenes[0].index if scenes else 1] * len(chunks)

    mapping: list[int] = []
    for scene, count in zip(scenes, per_scene):
        mapping.extend([scene.index] * count)
    return mapping


def build_directing_sheet(
    chunks,
    forecast: ForecastResult,
    scene_indices: list[int],
) -> list[DirectingSheetEntry]:
    """
    Produce one direction entry per audio chunk.

    `scene_indices` must be parallel to `chunks`; use `assign_scene_indices`
    to build it.
    """
    if len(scene_indices) != len(chunks):
        raise ValueError("scene_indices must be parallel to chunks.")

    survival_by_scene = {s.scene_index: s.survival for s in forecast.primary_curve.scenes}
    hazard_by_scene = {s.scene_index: s.hazard for s in forecast.primary_curve.scenes}
    ordered = [a.scene.index for a in forecast.scenes]
    total = len(ordered)
    last_index = ordered[-1] if ordered else None

    entries: list[DirectingSheetEntry] = []

    for chunk, scene_index in zip(chunks, scene_indices):
        analysis = forecast.scene_by_index(scene_index) or forecast.scenes[0]
        position = ordered.index(analysis.scene.index) if analysis.scene.index in ordered else 0
        is_final_scene = analysis.scene.index == last_index
        is_last_chunk = chunk is chunks[-1]

        band = risk_band(hazard_by_scene.get(analysis.scene.index, 0.0))
        role = _narrative_role(analysis, position, total, forecast.cliffhanger)
        speed = _target_speed(analysis, is_final_scene)
        is_narration = chunk.kind == "narration"

        if is_final_scene and is_last_chunk and role == "episode_end_hook":
            pause = PAUSE_FINAL_HOOK_MS
        elif role in ("reveal", "episode_end_hook"):
            pause = PAUSE_REVEAL_MS
        else:
            pause = PAUSE_STANDARD_MS

        hook_role = ""
        if is_final_scene:
            hook_role = "+".join(forecast.cliffhanger.types)

        entries.append(
            DirectingSheetEntry(
                chunk_index=chunk.index,
                scene=analysis.scene.index,
                character=chunk.character,
                narrative_role=role,
                engagement_risk=band,
                dominant_emotion=analysis.dna.dominant_emotion,
                delivery_tempo=_tempo_label(speed),
                target_speed=speed,
                pause_before_reveal_ms=pause,
                instruction=_instruction(analysis, role, band, is_narration, speed),
                foley=_foley_for_chunk(chunk.foley, analysis),
                hook_role=hook_role,
                survival_proxy=survival_by_scene.get(analysis.scene.index, 0.0),
                unlock_pull_index=forecast.cliffhanger.unlock_pull_index,
            )
        )

    return entries


def select_preview_scenes(
    forecast: ForecastResult,
    *,
    target_seconds: float = 90.0,
    min_seconds: float = 60.0,
    words_per_second: float = 2.5,
    spoken_word_count=None,
) -> list[SceneAnalysis]:
    """
    Choose the scenes for a 60-90 second preview.

    Works backwards from the ending, because the hook is the thing worth
    previewing. Earlier scenes are added only while the budget allows.

    Pass `spoken_word_count` to measure a scene by the words that will
    actually be voiced. Action lines and slug lines are never performed, so
    budgeting on raw scene length produces a preview roughly a third of the
    intended runtime.
    """
    if not forecast.scenes:
        return []

    count = spoken_word_count or (lambda scene: scene.word_count)

    # The timeline inserts a beat of silence between chunks, so pure speech
    # has to come in under the wall-clock target to land inside it.
    speech_fraction = 1.0 - PAUSE_OVERHEAD
    budget_words = target_seconds * words_per_second * speech_fraction
    floor_words = min_seconds * words_per_second * speech_fraction

    selected: list[SceneAnalysis] = []
    used = 0

    for analysis in reversed(forecast.scenes):
        words = count(analysis.scene)

        # Scenes are indivisible, so the budget rarely lands exactly. A
        # preview that runs slightly long is far more useful than one that
        # stops before the setup makes sense, so the minimum wins.
        overshoots = selected and used + words > budget_words
        if overshoots and used >= floor_words:
            break

        selected.append(analysis)
        used += words
        if used >= floor_words:
            break

    return list(reversed(selected))
