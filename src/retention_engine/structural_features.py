"""
Deterministic structural signals.

Everything here is computed locally with regular expressions and arithmetic.
No model is involved, so these numbers are reproducible, free, and citable.
When the UI tells a writer "exposition was high for two consecutive scenes",
that claim comes from this module and can be checked by hand.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from .schemas import Scene, SceneDNA, StructuralSignals

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")
CHARACTER_CUE_RE = re.compile(r"^([A-Z][A-Z0-9 .'\-]{0,30})(\s*\([^)]*\))*\s*$")
SCENE_HEADING_RE = re.compile(r"^\s*(INT\.|EXT\.|INT\./EXT\.|I/E\.)", re.IGNORECASE)
QUOTED_SPEECH_RE = re.compile(r"[\"\u201c][^\"\u201d]{2,}[\"\u201d]")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Words that put an explicit clock on the scene. Deliberately narrow: vague
# words like "soon" would fire on almost any passage and stop discriminating.
TIME_PRESSURE_TERMS = (
    "hours", "minutes", "seconds", "midnight", "deadline", "countdown",
    "before dawn", "by morning", "running out", "too late", "no time",
    "any moment", "last chance", "final warning", "tonight",
)

# Markers that a scene ends on a pull rather than a settle.
HOOK_MARKERS = (
    "--", "—", "...", "…",
)
HOOK_PHRASES = (
    "smash to", "cut to black", "silence.", "then nothing", "behind her",
    "behind him", "was gone", "wasn't there", "wasn't alone", "was already",
)

# A scene is treated as exposition-heavy or low-conflict above and below these
# lines. They are thresholds on sensor output, kept here so the streak logic
# and the hazard model cannot drift apart.
EXPOSITION_HEAVY = 0.45
LOW_EVENT_MOVEMENT = 0.30


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _is_character_cue(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.endswith("."):
        return False
    if SCENE_HEADING_RE.match(stripped):
        return False
    return bool(CHARACTER_CUE_RE.match(stripped))


def _clean_cue(line: str) -> str:
    name = re.sub(r"\([^)]*\)", "", line).strip()
    return re.sub(r"\s+", " ", name).upper()


def extract_speakers(text: str) -> list[str]:
    """Character cues in order of appearance, duplicates preserved."""
    return [_clean_cue(ln) for ln in text.splitlines() if _is_character_cue(ln)]


def compute_scene_signals(scene: Scene) -> StructuralSignals:
    """Measure one scene with no model in the loop."""
    text = scene.text
    words = text.split()
    word_count = len(words)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    sentences = _sentences(text)
    lengths = [len(s.split()) for s in sentences] or [0]
    mean_len = statistics.fmean(lengths)
    variance = statistics.pvariance(lengths) if len(lengths) > 1 else 0.0

    urgency_hits = text.count("!") + text.count("?") + text.count("--") + text.count("—")
    urgency_density = urgency_hits / word_count if word_count else 0.0

    speakers = extract_speakers(text)
    cue_lines = len(speakers)
    quoted = len(QUOTED_SPEECH_RE.findall(text))

    if cue_lines:
        # Screenplay: a cue plus the lines under it, capped at the line count.
        dialogue_ratio = min(1.0, (cue_lines * 2) / len(lines)) if lines else 0.0
    elif lines:
        dialogue_ratio = min(1.0, quoted / len(lines))
    else:
        dialogue_ratio = 0.0

    if speakers:
        counts = Counter(speakers)
        pov_dominance = counts.most_common(1)[0][1] / len(speakers)
        unique_named = len(counts)
    else:
        proper = Counter(PROPER_NOUN_RE.findall(text))
        unique_named = len(proper)
        pov_dominance = (
            proper.most_common(1)[0][1] / sum(proper.values()) if proper else 0.0
        )

    named_density = unique_named / (word_count / 100) if word_count else 0.0

    question_density = text.count("?") / word_count if word_count else 0.0

    lowered = text.lower()
    time_hits = sum(lowered.count(term) for term in TIME_PRESSURE_TERMS)

    tail = "\n".join(lines[-3:]).lower() if lines else ""
    hook = any(m in tail for m in HOOK_MARKERS) or any(p in tail for p in HOOK_PHRASES)

    return StructuralSignals(
        word_count=word_count,
        sentence_count=len(sentences),
        mean_sentence_length=round(mean_len, 2),
        sentence_length_variance=round(variance, 2),
        urgency_punctuation_density=round(urgency_density, 4),
        dialogue_line_ratio=round(dialogue_ratio, 3),
        named_character_density=round(named_density, 3),
        pov_dominance=round(pov_dominance, 3),
        question_density=round(question_density, 4),
        time_pressure_hits=time_hits,
        hook_marker_present=hook,
        speaking_characters=sorted(set(speakers)),
    )


def apply_sequence_signals(
    signals: list[StructuralSignals],
    dna: list[SceneDNA],
) -> list[StructuralSignals]:
    """
    Fill in the signals that only exist across a sequence.

    Payoff debt is the running count of mysteries opened minus those
    meaningfully resolved. It is cumulative on purpose: an episode that keeps
    opening threads without closing any accumulates a debt the listener feels
    even when each individual scene reads fine.
    """
    if len(signals) != len(dna):
        raise ValueError("signals and dna must describe the same scenes.")

    exposition_streak = 0
    low_conflict_streak = 0
    pov_streak = 0
    opened_total = 0
    resolved_total = 0
    previous_lead: str | None = None

    for signal, scene_dna in zip(signals, dna):
        if scene_dna.exposition_ratio >= EXPOSITION_HEAVY:
            exposition_streak += 1
        else:
            exposition_streak = 0

        if not scene_dna.conflict_present or scene_dna.event_movement < LOW_EVENT_MOVEMENT:
            low_conflict_streak += 1
        else:
            low_conflict_streak = 0

        opened_total += scene_dna.mystery_questions_opened
        resolved_total += scene_dna.mystery_questions_answered

        lead = signal.speaking_characters[0] if signal.speaking_characters else None
        if lead is not None and lead == previous_lead:
            pov_streak += 1
        else:
            pov_streak = 1 if lead is not None else 0
        previous_lead = lead

        signal.consecutive_exposition_scenes = exposition_streak
        signal.consecutive_low_conflict_scenes = low_conflict_streak
        signal.open_questions_running_total = max(0, opened_total - resolved_total)
        signal.payoff_debt = max(0, opened_total - resolved_total)
        signal.pov_repeat_streak = pov_streak

    return signals


def describe_evidence(signal: StructuralSignals, scene_dna: SceneDNA) -> list[str]:
    """
    Plain-language evidence lines for the UI.

    Only conditions that actually fired are returned, so an empty list means
    the scene is structurally clean.
    """
    notes: list[str] = []

    if signal.consecutive_exposition_scenes >= 2:
        notes.append(
            f"Exposition has been heavy for {signal.consecutive_exposition_scenes} "
            f"consecutive scenes (this scene {scene_dna.exposition_ratio:.0%})."
        )
    elif scene_dna.exposition_ratio >= EXPOSITION_HEAVY:
        notes.append(f"Exposition occupies {scene_dna.exposition_ratio:.0%} of the scene.")

    if not scene_dna.conflict_present:
        notes.append("No active conflict: nothing is pushing against anything.")

    if signal.consecutive_low_conflict_scenes >= 2:
        notes.append(
            f"{signal.consecutive_low_conflict_scenes} consecutive scenes with little "
            "event movement."
        )

    if signal.payoff_debt >= 3:
        notes.append(
            f"Payoff debt is {signal.payoff_debt}: mysteries opened well outnumber "
            "those resolved."
        )

    if signal.pov_repeat_streak >= 3:
        lead = signal.speaking_characters[0] if signal.speaking_characters else "one character"
        notes.append(f"{lead} has led {signal.pov_repeat_streak} scenes in a row.")

    if scene_dna.complexity >= 0.7:
        notes.append(
            f"High cognitive load (complexity {scene_dna.complexity:.2f}, "
            f"worldbuilding {scene_dna.worldbuilding_density:.2f})."
        )

    if scene_dna.emotional_intensity < 0.3:
        notes.append(
            f"Emotionally flat: peak intensity {scene_dna.emotional_intensity:.2f}."
        )

    if signal.mean_sentence_length > 28:
        notes.append(
            f"Long average sentence ({signal.mean_sentence_length:.0f} words) slows narration."
        )

    if not signal.hook_marker_present and scene_dna.cliffhanger_strength < 0.3:
        notes.append("The scene ends on a settle rather than a pull.")

    return notes
