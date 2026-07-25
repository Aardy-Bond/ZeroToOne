"""
Auto-rewrite loop for Project Anubhuti.

Takes the minutes where the synthetic audience disengaged, pairs them with the
expert panel's notes, and rewrites only those passages while holding tone and
canon steady.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Sequence

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

try:
    from .schemas import SceneCritique
except ImportError:  # pragma: no cover - direct script execution
    from schemas import SceneCritique

if TYPE_CHECKING:
    from audience_simulator.simulator import MinuteHeatmapEntry

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_THRESHOLD = 5.0
DEFAULT_WORDS_PER_MINUTE = 150

# How many individual personas must drop below threshold to flag a minute even
# when the group average stays above it.
DEFAULT_MIN_DROPPED_PERSONAS = 2

SYSTEM_PROMPT = """\
You are the showrunner performing a targeted punch-up pass on an audio drama \
script. A synthetic test audience has already listened to it and told you \
exactly which minutes lost them, and a panel of craft experts has already \
given notes.

Your mandate is narrow and strict:

FIX ONLY WHAT IS BROKEN. Rewrite the flagged passages. Leave every other line \
of the script byte-for-byte identical. You are not doing a general polish and \
you are not improving lines nobody complained about.

PRESERVE CANON. Any fact stated in the canon section is immutable. Do not \
contradict it, and do not invent new backstory, characters, or locations to \
patch a pacing problem.

PRESERVE TONE AND VOICE. The rewrite must sound like the same writer wrote it. \
Match the existing register, rhythm, and vocabulary. A horror script stays \
horror.

FIX THE STATED CAUSE. Each flagged minute comes with the specific reason \
listeners disengaged. Address that reason directly. If they were bored by \
exposition, dramatize it or cut it. If they were confused, clarify through \
action or dialogue rather than narration. If dread was not building, tighten \
and escalate.

HOLD THE RUNTIME. The rewritten passage should be close to the original length \
unless the note explicitly calls for cuts. Do not pad.

Return the complete rewritten script, including the untouched passages, so it \
can be used directly. Then log each change you made and why.\
"""

USER_TEMPLATE = """\
{canon_block}

[EXPERT PANEL NOTES]
Director: {director_notes}
Editor: {editor_notes}
Psychologist: {psychologist_notes}
Continuity: {continuity_critique}

[FLAGGED MINUTES]
{weak_block}

[FULL SCRIPT]
{script_text}\
"""


class RewriteEngineError(Exception):
    """Raised when the rewrite pass cannot complete."""


class SegmentChange(BaseModel):
    """One targeted edit made during the punch-up pass."""

    model_config = ConfigDict(extra="forbid")

    minute: int = Field(description="The heatmap minute this change addresses.")
    problem: str = Field(description="Why the audience disengaged here.")
    change_made: str = Field(description="What was rewritten, concretely.")


class RewriteResult(BaseModel):
    """Output of a targeted rewrite pass."""

    model_config = ConfigDict(extra="forbid")

    rewritten_script: str = Field(
        description="The complete script with flagged passages rewritten."
    )
    change_log: list[SegmentChange] = Field(
        description="One entry per flagged minute that was addressed."
    )
    tone_continuity_note: str = Field(
        description="How tone and established canon were preserved."
    )


def _build_client(api_key: str | None = None) -> OpenAI:
    load_dotenv()
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise RewriteEngineError("Missing required environment variable: OPENAI_API_KEY")
    try:
        return OpenAI(api_key=key)
    except OpenAIError as exc:
        raise RewriteEngineError(f"Failed to initialize OpenAI client: {exc}") from exc


def segment_script_by_minute(
    script_text: str,
    words_per_minute: int = DEFAULT_WORDS_PER_MINUTE,
) -> dict[int, str]:
    """
    Split a script into the same minute blocks the audience simulator scored.

    Splits on whitespace while preserving line structure, so the returned
    passages can be quoted back to the model verbatim.
    """
    if words_per_minute < 1:
        raise ValueError("words_per_minute must be at least 1.")

    segments: dict[int, str] = {}
    words: list[str] = []
    minute = 1

    for token in script_text.split(" "):
        words.append(token)
        # Newlines survive inside tokens, so count them as word boundaries too.
        if len(" ".join(words).split()) >= words_per_minute:
            segments[minute] = " ".join(words).strip()
            words = []
            minute += 1

    if words and " ".join(words).strip():
        segments[minute] = " ".join(words).strip()

    return segments


def select_weak_minutes(
    heatmap: Sequence[MinuteHeatmapEntry],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_segments: int | None = None,
    min_dropped_personas: int = DEFAULT_MIN_DROPPED_PERSONAS,
) -> list[MinuteHeatmapEntry]:
    """
    Return the minutes worth rewriting, weakest first.

    A minute qualifies if its average falls below `threshold` OR if at least
    `min_dropped_personas` individual listeners dropped below it. The second
    condition matters: one enthusiastic persona can drag an average above the
    line even when most of the audience checked out, and averaging alone would
    silently skip that passage.
    """
    weak: list[MinuteHeatmapEntry] = []
    for entry in heatmap:
        dropped = sum(
            1 for score in entry.persona_scores.values() if score < threshold
        )
        if entry.average_score < threshold or dropped >= min_dropped_personas:
            weak.append(entry)

    weak.sort(key=lambda e: e.average_score)
    if max_segments is not None:
        weak = weak[:max_segments]
    return weak


def _format_weak_block(
    weak: Sequence[MinuteHeatmapEntry],
    segments: dict[int, str],
) -> str:
    blocks: list[str] = []
    for entry in weak:
        passage = segments.get(entry.minute, "").strip()
        reasons = entry.drop_off_reasons or {}
        reason_lines = (
            "\n".join(f"    - {persona}: {reason}" for persona, reason in reasons.items())
            or "    - (no specific reason captured; score was simply low)"
        )
        blocks.append(
            f"MINUTE {entry.minute} — average engagement {entry.average_score:.2f}\n"
            f"  Why listeners disengaged:\n{reason_lines}\n"
            f"  Passage to rewrite:\n    \"\"\"\n{passage}\n    \"\"\""
        )
    return "\n\n".join(blocks)


def rewrite_weak_segments(
    script_text: str,
    heatmap: Sequence[MinuteHeatmapEntry],
    critique: SceneCritique,
    *,
    canon: list[dict[str, Any]] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_segments: int | None = None,
    min_dropped_personas: int = DEFAULT_MIN_DROPPED_PERSONAS,
    words_per_minute: int = DEFAULT_WORDS_PER_MINUTE,
    model: str = MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    client: OpenAI | None = None,
) -> RewriteResult | None:
    """
    Rewrite the passages where the synthetic audience disengaged.

    Returns None when no minute falls below `threshold`, so callers can skip
    the rewrite entirely rather than burning a request on a healthy script.
    """
    script_text = script_text.strip()
    if not script_text:
        raise ValueError("script_text must not be empty.")

    weak = select_weak_minutes(
        heatmap,
        threshold=threshold,
        max_segments=max_segments,
        min_dropped_personas=min_dropped_personas,
    )
    if not weak:
        logger.info("No minute flagged against threshold %.2f; skipping rewrite.", threshold)
        return None

    logger.info(
        "Rewriting %d weak minute(s): %s",
        len(weak),
        ", ".join(str(e.minute) for e in weak),
    )

    segments = segment_script_by_minute(script_text, words_per_minute)

    try:
        from .orchestrator import format_canon_warnings
    except ImportError:  # pragma: no cover - direct script execution
        from orchestrator import format_canon_warnings

    user_prompt = USER_TEMPLATE.format(
        canon_block=format_canon_warnings(canon or []),
        director_notes=critique.director_notes,
        editor_notes=critique.editor_notes,
        psychologist_notes=critique.psychologist_notes,
        continuity_critique=critique.continuity_critique,
        weak_block=_format_weak_block(weak, segments),
        script_text=script_text,
    )

    openai_client = client or _build_client()

    try:
        completion = openai_client.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RewriteResult,
        )
    except OpenAIError as exc:
        raise RewriteEngineError(
            f"OpenAI rewrite request failed for model '{model}': {exc}"
        ) from exc

    message = completion.choices[0].message
    if message.refusal:
        raise RewriteEngineError(f"Model refused the rewrite: {message.refusal}")

    result = message.parsed
    if result is None:
        raise RewriteEngineError("Model returned no parsable rewrite.")

    if not result.rewritten_script.strip():
        raise RewriteEngineError("Model returned an empty rewritten script.")

    logger.info("Rewrite complete with %d logged change(s).", len(result.change_log))
    return result
