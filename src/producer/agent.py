"""
AI Producer agent.

Reads a packed retention context plus numbered speakable lines and returns a
structured ProducerPlan. Casting, selective delivery cues, sound, and a short
marketing brief — all grounded in the forecast scores.
"""

from __future__ import annotations

import logging

from openai import OpenAI, OpenAIError

from projects.facts import build_client
from retention_engine.schemas import ForecastResult

from .context import pack_forecast_context
from .schemas import (
    AllowedVoice,
    CastingChoice,
    MarketingBrief,
    ProducerPlan,
    SpeakableLine,
)
from .script_align import align_plan_to_lines, extract_speakable_lines

logger = logging.getLogger(__name__)

PRODUCER_MODEL = "gpt-4o"

SYSTEM = """\
You are the AI Producer for a serialised audio drama studio (Pocket FM style).

You receive a retention forecast summary (survival proxy, Narrative EKG-style \
scene signals, cliffhanger / Unlock Pull Index, risk scenes) and a numbered \
list of speakable lines. Your job is to direct the performance so the episode \
stays engaging for the stated cohort.

Rules:
- Cast every character that appears. Use only voices from: onyx, nova, echo, \
shimmer, fable, alloy. Narrator (or NARRATOR) should usually be onyx.
- Give one SceneDirection per scene index that appears in the lines.
- Line cues are SELECTIVE. Aim for roughly one cue per two to three lines. \
Always cover: the first speakable line, reveal / high-hazard scenes, and the \
final hook. Leave ordinary connective tissue on the scene default.
- pause_before_ms is silence before that line (0–2000). Use larger pauses for \
reveals and the final hook.
- Sound cues are short labels (door latch, rain bed, heartbeat) attached to \
line indices — not full mix notes.
- MarketingBrief: one logline, up to three hook bullets, a target-listener \
line, and a title treatment. Never claim live listener, unlock, or purchase \
data. The disclaimer field must keep the simulated-forecast honesty language.
- Ground tempo, loudness, and strategy in the hazard / survival / EKG numbers. \
High dread → slower, quieter. High action / elevated hazard mid-episode → \
driving or urgent. Episode-end hook → hold pause, stop clean.
- Do not invent characters missing from the cast list.
- strategy: exactly one sentence."""


def run_producer(
    draft: str,
    forecast: ForecastResult,
    *,
    client: OpenAI | None = None,
    model: str = PRODUCER_MODEL,
) -> tuple[ProducerPlan, list[SpeakableLine]]:
    """
    Build speakable lines, call the model, align cues to real indices.

    Returns the plan and the line list used for display / TTS.
    """
    lines = extract_speakable_lines(draft, forecast)
    if not lines:
        raise ValueError(
            "No speakable lines found. Use screenplay cues (CHARACTER above "
            "dialogue) or prose paragraphs the narrator can read."
        )

    client = client or build_client()
    brief = pack_forecast_context(forecast, lines)

    try:
        completion = client.chat.completions.parse(
            model=model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": brief},
            ],
            response_format=ProducerPlan,
        )
    except OpenAIError as exc:
        raise RuntimeError(f"AI Producer could not reach the model: {exc}") from exc

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("AI Producer returned an empty plan.")

    plan = align_plan_to_lines(parsed, lines)
    plan = _ensure_casting(plan, lines)
    plan = _ensure_marketing_disclaimer(plan)
    return plan, lines


def _ensure_casting(plan: ProducerPlan, lines: list[SpeakableLine]) -> ProducerPlan:
    """Fill any missing cast with stable defaults from the audio voice pool."""
    from audio_engine.synthesizer import NARRATOR_KEY, NARRATOR_VOICE, VOICE_POOL

    have = {c.character.upper(): c for c in plan.casting}
    casting = list(plan.casting)
    pool_i = 0
    for line in lines:
        key = line.character.upper()
        if key in have:
            continue
        if NARRATOR_KEY in key:
            voice: AllowedVoice = NARRATOR_VOICE  # type: ignore[assignment]
        else:
            voice = VOICE_POOL[pool_i % len(VOICE_POOL)]  # type: ignore[assignment]
            pool_i += 1
        choice = CastingChoice(
            character=line.character,
            voice=voice,
            rationale="Default cast — model omitted this speaker.",
        )
        casting.append(choice)
        have[key] = choice
    return plan.model_copy(update={"casting": casting})


def _ensure_marketing_disclaimer(plan: ProducerPlan) -> ProducerPlan:
    from .schemas import PRODUCER_DISCLAIMER

    marketing = plan.marketing
    if PRODUCER_DISCLAIMER.split("—")[0].strip().lower() not in (
        marketing.disclaimer or ""
    ).lower() and "not calibrated" not in (marketing.disclaimer or "").lower():
        marketing = marketing.model_copy(update={"disclaimer": PRODUCER_DISCLAIMER})
    return plan.model_copy(
        update={
            "marketing": marketing,
            "disclaimer": PRODUCER_DISCLAIMER,
        }
    )


def empty_marketing(logline: str = "") -> MarketingBrief:
    from .schemas import PRODUCER_DISCLAIMER

    return MarketingBrief(logline=logline or "Untitled episode.", disclaimer=PRODUCER_DISCLAIMER)
