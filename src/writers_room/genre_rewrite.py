"""
Genre / style rewrite engine for Project Anubhuti.

Restyles the current draft into a chosen target (horror, romance, comedy,
thriller, or anime) while holding the core plot and established canon steady.
This is the opposite contract from the weak-minute punch-up in rewrite_engine:
tone is allowed — and expected — to change.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.85

GenreRewriteTarget = Literal[
    "horror", "romance", "comedy", "thriller", "anime"
]

GENRE_REWRITE_TARGETS: tuple[GenreRewriteTarget, ...] = (
    "horror",
    "romance",
    "comedy",
    "thriller",
    "anime",
)

TARGET_GUIDANCE: dict[GenreRewriteTarget, str] = {
    "horror": (
        "Lean into dread, unease, and threat. Prefer implication over gore. "
        "Heighten atmosphere, silence, and what the audience cannot see."
    ),
    "romance": (
        "Centre emotional intimacy, longing, and relational stakes. Soften or "
        "reframe violence into emotional conflict where the plot allows; keep "
        "events that must happen, but colour them through desire and vulnerability."
    ),
    "comedy": (
        "Find wit, irony, and comic timing without undoing the plot. Banter, "
        "misreads, and absurd juxtaposition are fair game; tragic outcomes still "
        "occur if the original plot requires them."
    ),
    "thriller": (
        "Escalate suspense, urgency, and paranoia. Tighten pacing, plant ticking "
        "clocks and reveals, and keep the audience one step behind the danger."
    ),
    "anime": (
        "This is a STYLE target, not a new plot genre. Heighten visual beats, "
        "emotional clarity, and series-episode energy — sharper reactions, "
        "clearer emotional peaks, kinetic scene turns — without inventing "
        "powers, tropes, or plot beats the original does not support."
    ),
}

SYSTEM_PROMPT = """\
You are the showrunner performing a genre / style transfer pass on an audio \
drama script.

Your mandate:

TRANSFORM THE GENRE OR STYLE. Rewrite the full draft so it reads as the \
requested target. Change dialogue register, atmosphere, imagery, pacing, and \
genre conventions as needed. This is not a polish pass and not a punch-up of \
weak minutes — the whole draft may be restyled.

PRESERVE THE CORE PLOT. Keep the same characters (names and identities), the \
same causal order of events, the same revelations and outcomes, and the same \
scene purposes. Do not invent new plot turns, characters, locations, or \
backstory that change what happens. Do not resolve open questions the canon \
says are still unanswered.

PRESERVE CANON. Anything in the story-context block is immutable. Do not \
contradict established facts. If a restyle would clash with a fact, keep the \
fact and find another stylistic route.

HOLD THE RUNTIME. Keep the rewritten script close to the original length. \
Do not pad and do not gut scenes.

Return the complete rewritten script, a concrete change log of what you \
restyled, and a short note on how the plot was preserved.\
"""

USER_TEMPLATE = """\
[TARGET]
{target}

[TARGET GUIDANCE]
{target_guidance}

[STORY CONTEXT — IMMUTABLE CANON]
{context_block}

[FULL SCRIPT TO RESTYLE]
{script_text}\
"""

NO_CONTEXT_NOTICE = (
    "No project canon was available. Preserve only what the script itself "
    "establishes; do not invent contradictory backstory."
)


class GenreRewriteError(Exception):
    """Raised when the genre rewrite pass cannot complete."""


class GenreChange(BaseModel):
    """One stylistic change made during the genre rewrite."""

    model_config = ConfigDict(extra="forbid")

    aspect: str = Field(
        description="What was restyled, e.g. tone, dialogue, imagery, pacing."
    )
    change_made: str = Field(description="What changed, concretely.")


class GenreRewriteResult(BaseModel):
    """Output of a genre / style rewrite pass."""

    model_config = ConfigDict(extra="forbid")

    rewritten_script: str = Field(
        description="The complete script restyled for the target."
    )
    change_log: list[GenreChange] = Field(
        description="Concrete stylistic changes made for the target."
    )
    plot_preservation_note: str = Field(
        description="How core plot beats and canon were preserved."
    )
    target_genre: str = Field(
        description="The requested target genre or style."
    )


def _build_client(api_key: str | None = None) -> OpenAI:
    load_dotenv()
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise GenreRewriteError(
            "Missing required environment variable: OPENAI_API_KEY"
        )
    try:
        return OpenAI(api_key=key)
    except OpenAIError as exc:
        raise GenreRewriteError(
            f"Failed to initialize OpenAI client: {exc}"
        ) from exc


def _normalize_target(target: str) -> GenreRewriteTarget:
    key = (target or "").strip().lower()
    if key not in GENRE_REWRITE_TARGETS:
        raise ValueError(
            f"Unsupported genre rewrite target '{target}'. "
            f"Choose one of: {', '.join(GENRE_REWRITE_TARGETS)}."
        )
    return key  # type: ignore[return-value]


def _format_context(context_text: str | None) -> str:
    text = (context_text or "").strip()
    return text or NO_CONTEXT_NOTICE


def rewrite_as_genre(
    script_text: str,
    target: str,
    *,
    context_pack: str | None = None,
    model: str = MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    client: OpenAI | None = None,
) -> GenreRewriteResult:
    """
    Restyle the full draft into ``target`` while preserving plot and canon.

    ``context_pack`` should be the rendered project context (e.g.
    ``ContextPack.to_prompt()``). When empty or omitted, the model is told
    that no project canon was available.
    """
    script_text = (script_text or "").strip()
    if not script_text:
        raise ValueError("script_text must not be empty.")

    normalized = _normalize_target(target)
    logger.info("Genre-rewriting draft as %s (%d chars).", normalized, len(script_text))

    user_prompt = USER_TEMPLATE.format(
        target=normalized,
        target_guidance=TARGET_GUIDANCE[normalized],
        context_block=_format_context(context_pack),
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
            response_format=GenreRewriteResult,
        )
    except OpenAIError as exc:
        raise GenreRewriteError(
            f"OpenAI genre rewrite request failed for model '{model}': {exc}"
        ) from exc

    message = completion.choices[0].message
    if message.refusal:
        raise GenreRewriteError(f"Model refused the genre rewrite: {message.refusal}")

    result = message.parsed
    if result is None:
        raise GenreRewriteError("Model returned no parsable genre rewrite.")

    if not result.rewritten_script.strip():
        raise GenreRewriteError("Model returned an empty rewritten script.")

    # Ensure the structured field reflects the requested target even if the
    # model drifts.
    result.target_genre = normalized

    logger.info(
        "Genre rewrite as %s complete with %d logged change(s).",
        normalized,
        len(result.change_log),
    )
    return result
