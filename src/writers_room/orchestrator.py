"""
Multi-agent Writers Room orchestrator for Project Anubhuti.

Runs a draft scene past a simulated panel of four industry experts and
returns their notes as a validated SceneCritique.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

try:
    from .schemas import SceneCritique
except ImportError:  # pragma: no cover - direct script execution
    from schemas import SceneCritique

if TYPE_CHECKING:
    from lore_engine.lore_manager import LoreGraph

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CANON_RESULTS = 3

SYSTEM_PROMPT = """\
You are a panel of four ruthless industry experts running a writers room \
table read. You do not flatter, you do not hedge, and you never praise work \
that does not earn it. Every note must be specific enough to act on today.

Speak in four distinct voices:

DIRECTOR — Visual storytelling. Call out blocking, shot selection, camera \
movement, lighting, and pacing. Name the shots you would actually use. Flag \
any beat that plays flat on camera.

EDITOR — Ruthless compression. Identify dialogue that can be cut outright, \
lines that state what the image already shows, redundant beats, and weak \
scene entrances or exits. Quote the offending lines and say what replaces them.

PSYCHOLOGIST — Emotional truth. Interrogate character motivation, subtext, \
and escalating tension. Point out reactions that are unearned or inconsistent \
with how a real person under this pressure would behave.

SOUND PRODUCER — Sound design. Produce concrete foley cues tied to specific \
moments in the scene. Each cue needs a timestamp (MM:SS) or an explicit scene \
beat reference, plus a precise sound effect. Prefer diegetic, texture-rich \
sounds over generic stings.

CONTINUITY — The user payload may include an [ESTABLISHED CANON WARNINGS] \
section drawn from the series bible. Treat every entry there as immutable \
fact. Compare the scene against it line by line and name any contradiction \
explicitly, quoting both the offending scene text and the canon it breaks. \
If the section is absent or the scene is consistent with it, say so plainly \
instead of inventing a violation.

Return your analysis in the required structured format. Write each set of \
notes as flowing prose in that expert's voice, not as bullet fragments.\
"""

NO_CANON_NOTICE = (
    "[ESTABLISHED CANON WARNINGS]\n"
    "No canon records were retrieved for this scene. Judge continuity only on "
    "internal consistency within the scene itself."
)


class WritersRoomError(Exception):
    """Raised when the Writers Room cannot produce a critique."""


def _build_client(api_key: str | None = None) -> OpenAI:
    load_dotenv()
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise WritersRoomError("Missing required environment variable: OPENAI_API_KEY")

    try:
        return OpenAI(api_key=key)
    except OpenAIError as exc:
        raise WritersRoomError(f"Failed to initialize OpenAI client: {exc}") from exc


def fetch_canon(
    script_text: str,
    *,
    character_id: str | None = None,
    lore_graph: LoreGraph | None = None,
    top_k: int = DEFAULT_CANON_RESULTS,
) -> list[dict[str, Any]]:
    """
    Retrieve established lore relevant to a scene.

    Returns an empty list when the Lore Engine is unreachable, so a Databricks
    outage degrades the critique rather than blocking it.
    """
    graph = lore_graph
    if graph is None:
        try:
            from lore_engine.lore_manager import LoreGraph as _LoreGraph

            graph = _LoreGraph()
        except Exception as exc:
            logger.warning("Lore Engine unavailable, skipping canon lookup: %s", exc)
            return []

    try:
        matches = graph.check_continuity(script_text, top_k=top_k)
    except Exception as exc:
        logger.warning("Canon lookup failed, continuing without it: %s", exc)
        return []

    if character_id:
        scoped = [m for m in matches if m.get("character_id") == character_id]
        if scoped:
            matches = scoped
        else:
            logger.info(
                "No canon matched character_id=%s; using unfiltered results.",
                character_id,
            )

    # The vector index can hold duplicate lore text from repeated ingests.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for match in matches:
        text = str(match.get("lore_text", "")).strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(match)

    logger.info("Retrieved %d canon record(s) for the panel.", len(deduped))
    return deduped


def format_canon_warnings(canon: list[dict[str, Any]]) -> str:
    """Render retrieved canon into the prompt section the panel reads."""
    if not canon:
        return NO_CANON_NOTICE

    lines = ["[ESTABLISHED CANON WARNINGS]"]
    for i, record in enumerate(canon, 1):
        character = record.get("character_id", "unknown")
        text = str(record.get("lore_text", "")).strip()
        score = record.get("score")
        relevance = f" (relevance {score:.3f})" if isinstance(score, float) else ""
        lines.append(f"{i}. [{character}]{relevance} {text}")

    return "\n".join(lines)


def analyze_script(
    script_text: str,
    *,
    character_id: str | None = None,
    lore_graph: LoreGraph | None = None,
    use_lore: bool = True,
    top_k: int = DEFAULT_CANON_RESULTS,
    model: str = MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    client: OpenAI | None = None,
) -> SceneCritique:
    """
    Run a draft scene past the four-expert panel.

    When `use_lore` is set, relevant canon is pulled from the Lore Engine and
    injected so the panel can flag continuity violations. Pass `character_id`
    to bias retrieval toward one character's established facts.

    Returns a validated SceneCritique. Raises WritersRoomError on failure.
    """
    script_text = script_text.strip()
    if not script_text:
        raise ValueError("script_text must not be empty.")

    openai_client = client or _build_client()

    canon: list[dict[str, Any]] = []
    if use_lore:
        canon = fetch_canon(
            script_text,
            character_id=character_id,
            lore_graph=lore_graph,
            top_k=top_k,
        )

    user_payload = (
        f"{format_canon_warnings(canon)}\n\n"
        f"[SCENE UNDER REVIEW]\n{script_text}"
    )

    logger.info("Sending %d chars to %s for panel critique.", len(script_text), model)

    try:
        completion = openai_client.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            response_format=SceneCritique,
        )
    except OpenAIError as exc:
        raise WritersRoomError(f"OpenAI request failed for model '{model}': {exc}") from exc

    message = completion.choices[0].message

    if message.refusal:
        raise WritersRoomError(f"Model refused the request: {message.refusal}")

    critique = message.parsed
    if critique is None:
        raise WritersRoomError("Model returned no parsable critique.")

    logger.info("Panel returned %d foley trigger(s).", len(critique.foley_triggers))
    return critique
