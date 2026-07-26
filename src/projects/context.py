"""
The context pack: what the model is told about the story before it helps.

Retrieval alone produces a bag of similar chunks with no beginning, middle, or
end, and a story is nothing but order. So the pack is assembled in layers, each
answering a question similarity cannot:

    1. Rolling synopsis     the arc
    2. The previous part    what literally just happened
    3. Active facts         what is true right now, on this timeline
    4. Similar passages     texture and callbacks, in story order
    5. Open questions       what is owed to the reader

Layer 2 is included unconditionally. The immediately preceding part is the most
important context for writing the next one, and it is exactly what a
similarity search will drop when the new scene changes location.

Layer 4 is re-sorted chronologically after ranking. Similarity decides what is
included; position decides how it reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from openai import OpenAI, OpenAIError

from .facts import EXTRACT_MODEL
from .schemas import Fact, Segment
from .visibility import BranchGraph

logger = logging.getLogger(__name__)

SYNOPSIS_MODEL = EXTRACT_MODEL
MAX_SYNOPSIS_WORDS = 220
VERBATIM_PARTS = 1
MAX_FACTS_IN_PACK = 30


@dataclass
class ContextPack:
    """Everything the model is given, kept inspectable so the writer can audit it."""

    synopsis: str = ""
    previous_parts: list[Segment] = field(default_factory=list)
    active_facts: list[Fact] = field(default_factory=list)
    passages: list = field(default_factory=list)
    open_questions: list[Fact] = field(default_factory=list)
    branch_name: str = ""
    position: int = 0
    passages_available: bool = True

    def to_prompt(self) -> str:
        """Render as an ordered brief rather than a pile of retrieved text."""
        blocks: list[str] = []

        if self.synopsis.strip():
            blocks.append(f"THE STORY SO FAR\n{self.synopsis.strip()}")

        if self.active_facts:
            lines = [
                f"- ({f.kind}, part {f.established_position}) {f.subject}: {f.claim}"
                for f in self.active_facts[:MAX_FACTS_IN_PACK]
            ]
            blocks.append(
                "ESTABLISHED AND STILL TRUE\n"
                "These hold as of this point on this timeline. Do not contradict "
                "them without showing the change on the page.\n" + "\n".join(lines)
            )

        if self.open_questions:
            lines = [
                f"- (opened part {f.established_position}) {f.claim}"
                for f in self.open_questions
            ]
            blocks.append("STILL UNANSWERED\n" + "\n".join(lines))

        if self.passages:
            lines = [
                f"[part {p.position}, {p.character_id}]\n{p.text}"
                for p in self.passages
            ]
            blocks.append(
                "EARLIER MOMENTS THAT MAY MATTER HERE\n"
                "In story order.\n\n" + "\n\n".join(lines)
            )

        if self.previous_parts:
            lines = [
                f"[part {s.position}]\n{s.text}" for s in self.previous_parts
            ]
            blocks.append("THE PART IMMEDIATELY BEFORE THIS ONE\n" + "\n\n".join(lines))

        return "\n\n".join(blocks)

    @property
    def is_empty(self) -> bool:
        return not (
            self.synopsis or self.previous_parts or self.active_facts or self.passages
        )


def build_context(
    *,
    store,
    project_id: str,
    branch_id: str,
    position: int,
    draft: str = "",
    canon_store=None,
    top_passages: int = 5,
) -> ContextPack:
    """
    Assemble the pack for one point on one timeline.

    `canon_store` is optional. When Databricks is unreachable the pack loses
    layer 4 and keeps everything else, so continuity checking and drafting
    still work from local state.
    """
    graph = store.graph(project_id)
    branch = store.get_branch(branch_id)
    branch_name = branch.name if branch else ""

    segments = store.story_so_far(project_id, branch_id, position)
    facts = store.list_facts(project_id)
    active = graph.active_facts(facts, branch_id, position)

    synopsis, _ = store.get_synopsis(branch_id)

    pack = ContextPack(
        synopsis=synopsis,
        previous_parts=segments[-VERBATIM_PARTS:] if segments else [],
        active_facts=[f for f in active if f.kind != "open_question"],
        open_questions=[f for f in active if f.kind == "open_question"],
        branch_name=branch_name,
        position=position,
    )

    if canon_store is not None and draft.strip() and segments:
        try:
            pack.passages = canon_store.search(
                draft,
                project_id=project_id,
                branch_id=branch_id,
                position=position,
                graph=graph,
                top_k=top_passages,
            )
        except Exception as exc:
            logger.info("Passage retrieval unavailable, continuing without it: %s", exc)
            pack.passages_available = False
    else:
        pack.passages_available = canon_store is not None

    # The verbatim tail is already in full, so a retrieved chunk of the same
    # part would just be duplicated text in the prompt.
    if pack.previous_parts:
        recent = {s.position for s in pack.previous_parts}
        pack.passages = [p for p in pack.passages if p.position not in recent]

    return pack


# ---------------------------------------------------------------------------
# Rolling synopsis
# ---------------------------------------------------------------------------

SYNOPSIS_SYSTEM = """\
You maintain a running synopsis of a serialised story, for the writer's own use.

Keep it under {max_words} words. Track the spine: who wants what, what stands \
in the way, what has changed, and where things stand right now.

Write plainly and in the present tense. No blurb language, no praise, no \
speculation about what happens next. If the new part changes something the \
synopsis asserted, correct it rather than appending to it."""

SYNOPSIS_USER = """\
Synopsis up to now:
{existing}

The part just added (part {position}):
{new_part}

Give the updated synopsis."""


def refresh_synopsis(
    *,
    store,
    project_id: str,
    branch_id: str,
    new_part: str,
    position: int,
    client: OpenAI,
    model: str = SYNOPSIS_MODEL,
) -> str:
    """
    Fold a newly finalised part into the branch's running synopsis.

    Rolling rather than regenerated from scratch, so cost stays flat as a
    story grows instead of climbing with every part.
    """
    existing, _ = store.get_synopsis(branch_id)

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": SYNOPSIS_SYSTEM.format(max_words=MAX_SYNOPSIS_WORDS),
                },
                {
                    "role": "user",
                    "content": SYNOPSIS_USER.format(
                        existing=existing.strip() or "(nothing yet, this is the opening)",
                        position=position,
                        new_part=new_part.strip(),
                    ),
                },
            ],
        )
    except OpenAIError as exc:
        logger.warning("Synopsis refresh failed, keeping the previous one: %s", exc)
        return existing

    text = (completion.choices[0].message.content or "").strip()
    if not text:
        return existing

    store.set_synopsis(project_id, branch_id, text, position)
    return text
