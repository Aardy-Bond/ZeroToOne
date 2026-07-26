"""
Split a story into passages worth embedding.

Handles both screenplay and prose, because a writer pasting "the story so far"
may bring either. Chunks are built on natural boundaries -- scene headings,
then paragraphs -- rather than a fixed character window, so a retrieved passage
reads as a coherent moment instead of a fragment cut mid-sentence.
"""

from __future__ import annotations

import re

SCENE_HEADING = re.compile(
    r"^\s*(INT\.|EXT\.|INT\./EXT\.|I/E\.|SCENE\s+\d+|CHAPTER\s+[\dIVXLC]+|PART\s+[\dIVXLC]+)",
    re.IGNORECASE,
)
CHARACTER_CUE = re.compile(r"^\s*([A-Z][A-Z0-9 .'\-]{1,30})(\s*\(.*?\))?\s*$")

TARGET_WORDS = 140
MAX_WORDS = 260
MIN_WORDS = 25

NON_CHARACTER_CUES = {
    "CUT TO",
    "SMASH TO BLACK",
    "FADE IN",
    "FADE OUT",
    "FADE TO BLACK",
    "CONTINUED",
    "THE END",
    "MONTAGE",
    "INTERCUT",
}


def _is_character_cue(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped.split()) > 5:
        return False
    if SCENE_HEADING.match(stripped):
        return False
    base = re.sub(r"\(.*?\)", "", stripped).strip().rstrip(":")
    if base.upper() in NON_CHARACTER_CUES or base.endswith(":"):
        return False
    return bool(CHARACTER_CUE.match(stripped)) and base.upper() == base


def dominant_character(text: str) -> str:
    """
    Who this passage belongs to, for attribution on retrieval.

    Screenplay cues are reliable, so they win. Prose falls back to the most
    frequently named capitalised word, which is rough but good enough for a
    label the writer reads rather than a key anything joins on.
    """
    cues = [
        re.sub(r"\(.*?\)", "", line).strip().rstrip(":")
        for line in text.splitlines()
        if _is_character_cue(line)
    ]
    if cues:
        counts: dict[str, int] = {}
        for cue in cues:
            counts[cue] = counts.get(cue, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0].title()

    words = re.findall(r"\b([A-Z][a-z]{2,})\b", text)
    ignore = {"The", "And", "But", "She", "His", "Her", "They", "That", "This", "Then"}
    names = [w for w in words if w not in ignore]
    if names:
        counts = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        best, hits = max(counts.items(), key=lambda kv: kv[1])
        if hits >= 2:
            return best

    return "Narration"


def _blocks(text: str) -> list[str]:
    """Break on scene headings first, then blank lines."""
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        if SCENE_HEADING.match(line) and current:
            blocks.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    out: list[str] = []
    for block in blocks:
        if len(block.split()) <= MAX_WORDS:
            out.append(block)
            continue
        for paragraph in re.split(r"\n\s*\n", block):
            if paragraph.strip():
                out.append(paragraph.strip())

    return [b for b in out if b.strip()]


def chunk_story(text: str) -> list[tuple[str, str]]:
    """
    Split into (character_id, passage) pairs.

    Small adjacent blocks are merged up towards `TARGET_WORDS` so a two-line
    exchange is not embedded on its own, where it would have no context to
    match against.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffered_words = 0

    for block in _blocks(text):
        words = len(block.split())

        if words >= MAX_WORDS:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffered_words = [], 0
            chunks.extend(_split_long(block))
            continue

        buffer.append(block)
        buffered_words += words

        if buffered_words >= TARGET_WORDS:
            chunks.append("\n\n".join(buffer))
            buffer, buffered_words = [], 0

    if buffer:
        tail = "\n\n".join(buffer)
        # A stub tail belongs with the passage before it, not alone.
        if chunks and len(tail.split()) < MIN_WORDS:
            chunks[-1] = chunks[-1] + "\n\n" + tail
        else:
            chunks.append(tail)

    return [(dominant_character(c), c.strip()) for c in chunks if c.strip()]


def _split_long(block: str) -> list[str]:
    """Break an oversized block on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", block)
    out: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        words = len(sentence.split())
        if count + words > TARGET_WORDS and current:
            out.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words

    if current:
        out.append(" ".join(current))

    return [c.strip() for c in out if c.strip()]


def split_into_parts(text: str, target_words: int = 900) -> list[str]:
    """
    Divide a pasted back-catalogue into story parts.

    Used when a writer brings an existing story and wants it as a sequence of
    positions rather than one enormous part zero, so branching has somewhere
    meaningful to fork from.
    """
    text = (text or "").strip()
    if not text:
        return []

    blocks = _blocks(text)
    parts: list[str] = []
    buffer: list[str] = []
    count = 0

    for block in blocks:
        buffer.append(block)
        count += len(block.split())
        if count >= target_words:
            parts.append("\n\n".join(buffer))
            buffer, count = [], 0

    if buffer:
        tail = "\n\n".join(buffer)
        if parts and count < target_words * 0.35:
            parts[-1] = parts[-1] + "\n\n" + tail
        else:
            parts.append(tail)

    return parts
