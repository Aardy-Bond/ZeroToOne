"""
Writer-facing vocabulary.

The engine's terms are precise and, to a novelist, meaningless. "Hazard 0.365"
and "Unlock Pull Index 46" describe real quantities that nobody outside this
repository has ever needed. Every one of them is translated here, in one place,
so the surface can speak plainly while the maths underneath stays exactly as
it was and stays visible under "Show the analysis".

Nothing here rounds a number into a claim it cannot support. Plain language is
not the same as a vaguer promise.
"""

from __future__ import annotations

# The analyst term on the left is still what the code computes. Only the label
# changes.
GLOSSARY: dict[str, str] = {
    "Engagement Survival Forecast": "Read-through",
    "engagement survival proxy": "how many readers are still with you",
    "hazard": "risk of losing the reader",
    "Unlock Pull Index": "pull to the next part",
    "cliffhanger strength": "how hard the ending pulls",
    "counterfactual cohort": "another kind of reader",
    "quality prior": "prose strength",
    "payoff debt": "questions still open",
    "Scene DNA": "scene reading",
    "exposition ratio": "how much is explanation",
    "scene tempo": "pace",
    "emotional intensity": "emotional charge",
    "POV dominance": "one character carrying too much",
}

RISK_WORDS: dict[str, str] = {
    "low": "Holding",
    "elevated": "Slipping",
    "high": "Losing them",
}

RISK_TONE: dict[str, str] = {
    "low": "calm",
    "elevated": "",
    "high": "risk",
}


def risk_label(band: str) -> str:
    return RISK_WORDS.get(band, band.title())


def risk_tone(band: str) -> str:
    return RISK_TONE.get(band, "")


def survival_sentence(value: float) -> str:
    """Describe the read-through score without implying it counts real people."""
    if value >= 70:
        return "Most readers should stay with this to the end."
    if value >= 50:
        return "You lose some readers in the middle, but the ending holds."
    if value >= 30:
        return "A meaningful share drift off before the ending lands."
    return "This loses most readers before the end."


def hook_sentence(index: float) -> str:
    if index >= 70:
        return "A strong reason to start the next part immediately."
    if index >= 50:
        return "A decent pull forward, though not urgent."
    if index >= 30:
        return "The ending is soft. A reader could comfortably stop here."
    return "This resolves too cleanly to pull anyone onward."


def prose_sentence(score: float, available: bool) -> str:
    if not available:
        return "Not scored yet."
    if score >= 0.68:
        return "Reads strongly against the reference set."
    if score >= 0.58:
        return "Reads solidly against the reference set."
    return "Reads below the reference set."


def word_count_line(text: str) -> str:
    words = len(text.split())
    if not words:
        return "Empty"
    minutes = words / 150
    return f"{words:,} words · about {minutes:.0f} min read"


def part_label(position: int) -> str:
    return f"Part {position + 1}"


def plural(count: int, singular: str, many: str = "") -> str:
    if count == 1:
        return f"1 {singular}"
    return f"{count} {many or singular + 's'}"


DISCLAIMER = (
    "This is a simulated read-through, based on how the writing is built and "
    "the feelings it carries. It has not been checked against real listeners."
)

CANON_NOTE = (
    "Only what is true on this timeline at this point. Anything a later part "
    "changed, or anything that happened on a different timeline, is left out."
)
