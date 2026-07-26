"""
Cliffhanger Lab.

Classifies how an episode ends and scores how hard that ending pulls toward
the next one. Classification is deterministic — Scene DNA plus surface markers
in the closing lines — so it costs nothing extra and can be unit tested.

The Unlock Pull Index is a relative 0-100 score describing narrative pull. It
is not a conversion rate, and it deliberately carries no recommendation about
paywall placement.
"""

from __future__ import annotations

import re

from .schemas import CliffhangerReport, CliffhangerType, SceneAnalysis

# Weights for the Unlock Pull Index. Stakes and information gap dominate
# because an ending that risks nothing, or explains everything, has nothing
# left to pull with.
W_STAKES = 0.28
W_INFORMATION_GAP = 0.24
W_SURPRISE = 0.18
W_EMOTIONAL_INVESTMENT = 0.18
W_NOVELTY = 0.12

# Subtracted after the weighted sum.
W_PAYOFF_DEBT_PENALTY = 0.18
W_FALSE_RESOLUTION_PENALTY = 0.22

RESOLVED_CEILING = 0.25
STRONG_HOOK = 60.0
MODERATE_HOOK = 40.0

# Closing-line markers. Matched against the tail of the final scene only.
TYPE_MARKERS: dict[str, tuple[str, ...]] = {
    "danger": (
        "behind her", "behind him", "trapped", "no way out", "the door slammed",
        "reached for", "grabbed her", "grabbed him", "breathing", "too late",
        "not alone", "in the room",
    ),
    "revelation": (
        "isn't who", "is not who", "the truth", "realised", "realized",
        "it was her", "it was him", "all along", "his own name", "her own name",
        "recognised", "recognized",
    ),
    "betrayal": (
        "betrayed", "on the contract", "working for", "sold her out",
        "sold him out", "had lied", "been lying", "trusted",
    ),
    "decision": (
        "had to choose", "or she would", "or he would", "which one",
        "she had to decide", "he had to decide", "either way",
    ),
    "countdown": (
        "hours", "minutes", "seconds", "midnight", "deadline", "by morning",
        "before dawn", "running out", "twenty-four",
    ),
    "disappearance": (
        "was gone", "were gone", "vanished", "empty bed", "no sign of",
        "wasn't there", "was not there", "disappeared",
    ),
    "false_resolution": (
        "smiled in relief", "it was over", "finally safe", "then her phone",
        "then his phone", "one new message", "and then", "until",
    ),
}

TAIL_SENTENCES = 6


def _tail(text: str, sentences: int = TAIL_SENTENCES) -> str:
    parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return " ".join(parts[-sentences:]).lower()


def classify_ending(analysis: SceneAnalysis) -> list[CliffhangerType]:
    """
    Identify the ending's type, or types — compound endings are common.

    Falls back to Scene DNA when no surface marker matches, so an ending
    phrased in a way the marker list does not cover is still classified.
    """
    dna = analysis.dna
    tail = _tail(analysis.scene.text)

    if dna.cliffhanger_strength < RESOLVED_CEILING:
        return ["weak_resolved"]

    matched: list[CliffhangerType] = []
    for label, markers in TYPE_MARKERS.items():
        if any(marker in tail for marker in markers):
            matched.append(label)  # type: ignore[arg-type]

    # A false resolution is only that if something genuinely reopens. Without
    # surprise it is just a calm ending that happens to use the word "over".
    if "false_resolution" in matched and dna.surprise_factor < 0.4:
        matched.remove("false_resolution")

    if matched:
        # Compound endings are real, but three-plus labels is marker noise.
        return matched[:2]

    if dna.violence_intensity > 0.4 or dna.dread > 0.6:
        return ["danger"]
    if dna.new_information_revealed and dna.surprise_factor > 0.5:
        return ["revelation"]
    if dna.mystery_questions_opened > 0:
        return ["revelation"]
    return ["weak_resolved"]


def _novelty(final: SceneAnalysis, earlier: list[SceneAnalysis]) -> float:
    """
    How fresh this ending is against the hooks that came before it.

    An episode whose every scene ends at the same intensity has trained the
    listener to expect it, and the final hook lands softer as a result.
    """
    if not earlier:
        return 0.7

    previous = [a.dna.cliffhanger_strength for a in earlier]
    mean_previous = sum(previous) / len(previous)
    lift = final.dna.cliffhanger_strength - mean_previous

    # Reward standing out from the episode's own baseline.
    novelty = 0.5 + lift
    tropes = set(final.dna.tropes_present)
    repeated = sum(1 for a in earlier if tropes & set(a.dna.tropes_present))
    if repeated:
        novelty -= 0.10 * min(3, repeated)

    return max(0.0, min(1.0, novelty))


def analyse_ending(analyses: list[SceneAnalysis]) -> CliffhangerReport:
    """Score the episode's final scene as a next-episode hook."""
    if not analyses:
        raise ValueError("analyses must not be empty.")

    final = analyses[-1]
    earlier = analyses[:-1]
    dna = final.dna
    types = classify_ending(final)

    stakes = dna.stakes_level

    # An ending that answers more than it asks leaves nothing hanging.
    opened = dna.mystery_questions_opened
    answered = dna.mystery_questions_answered
    information_gap = max(0.0, min(1.0, (opened - answered * 0.5) / 3.0))
    if dna.cliffhanger_strength > 0.6:
        information_gap = max(information_gap, dna.cliffhanger_strength * 0.8)

    surprise = dna.surprise_factor
    emotional_investment = max(
        dna.emotional_intensity,
        0.6 if dna.character_development_present else 0.0,
    )
    novelty = _novelty(final, earlier)

    debt = final.structural.payoff_debt
    payoff_debt_penalty = min(1.0, max(0, debt - 2) / 5.0)

    false_resolution_risk = 0.0
    if "false_resolution" in types:
        false_resolution_risk = 0.25
    if "weak_resolved" in types:
        false_resolution_risk = 0.75
    if dna.cliffhanger_strength < 0.4 and answered > opened:
        false_resolution_risk = max(false_resolution_risk, 0.6)

    raw = (
        W_STAKES * stakes
        + W_INFORMATION_GAP * information_gap
        + W_SURPRISE * surprise
        + W_EMOTIONAL_INVESTMENT * emotional_investment
        + W_NOVELTY * novelty
        - W_PAYOFF_DEBT_PENALTY * payoff_debt_penalty
        - W_FALSE_RESOLUTION_PENALTY * false_resolution_risk
    )
    unlock_pull_index = round(max(0.0, min(1.0, raw)) * 100, 1)

    hook_strength = round(
        max(0.0, min(1.0, dna.cliffhanger_strength * (0.6 + 0.4 * stakes))) * 100, 1
    )

    return CliffhangerReport(
        types=types,
        hook_strength=hook_strength,
        stakes=round(stakes, 3),
        information_gap=round(information_gap, 3),
        surprise=round(surprise, 3),
        emotional_investment=round(emotional_investment, 3),
        novelty=round(novelty, 3),
        payoff_debt_penalty=round(payoff_debt_penalty, 3),
        false_resolution_risk=round(false_resolution_risk, 3),
        unlock_pull_index=unlock_pull_index,
        recommendation=_recommend(types, unlock_pull_index, payoff_debt_penalty, novelty),
    )


def _recommend(
    types: list[CliffhangerType],
    index: float,
    debt_penalty: float,
    novelty: float,
) -> str:
    """One actionable sentence. Never a claim about conversion."""
    if "weak_resolved" in types:
        return (
            "The ending resolves too cleanly to create a strong next-episode pull. "
            "Consider closing one beat earlier, before the reaction that settles it."
        )

    if index >= STRONG_HOOK:
        note = (
            "Strong candidate for an episode-end hook: stakes and information gap "
            "are both carrying weight."
        )
        if debt_penalty > 0.4:
            note += (
                " Resolve at least one older thread next episode — accumulated payoff "
                "debt is the main thing holding this score down."
            )
        return note

    if index >= MODERATE_HOOK:
        if novelty < 0.4:
            return (
                "Workable hook, but it repeats the shape of earlier scene endings. "
                "Vary the type — a decision or betrayal would land fresher here."
            )
        return (
            "Workable hook. The clearest lever is widening the information gap: end "
            "before the explanation rather than after it."
        )

    return (
        "Weak next-episode pull. Raise what is at risk in the final beat, or cut to "
        "the unanswered question instead of the character's response to it."
    )
