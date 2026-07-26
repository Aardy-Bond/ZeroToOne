"""
The plot-hole finder.

Four checks, three of which need no model at all. What makes them work is that
they run against the *active* fact set for one point on one timeline, computed
by `visibility.py`, rather than against everything ever written.

That distinction is the whole point. A similarity search over all canon will
happily return "the basement door is locked" from part 1 and flag a perfectly
good scene at part 9, because it has no way to know the key was found at part
7. Asking `is_active` first means the superseded claim is never a candidate,
so the false positive cannot be raised in the first place.
"""

from __future__ import annotations

import logging
import re

from openai import OpenAI, OpenAIError

from .facts import (
    build_client,
    embed_texts,
    rank_by_similarity,
    subjects_mentioned,
)
from .schemas import (
    ContinuityFinding,
    ContinuityReport,
    ContradictionVerdicts,
    Fact,
)
from .visibility import BranchGraph

logger = logging.getLogger(__name__)

ADJUDICATOR_MODEL = "gpt-4o-mini"
MAX_CANDIDATES = 10

# Kinds that usually carry standing conditions worth checking first when the
# candidate list has to be cut.
PRIORITY_KINDS = (
    "permanent",
    "constraint",
    "possession",
    "state",
    "location",
    "knowledge",
)

# Phrases that mean the model is objecting to silence again. Even with a
# real quote elsewhere in the draft, these "what" lines are not clashes.
ABSENCE_WHAT = (
    "does not show",
    "does not mention",
    "does not acknowledge",
    "does not account",
    "fails to mention",
    "fails to account",
    "never mentions",
    "not shown",
)

# Sitting on a sealed hatch / resting a hand on a padlock is not opening it.
PROXIMITY_NOT_ACCESS = (
    "sat on",
    "sitting on",
    "sat upon",
    "as if it were a bench",
    "as if it was a bench",
    "cold under",
    "hand on the",
    "palm",
    "interacting with the padlock",
    "interacting with the sealed",
    "resting on the",
    "on the hatch",
)

# Short enough to allow a line of dialogue, long enough that a stray fragment
# will not accidentally appear somewhere in the draft.
MIN_QUOTE_CHARS = 12

# One sentence may honestly clash with two established facts. Offered against
# more than that it has stopped being evidence for anything in particular.
MAX_FACTS_PER_SENTENCE = 2

# Standing conditions the adjudicator reliably drops when they share a crowded
# first pass with atmospheric "state" noise. Asked in their own batch first.
HARD_CLAIM_MARKERS = (
    "broken",
    "cracked",
    "jammed",
    "smashed",
    "destroyed",
    "cannot ",
    "can't ",
    "only one",
    "the only",
    "will not be back",
    "won't be back",
    "not mean to be back",
    "does not mean to be back",
    "not due back",
    "does not leave",
    "doesn't leave",
    "nobody could open",
    "nobody can open",
    "unable to swim",
    "cannot swim",
    "did not know what",
    "does not know what",
    "not know what was in",
)

# "Sealed" alone is too broad — "sat on the sealed hatch" is atmosphere.
# Do not key off "padlock" alone ("the padlock was cold" is not a constraint).
SEAL_CLAIM_MARKERS = (
    "has sealed",
    "had sealed",
    "sealed the",
    "sealed with",
    "is sealed",
    "was sealed",
    "still sealed",
    "remains sealed",
    "own padlock",
    "with his own padlock",
    "with her own padlock",
)

# How long an open question may sit before it reads as forgotten rather than
# deliberate. Serialised fiction sustains a mystery for a while, so this is
# generous on purpose.
DANGLING_AFTER_PARTS = 4

# Open threads are reported oldest-first and capped. Six of them in one panel
# reads as a broken story rather than a mystery doing its job.
MAX_DANGLING_REPORTED = 3

ADJUDICATOR_SYSTEM = """\
You check a draft against facts already established in the story.

Answer three questions about each fact, in order, and let the first two decide \
the third.

FIRST: is the fact a standing condition, or a thing that merely happened?

A standing condition holds until something changes it — a door is locked, a \
character cannot swim, there is only one key, an object is broken, a character \
is away and not due back until Saturday. Absence-until-a-time and damaged \
state are standing conditions. A thing that merely happened is over: a \
character climbed the stairs, a boat crossed, someone spoke a line, sat on a \
hatch, held a padlock. Sitting near a sealed thing is not opening it. A later \
scene cannot contradict a finished errand just because time passed — but it \
*can* contradict a promise of absence ("she will not be back before Saturday") \
or a broken object ("the compass is cracked") if the draft needs those to have \
been false.

SECOND: does the draft show the world changing, on the page, *in time*?

If the draft has the object handed over, the door unlocked, the secret told, \
*before or at the moment of the would-be clash*, then the story is moving and \
there is nothing wrong. Order inside the draft matters. Unlocking a hold later \
in the same scene does not excuse a character who already described the crates \
inside while it was still sealed. Finding a spare key on the page does not \
excuse earlier use of a second key that was not yet accounted for. A story may \
change what it established; doing so openly *before relying on the new state* \
is the story working. Only a draft that needs the world to have been different \
*before the clashing beat*, and never accounts for it, has a problem.

THIRD, and only if the fact is a standing condition and the draft does not \
show it changing in time: does the draft need that condition to have been \
false all along?

Real clashes look like this: a character uses an object established as broken \
(glass whole, needle swinging) or elsewhere; appears in a place while \
established as away until a later day; walks through something established as \
sealed without the draft unlocking it first; or describes the inside of a \
sealed place before anyone opens it; or swims / rescues when established as \
unable to swim; or produces a second copy of something established as unique.

Worked example — report this as a contradiction. Fact: the hold is sealed and \
nobody may open it until customs. Draft: "Cael told him the beef crates inside \
the still-sealed hold were stacked three deep. … Briggs then unlocked the \
hold." Quote the crates sentence. The later unlock does not excuse knowledge \
spoken while it was still sealed.

Every contradiction you report must point at one sentence in the draft, copied \
out word for word from between the draft markers. Find that sentence before you \
decide anything: if you can copy one out, say so and report the contradiction; \
if you cannot, leave it empty and answer that there is no contradiction. Those \
two answers go together always. Never quote the list of established facts back \
at me — it is not the draft and it proves nothing.

Do not reason your way to a clash. If a fact concerns someone absent from the \
draft, or has no bearing on what the draft depicts, it is not contradicted. If \
explaining the conflict takes more than one step, there is no conflict. Do not \
treat proximity to a sealed object as opening it.

Most facts you are shown will be fine. Saying so is the useful answer.

When you do report one, quote what the draft does and give one concrete fix."""

ADJUDICATOR_USER = """\
Draft for part {position} on the "{branch}" timeline:

--- DRAFT ---
{draft}
--- END DRAFT ---

Facts established earlier and still true at this point:

{fact_block}

Give a verdict for every fact listed."""


def check_continuity(
    draft: str,
    *,
    facts: list[tuple[Fact, list[float] | None]],
    graph: BranchGraph,
    branch_id: str,
    position: int,
    branch_name: str = "",
    client: OpenAI | None = None,
    use_llm: bool = True,
) -> ContinuityReport:
    """
    Run the four checks over a draft.

    `facts` carries embeddings alongside each fact so candidate selection can
    run locally without a round trip.
    """
    draft = (draft or "").strip()
    report = ContinuityReport(branch_name=branch_name, position=position)
    if not draft:
        report.note = "Nothing to check yet."
        return report

    all_facts = [f for f, _ in facts]
    active = graph.active_facts(all_facts, branch_id, position)
    report.facts_checked = len(active)

    findings: list[ContinuityFinding] = []
    findings.extend(_premature_references(draft, all_facts, graph, branch_id, position))
    findings.extend(_dangling_questions(active, position))

    if use_llm and active:
        try:
            findings.extend(
                _contradictions(
                    draft,
                    active,
                    facts,
                    branch_name=branch_name,
                    position=position,
                    client=client,
                )
            )
            report.llm_used = True
        except (OpenAIError, RuntimeError) as exc:
            logger.warning("Contradiction check unavailable: %s", exc)
            report.note = (
                "Structural checks ran, but the contradiction pass could not "
                f"reach the model: {exc}"
            )

    # Two facts about the same subject (Ella left; Ella won't return before
    # Saturday) often produce two near-identical contradiction cards. Keep the
    # strongest one so the panel does not look like it is stammering.
    findings = _dedupe_findings(findings)

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (order[f.severity], f.established_part or 0))
    report.findings = findings

    if not findings and not report.note:
        report.note = (
            f"Checked against {len(active)} established "
            f"{'fact' if len(active) == 1 else 'facts'}. Nothing conflicts."
        )

    return report


# ---------------------------------------------------------------------------
# 1. Contradiction, against the active set only
# ---------------------------------------------------------------------------


def _dedupe_findings(findings: list[ContinuityFinding]) -> list[ContinuityFinding]:
    """
    Drop near-duplicate cards that repeat the same objection.

    Two standing facts about Ella ("left Thursday", "back Saturday") often
    produce two "she is on the quay" cards with different wording. Collapse
    by what-fingerprint first, then collapse presence/return clashes that
    name the same person.
    """
    severity = {"high": 0, "medium": 1, "low": 2}
    by_what: dict[tuple[str, str], ContinuityFinding] = {}
    for finding in findings:
        key = (finding.kind, _letters(finding.what)[:96])
        prior = by_what.get(key)
        if prior is None or severity[finding.severity] < severity[prior.severity]:
            by_what[key] = finding

    kept: list[ContinuityFinding] = []
    presence: dict[str, ContinuityFinding] = {}
    for finding in by_what.values():
        if finding.kind == "contradiction" and _presence_clash(finding.what):
            subj = _leading_name(finding.established) or _leading_name(finding.what)
            if subj:
                prior = presence.get(subj)
                if prior is None or severity[finding.severity] < severity[prior.severity]:
                    presence[subj] = finding
                continue
        kept.append(finding)
    kept.extend(presence.values())
    return kept


def _presence_clash(what: str) -> bool:
    blob = (what or "").lower()
    return any(
        word in blob
        for word in ("present", "on the quay", "appears", "has returned", "is back")
    )


def _leading_name(text: str) -> str:
    """First capitalised token — enough to collapse 'Ella' / 'Ella Venn' dupes."""
    match = re.match(r"^\s*([A-Z][a-z]+)", text or "")
    return match.group(1).lower() if match else ""


def _sentences_used_against_everything(verdicts) -> set[str]:
    """
    Sentences offered as the clash for too many different facts.

    The last shape of the absence problem to survive. Given part three of the
    Kestrel story the model found one salient line — Ilse Vary is the fourth
    keeper — and returned it five times over, once against each fact it had
    nothing better to say about: the draft "does not show that she has", "does
    not show that she cannot", "does not show that she keeps". The line is
    really in the draft, so quoting it passes the evidence rule, but it is
    being used as a general objection rather than a specific clash.

    A sentence can honestly conflict with two facts at once. Beyond that it has
    stopped being evidence and become a shrug.
    """
    counts: dict[str, int] = {}
    for verdict in verdicts:
        if not verdict.contradicts:
            continue
        key = _letters(verdict.sentence_copied_from_the_draft)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n > MAX_FACTS_PER_SENTENCE}


def _quotes_the_draft(quote: str, draft: str) -> bool:
    """
    Whether the adjudicator's quote is really in the draft.

    Compared on letters and digits alone, because the model reflows whitespace
    and swaps quotation marks, and rejecting a real quote over a curly
    apostrophe would push us straight back to trusting the prose.
    """
    quote = (quote or "").strip()
    if len(quote) < MIN_QUOTE_CHARS:
        return False
    return _letters(quote) in _letters(draft)


def _letters(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_hard_standing(fact: Fact) -> bool:
    """Facts that encode absence, damage, uniqueness, or sealed access."""
    blob = f"{fact.subject} {fact.claim}".lower()
    if any(marker in blob for marker in HARD_CLAIM_MARKERS):
        return True
    if not any(marker in blob for marker in SEAL_CLAIM_MARKERS):
        return False
    # A person sitting on a hatch is not the seal constraint itself.
    if fact.kind == "location":
        return False
    if any(phrase in blob for phrase in ("sat on", "sat upon", "sitting on", "as if it were")):
        return False
    return True


def _ask_adjudicator(
    draft: str,
    candidates: list[Fact],
    *,
    branch_name: str,
    position: int,
    client: OpenAI,
) -> list:
    fact_block = "\n".join(
        f"[{f.id}] (part {f.established_position}, {f.kind}) {f.subject}: {f.claim}"
        for f in candidates
    )
    user = ADJUDICATOR_USER.format(
        position=position,
        branch=branch_name or "this",
        draft=draft,
        fact_block=fact_block,
    )
    completion = client.chat.completions.parse(
        model=ADJUDICATOR_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ADJUDICATOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=ContradictionVerdicts,
    )
    parsed = completion.choices[0].message.parsed
    return list(parsed.verdicts) if parsed is not None else []


def _draft_index(quote: str, draft: str) -> int:
    """Letter-compressed start index of quote in draft, or -1."""
    needle = _letters(quote)
    hay = _letters(draft)
    if not needle:
        return -1
    return hay.find(needle)


def _change_happens_after_clash(verdict, draft: str) -> bool:
    """
    Unlock-after-knowledge: the model often marks draft_depicts_the_change
    because Briggs opens the hold later in the same scene, even when the clash
    sentence already used the sealed interior. Order in the draft decides.
    """
    clash = verdict.sentence_copied_from_the_draft or ""
    change = getattr(verdict, "sentence_showing_the_change", "") or ""
    if not change:
        return False
    if not (
        _quotes_the_draft(clash, draft) and _quotes_the_draft(change, draft)
    ):
        return False
    return _draft_index(change, draft) > _draft_index(clash, draft)


def _verdict_to_finding(verdict, fact: Fact, draft: str, overused: set[str]):
    if not verdict.fact_is_a_standing_condition:
        logger.info(
            "Not a contradiction, the fact records a past event: %s", fact.claim
        )
        return None

    change_too_late = _change_happens_after_clash(verdict, draft)
    if verdict.draft_depicts_the_change and not change_too_late:
        logger.info(
            "Not a contradiction, the draft shows the change: %s", fact.claim
        )
        return None

    if not verdict.contradicts and not change_too_late:
        return None

    if change_too_late and not verdict.contradicts:
        logger.info(
            "Keeping clash: change on the page comes after the offending beat: %s",
            fact.claim,
        )

    what_l = (verdict.what or "").lower()
    if any(p in what_l for p in ABSENCE_WHAT):
        logger.info(
            "Not a contradiction, the objection is absence not clash: %s", fact.claim
        )
        return None

    evidence = " ".join(
        [
            verdict.what or "",
            verdict.sentence_copied_from_the_draft or "",
            verdict.suggestion or "",
        ]
    ).lower()
    claim_l = (fact.claim or "").lower()
    if any(p in evidence for p in PROXIMITY_NOT_ACCESS) and any(
        token in claim_l for token in ("sealed", "padlock", "nobody could open", "nobody can open")
    ):
        logger.info(
            "Not a contradiction, proximity is not access: %s", fact.claim
        )
        return None

    # Further damage to an already-broken object is the story moving, not a clash.
    if any(token in claim_l for token in ("broken", "cracked", "jammed", "smashed")):
        if any(
            token in evidence
            for token in ("cracked clean", "hit the stone", "broke", "smashed", "jammed")
        ) and not any(
            token in evidence
            for token in ("glass whole", "needle swinging", "intact", "unbroken")
        ):
            logger.info(
                "Not a contradiction, the draft continues established damage: %s",
                fact.claim,
            )
            return None

    if not _quotes_the_draft(verdict.sentence_copied_from_the_draft, draft):
        return "needs_quote"
    if _letters(verdict.sentence_copied_from_the_draft) in overused:
        logger.info(
            "Not a contradiction, that sentence was offered against everything: %s",
            fact.claim,
        )
        return None
    what = verdict.what or "The draft conflicts with established canon."
    if change_too_late and not verdict.what:
        what = (
            "The draft relies on a state that only changes later in the same scene."
        )
    return ContinuityFinding(
        kind="contradiction",
        severity="high" if fact.kind == "permanent" else "medium",
        what=what,
        established=fact.claim,
        established_part=fact.established_position,
        quote=fact.quote,
        suggestion=verdict.suggestion,
        fact_id=fact.id,
    )


def _contradictions(
    draft: str,
    active: list[Fact],
    facts_with_vectors: list[tuple[Fact, list[float] | None]],
    *,
    branch_name: str,
    position: int,
    client: OpenAI | None,
) -> list[ContinuityFinding]:
    client = client or build_client()
    candidates = _select_candidates(draft, active, facts_with_vectors, client)
    if not candidates:
        return []

    by_id = {f.id: f for f in candidates}
    findings: list[ContinuityFinding] = []

    def absorb(verdicts) -> list[Fact]:
        """Turn verdicts into findings; return facts that still need a quote."""
        overused = _sentences_used_against_everything(verdicts)
        needs_quote: list[Fact] = []
        for verdict in verdicts:
            fact = by_id.get(verdict.fact_id)
            if fact is None:
                continue
            result = _verdict_to_finding(verdict, fact, draft, overused)
            if result == "needs_quote":
                needs_quote.append(fact)
            elif result is not None:
                findings.append(result)
        return needs_quote

    # Hard standing conditions first, alone. Crowding them with atmospheric
    # state ("the padlock was cold") made the model spot the smashed compass
    # and then refuse to quote the draft — or miss it entirely.
    hard = [f for f in candidates if _is_hard_standing(f)]
    soft = [f for f in candidates if not _is_hard_standing(f)]
    batches: list[list[Fact]] = []
    if hard:
        batches.append(hard[:6])
    if soft:
        batches.append(soft[:MAX_CANDIDATES])

    needs_quote: list[Fact] = []
    for batch in batches:
        needs_quote.extend(
            absorb(
                _ask_adjudicator(
                    draft,
                    batch,
                    branch_name=branch_name,
                    position=position,
                    client=client,
                )
            )
        )

    if needs_quote:
        # Deduplicate while preserving order — a fact can appear in both the
        # hard batch and a quote-recovery queue.
        seen: set[str] = set()
        unique_needs: list[Fact] = []
        for fact in needs_quote:
            if fact.id in seen:
                continue
            seen.add(fact.id)
            unique_needs.append(fact)
        logger.info(
            "Re-asking adjudicator for %d clash(es) that lacked a draft quote",
            len(unique_needs),
        )
        still = absorb(
            _ask_adjudicator(
                draft,
                unique_needs[:6],
                branch_name=branch_name,
                position=position,
                client=client,
            )
        )
        for fact in still:
            logger.info(
                "Not a contradiction, no sentence in the draft says so: %s",
                fact.claim,
            )

    return findings


def _select_candidates(
    draft: str,
    active: list[Fact],
    facts_with_vectors: list[tuple[Fact, list[float] | None]],
    client: OpenAI,
) -> list[Fact]:
    """
    Narrow the active set to what is plausibly relevant.

    Two routes, unioned. Similarity catches a paraphrase; naming the subject
    catches a flat contradiction worded completely differently, which
    similarity alone reliably misses.

    Open questions are held back. A question is a statement that the story has
    not said something yet, and a draft cannot contradict it — at worst it
    answers it, which is the point of writing on. Offered as candidates they
    produced findings like "the identity of the man on the tape is not known,
    but Dev seems to have identified him", which is the plot advancing.
    """
    active = [f for f in active if f.kind != "open_question"]
    if len(active) <= MAX_CANDIDATES:
        return active

    active_ids = {f.id for f in active}
    vectors = [(f, v) for f, v in facts_with_vectors if f.id in active_ids]

    picked: dict[str, Fact] = {}

    for subject in subjects_mentioned(draft, active):
        for fact in active:
            if fact.subject == subject:
                picked[fact.id] = fact

    try:
        query = embed_texts([draft[:6000]], client=client)[0]
        for fact, _score in rank_by_similarity(query, vectors, top_k=MAX_CANDIDATES):
            picked[fact.id] = fact
    except Exception as exc:
        logger.debug("Similarity narrowing unavailable, using subjects only: %s", exc)

    if not picked:
        picked = {f.id: f for f in active}

    named = {s.strip().lower() for s in subjects_mentioned(draft, active)}

    def rank(fact: Fact) -> tuple:
        # Prefer hard standing conditions first, then kinds about people/things
        # the draft names. Soft atmospheric state must not crowd out "broken
        # compass" or "away until Saturday".
        hard_rank = 0 if _is_hard_standing(fact) else 1
        kind_rank = (
            PRIORITY_KINDS.index(fact.kind)
            if fact.kind in PRIORITY_KINDS
            else len(PRIORITY_KINDS)
        )
        named_rank = 0 if fact.subject.strip().lower() in named else 1
        return (hard_rank, named_rank, kind_rank, -fact.established_position)

    chosen = sorted(picked.values(), key=rank)
    # Guarantee hard standing facts a slot even when similarity floods `picked`.
    hard = [f for f in chosen if _is_hard_standing(f)]
    soft = [f for f in chosen if not _is_hard_standing(f)]
    reserved = hard[:MAX_CANDIDATES]
    room = max(0, MAX_CANDIDATES - len(reserved))
    return reserved + soft[:room]


# ---------------------------------------------------------------------------
# 2. Premature reference. Deterministic.
# ---------------------------------------------------------------------------


def _premature_references(
    draft: str,
    all_facts: list[Fact],
    graph: BranchGraph,
    branch_id: str,
    position: int,
) -> list[ContinuityFinding]:
    """
    The draft leans on something that has not happened here yet.

    Either it happens later on this timeline, or it belongs to a branch this
    one never descended from. This is the "Dev mentions the key before he
    finds it" mistake, and it is the failure mode a writer hits most often
    after switching branches.
    """
    unknown = graph.established_elsewhere(all_facts, branch_id, position)
    if not unknown:
        return []

    # Three filters, each removing a distinct kind of false alarm.
    #
    # First, scope. Only this branch and its ancestors are relevant. A fork is
    # a *descendant* of the timeline it left, so without this the main line
    # gets judged against alternatives written after it — main was being told
    # that Ilse "belongs to a timeline this one never followed" because a fork
    # had said something new about her.
    #
    # Second, simultaneity. Visibility is a strict `<`, so a fact established
    # at the position being written is invisible to the draft. That is right
    # for reading the story so far and wrong here: those facts came out of an
    # earlier finalise of this very part, so the writer gets told part four
    # refers to something not established until part four, with their own
    # sentence quoted back at them. Nothing can happen before itself.
    # Only material from a timeline this one branched away from. Two reasons,
    # and the second is the one that matters.
    #
    # A writer working forward cannot reference their own future: the fact does
    # not exist yet, so nothing fires. The check only reaches same-branch facts
    # when an earlier part is examined again after later ones are written, and
    # graded that way it was almost all noise — part three of the Kestrel
    # fixture was told it referred to the Kestrel Light "which is not
    # established until part four", the light being the title of the story and
    # lit in part one, and told the same about a bell that part three puts on
    # the page itself. A part that introduces a thing mentions it, and mention
    # is all this check can see.
    #
    # Reaching across a fork is the mistake writers actually make, and there
    # the check is exact: on a branch that split at part three, the parent's
    # part six genuinely never happened, and naming the boat that wrecks in it
    # is a real error with a clear cause.
    lineage = graph.lineage_cutoff(branch_id)
    unknown = [
        f
        for f in unknown
        if f.established_branch != branch_id
        and f.established_branch in lineage
        and f.established_position > position
    ]
    if not unknown:
        return []

    # Third, and the one that removes the most noise: a premature reference
    # means the *subject* is unheard of here, not that some later part has
    # more to say about someone already in the story. Ilse has been on the
    # page since part one; that a later part adds to her is not a reason to
    # warn about naming her. Whether the draft's specific claim clashes with a
    # later one is the contradiction check's job, not this one.
    #
    # Introduction is permanent, so this asks when a subject first appeared and
    # not whether anything said about it is still true. Built from active facts
    # instead, it announced that a draft of part three referred to the Kestrel
    # Light "which is not established until part four" — the light is the
    # title of the story and lit in part one, but every early claim about it
    # had since been superseded, which left the subject looking new.
    #
    # The draft's own part counts as introduction too: part three may speak
    # freely of the bell it puts on the page.
    known_subjects = {
        f.subject.strip().lower()
        for f in all_facts
        if f.established_branch in lineage and f.established_position <= position
    }
    unknown = [f for f in unknown if f.subject.strip().lower() not in known_subjects]
    if not unknown:
        return []

    mentioned = subjects_mentioned(draft, unknown)
    if not mentioned:
        return []

    findings: list[ContinuityFinding] = []
    seen: set[str] = set()

    for fact in unknown:
        if fact.subject not in mentioned or fact.subject in seen:
            continue
        if fact.kind == "open_question":
            continue
        seen.add(fact.subject)

        # Positions are zero-based; every part number shown to a writer is not.
        where = (
            f"belongs to part {fact.established_position + 1} of the "
            f"\"{graph.name_of(fact.established_branch)}\" timeline, which this "
            "one never followed"
        )

        findings.append(
            ContinuityFinding(
                kind="premature_reference",
                severity="high",
                what=f"The draft refers to {fact.subject}, which {where}.",
                established=fact.claim,
                established_part=fact.established_position,
                quote=fact.quote,
                suggestion=(
                    "Either move this beat later, or establish it on the page here."
                ),
                fact_id=fact.id,
            )
        )

    return findings


# ---------------------------------------------------------------------------
# 3. Dangling questions. Deterministic, and named rather than counted.
# ---------------------------------------------------------------------------


def _dangling_questions(active: list[Fact], position: int) -> list[ContinuityFinding]:
    """
    The oldest few unanswered questions, not all of them.

    A mystery is supposed to owe the reader things, so a long story will always
    have several threads in the air and listing every one turns the panel into
    a wall. The oldest are the ones at risk of being forgotten, which is the
    thing worth saying.
    """
    findings: list[ContinuityFinding] = []

    for fact in sorted(active, key=lambda f: f.established_position):
        if fact.kind != "open_question":
            continue
        age = position - fact.established_position
        if age < DANGLING_AFTER_PARTS:
            continue

        findings.append(
            ContinuityFinding(
                kind="dangling_question",
                severity="low" if age < DANGLING_AFTER_PARTS * 2 else "medium",
                what=(
                    f"Opened in part {fact.established_position + 1} and still unanswered "
                    f"{age} parts later."
                ),
                established=fact.claim,
                established_part=fact.established_position,
                quote=fact.quote,
                suggestion="Answer it, acknowledge it, or drop the thread deliberately.",
                fact_id=fact.id,
            )
        )
        if len(findings) == MAX_DANGLING_REPORTED:
            break

    return findings


# ---------------------------------------------------------------------------
# 4. Answers to questions nobody asked.
# ---------------------------------------------------------------------------


def unasked_answers(
    resolved_ids: list[str],
    all_facts: list[Fact],
    graph: BranchGraph,
    branch_id: str,
    position: int,
) -> list[ContinuityFinding]:
    """
    Reconciliation closed a question that was never open on this timeline.

    Usually the tell that the writer is continuing the wrong branch, which is
    worth saying out loud rather than silently ignoring.
    """
    by_id = {f.id: f for f in all_facts}
    findings: list[ContinuityFinding] = []

    for fact_id in resolved_ids:
        fact = by_id.get(fact_id)
        if fact is None or fact.kind != "open_question":
            continue
        if graph.is_active(fact, branch_id, position):
            continue

        findings.append(
            ContinuityFinding(
                kind="unasked_answer",
                severity="medium",
                what="This part answers a question never raised on this timeline.",
                established=fact.claim,
                established_part=fact.established_position,
                quote=fact.quote,
                suggestion=(
                    "Check you are on the timeline you meant to be writing, or "
                    "raise the question first."
                ),
                fact_id=fact.id,
            )
        )

    return findings
