"""
The canon fact ledger: extraction, reconciliation, and local retrieval.

A finalised part is read once and reduced to structured claims. Later parts can
end those claims, which is recorded as a supersession event rather than a
delete, so the ledger can answer "what was true at part 4 on this branch"
instead of only "what is true now".

Facts live in SQLite with their embeddings, and similarity runs locally in
numpy. Three reasons: supersession needs UPDATE, which Delta Sync indexes
handle poorly; there are hundreds of facts rather than millions, so an exact
dot product is both faster and more accurate than an approximate index; and
newly written canon is queryable immediately, with no sync lag between
finalising a part and checking the next one against it.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI, OpenAIError

from .schemas import (
    AnswersFound,
    ExtractedFact,
    Fact,
    FactExtraction,
    Reconciliation,
)
from .store import new_id

logger = logging.getLogger(__name__)

EXTRACT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
EXTRACT_TEMPERATURE = 0.0

# Facts that quietly reverse are the ones that break a story, so deaths and
# destructions are held to a higher bar than ordinary state.
PERMANENT_KINDS = {"permanent"}


class FactError(Exception):
    """Raised when extraction or reconciliation cannot complete."""


EXTRACT_SYSTEM = """\
You are a continuity clerk for a serialised story. You read one part and write \
down what a future writer must not contradict.

Record a claim when it constrains what can happen next:
- state: how something is, and could change ("the generator is out of fuel")
- location: where a character or object is
- possession: who holds something ("Dev has the basement key")
- knowledge: what a character knows or believes, and crucially what they do NOT
- relationship: how two characters stand ("Meera no longer trusts Arjun")
- constraint: something that cannot happen yet, and why ("the door is locked;
  only Aldous has the key")
- open_question: a mystery raised and not answered in this part
- permanent: a death or destruction that should not reverse

Record every question the part raises and leaves unanswered, as open_question. \
Do this even though a mystery does not constrain what happens next and so will \
not feel like a fact. A question asked aloud and not answered, a discovery with \
no explanation, a name or identity the part withholds: each is an open \
question. These are the debts a serialised story owes its reader, and a part \
that opens one and has it go unrecorded is how a thread gets quietly dropped.

Do not record atmosphere, prose style, mood, or a summary of the action. If a \
sentence could be deleted without changing what is possible later, and it \
raises no question, it is not a fact.

Write each claim so it still makes sense alone, months later. Name people and \
places explicitly; never write "he", "she", "it", or "there". Copy the quote \
verbatim from the text.

Prefer a handful of load-bearing claims over an exhaustive list. Open questions \
are the exception: record all of them."""

EXTRACT_USER = """\
Story part {position} on the "{branch}" timeline.

{context_block}
--- THE PART ---
{text}
--- END ---

List the claims a future writer must not contradict, and every question this \
part leaves unanswered."""

RECONCILE_SYSTEM = """\
You decide which established facts a new part has brought to an end.

A fact is superseded only when the new part makes it no longer true:
- a locked door is unlocked
- an object changes hands
- a character learns what they did not know

A fact is NOT superseded when the new part merely mentions it, refers back to \
it, or is consistent with it. Leaving a fact alone is the safe default, and \
the common case. Most parts supersede nothing.

Never supersede a fact of kind "permanent" unless the new part explicitly \
reverses it on the page. Do not infer resurrection.

Return only genuine endings, each with the one sentence from the new part that \
caused it."""

RECONCILE_USER = """\
New part {position} on the "{branch}" timeline:

--- THE PART ---
{text}
--- END ---

Facts currently true at this point in the story:

{fact_block}

Which of these has the new part ended?"""


def build_client(api_key: str | None = None) -> OpenAI:
    import os

    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise FactError("OPENAI_API_KEY is not set.")
    try:
        return OpenAI(api_key=key)
    except OpenAIError as exc:
        raise FactError(f"Could not start the OpenAI client: {exc}") from exc


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_facts(
    text: str,
    *,
    position: int,
    branch_name: str,
    client: OpenAI,
    prior_summary: str = "",
    model: str = EXTRACT_MODEL,
) -> list[ExtractedFact]:
    """Read one finalised part and return the claims worth remembering."""
    text = text.strip()
    if not text:
        return []

    context_block = ""
    if prior_summary.strip():
        context_block = (
            "The story so far, for reference only. Do not extract facts from "
            f"this:\n{prior_summary.strip()}\n\n"
        )

    user = EXTRACT_USER.format(
        position=position,
        branch=branch_name,
        context_block=context_block,
        text=text,
    )

    try:
        completion = client.chat.completions.parse(
            model=model,
            temperature=EXTRACT_TEMPERATURE,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format=FactExtraction,
        )
    except OpenAIError as exc:
        raise FactError(f"Fact extraction failed: {exc}") from exc

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise FactError("Fact extraction returned nothing parsable.")

    return _dedupe(parsed.facts)


QUESTIONS_SYSTEM = """\
You read one part of a serialised story and write down what it asks and does \
not answer.

An open question is anything the part deliberately withholds: a question a \
character asks aloud and nobody answers, a discovery with no explanation, an \
identity left blank, an object whose origin is unaccounted for, a motive that \
does not yet make sense.

Write each one as a sentence rather than as a question, so it reads plainly \
next to the story's other notes: "Who signed the manifest is not known."

Record a question only when the story poses it. Someone asks it, or the \
narration turns and looks at it, or a character is plainly troubled by it. Do \
not record something merely because the part has not explained it yet. A part \
that shows a woman hiding a tin has not asked what the tin is for; it has shown \
her hiding it, and the reader is content. "The significance of the ledger is \
not known" and "the purpose of the tin is not known" are not questions the \
story asked, they are notes on your own understanding, and counting them as \
debts the story owes makes every part look like it is hiding something.

Do not answer them. Do not record a question the part goes on to settle, and \
do not invent tension the text does not raise. If the part genuinely withholds \
nothing, return an empty list; a part that simply moves the plot along is \
allowed to leave nothing open."""

QUESTIONS_USER = """\
Story part {position} on the "{branch}" timeline.

--- THE PART ---
{text}
--- END ---

What does this part ask, or leave unaccounted for, without answering?"""


def extract_open_questions(
    text: str,
    *,
    position: int,
    branch_name: str,
    client: OpenAI,
    model: str = EXTRACT_MODEL,
) -> list[ExtractedFact]:
    """
    Ask, in a call of its own, what the part leaves unanswered.

    This is deliberately not folded into `extract_facts`. That prompt asks for
    claims a future writer must not contradict, and an open question is the
    opposite of a claim — it constrains nothing. Asked for both at once the
    model reliably chose constraints: across the whole six-part Kestrel
    fixture, a mystery in which a character asks a question outright and it is
    underlined twice, not one open question was recorded. Payoff debt sat at
    zero and the dangling-thread check could never fire. Rewording the shared
    prompt moved the behaviour but never made it dependable, because the two
    requests genuinely pull in opposite directions. One extra cheap call buys
    an instruction with nothing competing against it.
    """
    text = text.strip()
    if not text:
        return []

    try:
        completion = client.chat.completions.parse(
            model=model,
            temperature=EXTRACT_TEMPERATURE,
            messages=[
                {"role": "system", "content": QUESTIONS_SYSTEM},
                {
                    "role": "user",
                    "content": QUESTIONS_USER.format(
                        position=position, branch=branch_name, text=text
                    ),
                },
            ],
            response_format=FactExtraction,
        )
    except OpenAIError as exc:
        raise FactError(f"Open-question extraction failed: {exc}") from exc

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return []

    questions = [f.model_copy(update={"kind": "open_question"}) for f in parsed.facts]
    questions = [q for q in questions if not _is_noise_question(q.claim)]
    return _dedupe(questions)


def _is_noise_question(claim: str) -> bool:
    """
    Drop 'questions' that are only the model's curiosity, not the story's.

    Real texts (and even careful fixtures) were sprouting threads like "the
    significance of the compass is not known" whenever an object appeared.
    Those are notes on understanding, not debts the prose opened, and they
    drown the dangling-question panel.
    """
    lowered = claim.strip().lower()
    noise_stems = (
        "significance of",
        "the significance",
        "purpose of",
        "the purpose of",
        "importance of",
        "the importance of",
        "reason for",
        "the reason for",
        "not fully understood",
        "is not clear",
        "is not explained",
    )
    return any(stem in lowered for stem in noise_stems)


def _dedupe(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Drop repeats of the same claim about the same subject."""
    seen: set[tuple[str, str]] = set()
    out: list[ExtractedFact] = []
    for fact in facts:
        key = (fact.subject.strip().lower(), _normalise(fact.claim))
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)
    return out


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def to_facts(
    extracted: list[ExtractedFact],
    *,
    project_id: str,
    branch_id: str,
    position: int,
    segment_id: str,
) -> list[Fact]:
    """Stamp extracted claims with where and when they were established."""
    return [
        Fact(
            id=new_id("fct"),
            project_id=project_id,
            subject=item.subject,
            claim=item.claim,
            kind=item.kind,
            established_branch=branch_id,
            established_position=position,
            source_segment_id=segment_id,
            quote=item.quote.strip(),
        )
        for item in extracted
    ]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _reconcile_candidates(text: str, active: list[Fact], limit: int) -> list[Fact]:
    """
    Show the model what this part is actually about, most relevant first.

    Reconciliation recall falls off badly with the size of the list. Halberd
    Street's part four has engineers repair the lift and run it up and down,
    and asked about that scene against four facts the model retires "the lift
    is out of service" every time; asked against the thirty-odd facts standing
    by then, it sailed past it. The stale fact then survived into part five,
    where a perfectly good scene of two people riding the lift was reported as
    a contradiction — a false positive whose real cause was a missed
    supersession two parts earlier.

    Subjects named in the text come first, because a part can only end a fact
    about something it mentions. The rest fills up with the most recent, since
    a claim made lately is likelier to still be in play.
    """
    if len(active) <= limit:
        return active

    named = {s.strip().lower() for s in subjects_mentioned(text, active)}
    front = [f for f in active if f.subject.strip().lower() in named]
    rest = [f for f in active if f.subject.strip().lower() not in named]
    return (front + rest[::-1])[:limit]


def reconcile(
    text: str,
    active: list[Fact],
    *,
    position: int,
    branch_name: str,
    client: OpenAI,
    model: str = EXTRACT_MODEL,
    max_candidates: int = 24,
) -> Reconciliation:
    """
    Decide which currently-true facts this new part has ended.

    Only facts active at this point are offered, so the model is never asked
    about something that already stopped being true on an earlier part.

    Open questions are withheld and judged by `answered_questions` instead. A
    question ending is a different event from a fact ceasing to be true, and
    asking for both in one call cost the story most of its open threads.
    """
    text = text.strip()
    active = [f for f in active if f.kind != "open_question"]
    if not text or not active:
        return Reconciliation(superseded=[])

    candidates = _reconcile_candidates(text, active, max_candidates)
    fact_block = "\n".join(
        f"[{f.id}] ({f.kind}, part {f.established_position}) {f.subject}: {f.claim}"
        for f in candidates
    )

    user = RECONCILE_USER.format(
        position=position, branch=branch_name, text=text, fact_block=fact_block
    )

    try:
        completion = client.chat.completions.parse(
            model=model,
            temperature=EXTRACT_TEMPERATURE,
            messages=[
                {"role": "system", "content": RECONCILE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format=Reconciliation,
        )
    except OpenAIError as exc:
        raise FactError(f"Reconciliation failed: {exc}") from exc

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return Reconciliation(superseded=[])

    # The model occasionally invents an id, and a permanent fact must never be
    # reversed by inference alone.
    known = {f.id: f for f in candidates}
    kept = []
    for call in parsed.superseded:
        fact = known.get(call.fact_id)
        if fact is None:
            logger.debug("Reconciliation named an unknown fact id: %s", call.fact_id)
            continue
        if fact.kind in PERMANENT_KINDS:
            logger.info(
                "Refusing to supersede permanent fact %s (%s) by inference.",
                fact.id,
                fact.claim,
            )
            continue
        kept.append(call)

    return Reconciliation(superseded=kept)


# ---------------------------------------------------------------------------
# Local embeddings and retrieval
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str], *, client: OpenAI, model: str = EMBED_MODEL, batch_size: int = 96
) -> list[list[float]]:
    """Embed in batches, preserving input order."""
    cleaned = [t.strip() or " " for t in texts]
    if not cleaned:
        return []

    batches = [
        cleaned[i : i + batch_size] for i in range(0, len(cleaned), batch_size)
    ]

    def run(batch: list[str]) -> list[list[float]]:
        try:
            response = client.embeddings.create(model=model, input=batch)
        except OpenAIError as exc:
            raise FactError(f"Embedding failed: {exc}") from exc
        return [item.embedding for item in response.data]

    if len(batches) == 1:
        return run(batches[0])

    with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
        results = list(pool.map(run, batches))

    return [vector for batch in results for vector in batch]


def embed_facts(facts: list[Fact], *, client: OpenAI) -> dict[str, list[float]]:
    """Embed the subject and claim together, which is how they get searched."""
    if not facts:
        return {}
    payload = [f"{f.subject}: {f.claim}" for f in facts]
    vectors = embed_texts(payload, client=client)
    return {fact.id: vector for fact, vector in zip(facts, vectors)}


def rank_by_similarity(
    query_vector: list[float],
    candidates: list[tuple[Fact, list[float] | None]],
    *,
    top_k: int = 12,
    floor: float = 0.15,
) -> list[tuple[Fact, float]]:
    """
    Exact cosine ranking over the candidate facts.

    Brute force is the right call at this scale: a few hundred rows is a
    sub-millisecond matrix multiply, and unlike an approximate index it cannot
    silently miss the one fact that matters.
    """
    usable = [(f, v) for f, v in candidates if v]
    if not usable or not query_vector:
        return []

    matrix = np.asarray([v for _, v in usable], dtype=np.float32)
    query = np.asarray(query_vector, dtype=np.float32)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []

    denominator = np.where(matrix_norms == 0, 1.0, matrix_norms) * query_norm
    scores = (matrix @ query) / denominator

    order = np.argsort(-scores)[:top_k]
    return [
        (usable[i][0], float(scores[i])) for i in order if float(scores[i]) >= floor
    ]


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "of", "to", "in", "on", "at", "for", "with", "and", "or", "but",
    "that", "this", "it", "its", "as", "by", "from", "no", "not", "only",
}

RESTATEMENT_THRESHOLD = 0.6


def _content_tokens(text: str) -> set[str]:
    return {
        word
        for word in re.split(r"\W+", text.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def is_restatement(candidate: Fact, ended: Fact, threshold: float = RESTATEMENT_THRESHOLD) -> bool:
    """
    Whether a newly extracted fact merely restates one the same part just ended.

    This closes a real gap. Reconciliation compares a part against the facts
    that held *before* it, which is correct -- a part must not supersede its own
    claims. But the extractor reads the same part and happily writes down
    background it restates on the way past, so a part that unlocks a door can
    also re-establish "the door is locked" as if it were current.

    The overlap coefficient rather than Jaccard, because a restatement usually
    carries extra detail ("Aldous kept the key" becoming "Aldous, the previous
    owner, kept the key"), and Jaccard punishes exactly that.
    """
    if candidate.subject.strip().lower() != ended.subject.strip().lower():
        return False

    left, right = _content_tokens(candidate.claim), _content_tokens(ended.claim)
    if not left or not right:
        return False

    return len(left & right) / min(len(left), len(right)) >= threshold


def drop_restatements(candidates: list[Fact], ended: list[Fact]) -> tuple[list[Fact], list[Fact]]:
    """Split new facts into those worth keeping and those that restate dead canon."""
    keep: list[Fact] = []
    dropped: list[Fact] = []

    for candidate in candidates:
        if any(is_restatement(candidate, gone) for gone in ended):
            dropped.append(candidate)
        else:
            keep.append(candidate)

    return keep, dropped


# A question restated in different words shares less wording than a restated
# fact does, because the story is usually asking it a second time rather than
# repeating itself. "Somebody called this in three weeks ago, I'd like to know
# who" and "Who called it in, Dev? Three weeks ago" are the same debt.
QUESTION_ECHO_THRESHOLD = 0.45


def drop_asked_again(
    candidates: list[Fact], open_already: list[Fact]
) -> tuple[list[Fact], list[Fact]]:
    """
    Split new open questions into genuinely new ones and ones already open.

    A serialised story asks its central question more than once — that is how
    the reader is reminded a debt is outstanding. Recording each asking as a
    separate thread inflates payoff debt and, worse, reports the same unanswered
    question twice in the panel, which reads as two problems rather than one.
    """
    keep: list[Fact] = []
    echoes: list[Fact] = []

    for candidate in candidates:
        if candidate.kind != "open_question":
            keep.append(candidate)
            continue

        left = _content_tokens(candidate.claim)
        for existing in open_already:
            right = _content_tokens(existing.claim)
            if not left or not right:
                continue
            overlap = len(left & right) / min(len(left), len(right))
            if overlap >= QUESTION_ECHO_THRESHOLD:
                echoes.append(candidate)
                break
        else:
            keep.append(candidate)

    return keep, echoes


ANSWERS_SYSTEM = """\
You decide which of a story's open questions a new part has actually answered.

A question is answered when the part tells the reader the thing the question \
was asking. Not hinted at, not made more interesting, not brought up again: \
answered, so that a reader would no longer wonder.

For each question, first find the sentence in the new part that answers it and \
copy it out word for word. If you cannot find such a sentence, there is no \
answer, and that is the ordinary case. A story keeps most of its questions open \
for a long time, and that is what makes it worth continuing.

A part that repeats the question, or deepens it, has not answered it."""

ANSWERS_USER = """\
New part {position} on the "{branch}" timeline:

--- THE PART ---
{text}
--- END ---

Questions currently open:

{question_block}

Which of these does this part answer, and with which sentence?"""


def answered_questions(
    text: str,
    open_questions: list[Fact],
    *,
    position: int,
    branch_name: str,
    client: OpenAI,
    model: str = EXTRACT_MODEL,
) -> list[tuple[Fact, str]]:
    """
    Which open questions this part settles, each with the line that settles it.

    Separated from `reconcile` because the two judgements are not alike, and
    folding them together lost questions wholesale. Reconciliation asks whether
    a fact has stopped being true, and "an open question is answered" sat in
    that list as one bullet among several. Read that way the model retired
    threads for merely being touched on: Halberd Street's "who called it in?"
    was marked answered by a part that does not mention the call at all, which
    silently zeroed the payoff debt and left the dangling check nothing to
    report.

    The quote is not decoration. It is checked against the part before the
    question is closed, so a question can only be retired by a sentence that
    actually exists.
    """
    if not open_questions:
        return []

    block = "\n".join(f"[{q.id}] {q.claim}" for q in open_questions)
    try:
        completion = client.chat.completions.parse(
            model=model,
            temperature=EXTRACT_TEMPERATURE,
            messages=[
                {"role": "system", "content": ANSWERS_SYSTEM},
                {
                    "role": "user",
                    "content": ANSWERS_USER.format(
                        position=position,
                        branch=branch_name,
                        text=text,
                        question_block=block,
                    ),
                },
            ],
            response_format=AnswersFound,
        )
    except OpenAIError as exc:
        raise FactError(f"Could not check which questions were answered: {exc}") from exc

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return []

    haystack = _normalise(text)
    by_id = {q.id: q for q in open_questions}
    closed: list[tuple[Fact, str]] = []

    for call in parsed.answers:
        question = by_id.get(call.fact_id)
        if question is None or not call.answered:
            continue
        quote = call.answer_quote.strip()
        if not quote:
            continue
        # A question is not an answer to itself. Halberd Street closes its
        # central thread by re-asking it — "Who called it in, Dev? Three weeks
        # ago." — and the model offered that line as the answer.
        if quote.endswith("?"):
            logger.info(
                "Refusing to close %r: the quoted answer is a question.",
                question.claim,
            )
            continue
        if _normalise(quote) not in haystack:
            logger.info(
                "Refusing to close %r: the quoted answer is not in the part.",
                question.claim,
            )
            continue
        closed.append((question, quote))

    return closed


def subjects_mentioned(text: str, facts: list[Fact]) -> set[str]:
    """
    Fact subjects named in the draft, matched on whole words.

    Similarity alone misses a flat contradiction phrased in different words,
    so subject matching runs alongside it and the two are unioned.
    """
    lowered = text.lower()
    hits: set[str] = set()

    for fact in facts:
        subject = fact.subject.strip().lower()
        if len(subject) < 3:
            continue
        if re.search(rf"\b{re.escape(subject)}\b", lowered):
            hits.add(fact.subject)
            continue
        # "the basement door" should also match a draft that says "basement".
        head = [w for w in re.split(r"\W+", subject) if len(w) > 3]
        if head and all(re.search(rf"\b{re.escape(w)}\b", lowered) for w in head):
            hits.add(fact.subject)

    return hits
