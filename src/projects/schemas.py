"""
Data model for projects, branches, and the canon fact ledger.

The central idea is that branching and fact supersession are the same problem.
A story part written on a branch, and a fact ceasing to be true, are both
*events* with a branch and a position, and one predicate decides whether any
event is visible from where the writer is standing. See `visibility.py`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAIN_BRANCH_NAME = "Main timeline"

FactKind = Literal[
    "state",
    "location",
    "possession",
    "knowledge",
    "relationship",
    "constraint",
    "open_question",
    "permanent",
]

FACT_KIND_HELP: dict[str, str] = {
    "state": "How something is right now, and could change.",
    "location": "Where a character or object is.",
    "possession": "Who holds or owns something.",
    "knowledge": "What a character knows or believes.",
    "relationship": "How two characters stand with each other.",
    "constraint": "Something that cannot happen yet, and why.",
    "open_question": "A mystery the story has raised but not answered.",
    "permanent": "A death or destruction that should not quietly reverse.",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(BaseModel):
    """One story the writer is working on."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    logline: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A project needs a title.")
        return value.strip()


class Branch(BaseModel):
    """
    One timeline through a project.

    `forked_at` is a position in the *parent's* numbering, and the fork point
    is exclusive: a branch inherits parent parts strictly below it. Positions
    continue rather than restart, so a branch forked at 3 writes its own first
    part at position 3. That shared numbering is what lets one comparison
    order events across a whole lineage.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str
    name: str
    parent_id: str | None = None
    forked_at: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A branch needs a name.")
        return value.strip()


class Segment(BaseModel):
    """A finalised part of the story, at a position on a branch."""

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str
    branch_id: str
    position: int
    title: str = ""
    text: str
    word_count: int = 0
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A story part cannot be empty.")
        return value


class Fact(BaseModel):
    """
    One claim the story has established, with a lifecycle.

    A fact is not simply true or false. It is true *from* the event that
    established it *until* the event that superseded it, and both of those
    events sit on a branch. The same fact can therefore be live on one
    timeline and dead on another, which is precisely what a plot-hole checker
    needs to know and what plain similarity search cannot represent.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    project_id: str

    subject: str
    claim: str
    kind: FactKind = "state"

    established_branch: str
    established_position: int

    # Set when a later part makes this stop being true. Nullable because most
    # facts never get superseded.
    superseded_branch: str | None = None
    superseded_position: int | None = None
    superseded_by: str | None = None

    source_segment_id: str = ""
    quote: str = ""

    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def is_superseded(self) -> bool:
        return self.superseded_position is not None

    @field_validator("subject", "claim")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Facts need a subject and a claim.")
        return value.strip()


# ---------------------------------------------------------------------------
# Structured LLM output
# ---------------------------------------------------------------------------


class ExtractedFact(BaseModel):
    """One claim pulled out of a finalised part by the extractor."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="The single entity this claim is about.")
    claim: str = Field(
        description=(
            "One short declarative sentence, true as of this part. Write it so "
            "it still makes sense on its own, months later, with no other "
            "context. For an open_question, state what is not yet known as a "
            "sentence rather than as a question: 'Who signed the manifest is "
            "not known.' Demanding a declarative sentence with no exception "
            "for these quietly turned every unanswered question into a "
            "statement of some adjacent fact, and the story's open threads "
            "were never recorded at all."
        )
    )
    kind: FactKind = Field(description="Which kind of claim this is.")
    quote: str = Field(
        description="The exact sentence from the text that establishes it, copied verbatim."
    )


class FactExtraction(BaseModel):
    """Everything the extractor found in one part."""

    model_config = ConfigDict(extra="forbid")

    facts: list[ExtractedFact] = Field(
        description="Every claim worth remembering. Omit atmosphere and prose style."
    )


class SupersessionCall(BaseModel):
    """A judgement that a new part ended an existing fact."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(description="The existing fact that stopped being true.")
    reason: str = Field(
        description="One sentence naming what in the new part ended it."
    )
    new_claim: str = Field(
        default="",
        description=(
            "The claim that replaces it, if the new part states one. Empty when "
            "the fact simply ceased to apply."
        ),
    )


class AnswerCall(BaseModel):
    """A judgement that a new part answers an open question."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    answer_quote: str = Field(
        description=(
            "The sentence from the new part that answers it, copied word for "
            "word. Empty when the part does not answer it."
        )
    )
    answered: bool = Field(
        description="True only when the quote above genuinely settles the question."
    )


class AnswersFound(BaseModel):
    """Which open questions a part has settled."""

    model_config = ConfigDict(extra="forbid")

    answers: list[AnswerCall] = Field(default_factory=list)


class Reconciliation(BaseModel):
    """Which previously active facts this part brought to an end."""

    model_config = ConfigDict(extra="forbid")

    superseded: list[SupersessionCall] = Field(
        description=(
            "Only facts the new part genuinely ends. A fact merely mentioned "
            "again, or still true, must not appear here."
        )
    )


# ---------------------------------------------------------------------------
# Continuity findings
# ---------------------------------------------------------------------------

FindingKind = Literal[
    "contradiction",
    "premature_reference",
    "dangling_question",
    "unasked_answer",
]

FINDING_LABELS: dict[str, str] = {
    "contradiction": "Contradicts the story so far",
    "premature_reference": "Happens before it should",
    "dangling_question": "Still unanswered",
    "unasked_answer": "Answers something never asked",
}


class ContinuityFinding(BaseModel):
    """
    One problem found in a draft, with the evidence to judge it.

    Always carries the part number and the original line, because a writer
    should be able to overrule the tool in one glance rather than take its
    word for it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: FindingKind
    severity: Literal["high", "medium", "low"] = "medium"
    what: str = Field(description="What the draft does, in the writer's terms.")
    established: str = Field(default="", description="The canon claim it runs into.")
    established_part: int | None = None
    quote: str = Field(default="", description="The original line from that part.")
    suggestion: str = Field(default="", description="One concrete way to resolve it.")
    fact_id: str = ""

    @property
    def label(self) -> str:
        return FINDING_LABELS[self.kind]


class ContinuityReport(BaseModel):
    """Everything the story check found for one draft."""

    model_config = ConfigDict(extra="forbid")

    findings: list[ContinuityFinding] = Field(default_factory=list)
    facts_checked: int = 0
    branch_name: str = ""
    position: int = 0
    llm_used: bool = False
    note: str = ""

    @property
    def is_clean(self) -> bool:
        return not self.findings

    def by_kind(self, kind: FindingKind) -> list[ContinuityFinding]:
        return [f for f in self.findings if f.kind == kind]


class ContradictionCall(BaseModel):
    """The adjudicator's verdict on one candidate contradiction."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    fact_is_a_standing_condition: bool = Field(
        description=(
            "True when the fact describes how the world stands until something "
            "changes it: a door is locked, a character cannot swim, an object "
            "is the only one of its kind. False when it merely records "
            "something that happened at an earlier moment: a character climbed "
            "the stairs, a boat sailed, someone said a line. Things that "
            "happened do not stay true in a way a later scene can violate."
        )
    )
    draft_depicts_the_change: bool = Field(
        description=(
            "True when the draft shows the world changing out of the fact's "
            "state on the page *at or before* the would-be clash: the character "
            "is given the object, unlocks the door, is told the secret. Order "
            "inside the draft matters — unlocking a hold after someone already "
            "described its sealed interior does not count as depicting the "
            "change in time. A story may change what it established; doing so "
            "openly before relying on the new state is the story working. If "
            "the sentence you would copy is itself the moment of change — she "
            "takes the watch off and puts it into his hand — then this is true."
        )
    )
    sentence_showing_the_change: str = Field(
        default="",
        description=(
            "When draft_depicts_the_change is true, copy the sentence from the "
            "draft that performs the change (the unlock, the handoff, the "
            "confession). Leave empty when draft_depicts_the_change is false. "
            "If the clash sentence and the change sentence are the same line, "
            "copy it here as well."
        ),
    )
    sentence_copied_from_the_draft: str = Field(
        description=(
            "A sentence taken from between --- DRAFT --- and --- END DRAFT ---, "
            "copied word for word, which cannot be true alongside the fact. "
            "Never copy from the list of established facts below the draft: "
            "that text is not the draft, and quoting it back proves nothing. "
            "Leave this empty when the draft contains no such sentence, which "
            "is the ordinary case."
        )
    )
    contradicts: bool = Field(
        description=(
            "True only when the draft needs the fact to have been false before "
            "the scene began, and never accounts for it."
        )
    )
    what: str = Field(default="", description="What the draft does that clashes.")
    suggestion: str = Field(default="", description="One concrete fix.")


class ContradictionVerdicts(BaseModel):
    """Verdicts for every candidate the checker was given."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[ContradictionCall] = Field(default_factory=list)
