"""
The workflow layer: one call per thing the writer actually does.

Keeps the UI free of orchestration. Finalising a part is five steps that must
happen in order and partially tolerate failure, which is exactly the kind of
thing that rots when it lives in a Streamlit callback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from openai import OpenAI

from .chunking import chunk_story, split_into_parts
from .context import build_context, refresh_synopsis
from .continuity import check_continuity, unasked_answers
from .facts import (
    answered_questions,
    build_client,
    drop_asked_again,
    drop_restatements,
    embed_facts,
    extract_facts,
    extract_open_questions,
    reconcile,
    to_facts,
)
from .schemas import Branch, ContinuityReport, Fact, Project, Segment
from .store import ProjectStore

logger = logging.getLogger(__name__)


@dataclass
class FinaliseResult:
    """What happened when a part was committed to canon."""

    segment: Segment
    facts_added: int = 0
    facts_superseded: int = 0
    passages_ingested: int = 0
    superseded_claims: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    findings: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        bits = [f"Part {self.segment.position} saved"]
        if self.facts_added:
            bits.append(f"{self.facts_added} new facts")
        if self.facts_superseded:
            bits.append(f"{self.facts_superseded} no longer true")
        return ", ".join(bits) + "."


class StoryService:
    """Everything the writer can do to a project."""

    def __init__(
        self,
        store: ProjectStore,
        openai_client: OpenAI | None = None,
        canon_store=None,
    ) -> None:
        self.store = store
        self._client = openai_client
        self.canon = canon_store

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = build_client()
        return self._client

    # -- creating ----------------------------------------------------------

    def create_project(
        self,
        title: str,
        story_so_far: str = "",
        *,
        logline: str = "",
        split_existing: bool = True,
        progress=None,
    ) -> tuple[Project, Branch]:
        """
        Start a project, optionally seeded with an existing story.

        Existing material is split into parts rather than dropped in as one
        block, so the writer can branch from any point in their own back
        catalogue instead of only from the end.
        """
        project, branch = self.store.create_project(title, logline)

        text = (story_so_far or "").strip()
        if not text:
            return project, branch

        parts = split_into_parts(text) if split_existing else [text]
        for index, part in enumerate(parts):
            if progress:
                progress(index, len(parts), f"Reading part {index + 1} of {len(parts)}")
            self.finalise_part(
                project.id, branch.id, part, extract=True, update_synopsis=True
            )

        if progress:
            progress(len(parts), len(parts), "Ready")

        return project, branch

    def create_branch(
        self, project_id: str, from_branch_id: str, at_position: int, name: str
    ) -> Branch:
        """
        Fork a timeline, carrying the synopsis as it stood at the fork.

        Copying the parent's synopsis is a starting point, not a claim of
        accuracy: it gets rewritten the moment the branch finalises its own
        first part.
        """
        branch = self.store.create_branch(from_branch_id, at_position, name)

        parent_synopsis, _ = self.store.get_synopsis(from_branch_id)
        if parent_synopsis:
            self.store.set_synopsis(
                project_id, branch.id, parent_synopsis, at_position
            )

        return branch

    # -- the main event ----------------------------------------------------

    def finalise_part(
        self,
        project_id: str,
        branch_id: str,
        text: str,
        *,
        title: str = "",
        extract: bool = True,
        update_synopsis: bool = True,
        ingest_passages: bool = True,
    ) -> FinaliseResult:
        """
        Commit a draft to canon.

        Order matters. The part is saved first so nothing is lost if a later
        step fails, then facts are extracted, then reconciled against what was
        true *before* this part, then passages and synopsis. Every step after
        the save degrades to a warning rather than an exception, because
        losing the writer's prose to a failed API call would be unforgivable.
        """
        branch = self.store.get_branch(branch_id)
        if branch is None:
            raise ValueError(f"No such timeline: {branch_id}")

        position = self.store.next_position(project_id, branch_id)
        segment = self.store.add_segment(
            project_id, branch_id, text, title=title, position=position
        )

        result = FinaliseResult(segment=segment)
        if not extract:
            return result

        graph = self.store.graph(project_id)
        all_facts = self.store.list_facts(project_id)
        # Reconcile against what held *before* this part, not including it.
        active_before = graph.active_facts(all_facts, branch_id, position)

        # Reconciliation runs first so extraction knows which claims this part
        # has already killed. Otherwise a part that unlocks a door can also
        # re-establish "the door is locked", since the extractor sees that
        # sentence sitting in the text as background.
        ended: list[Fact] = []
        try:
            call = reconcile(
                text,
                active_before,
                position=position,
                branch_name=branch.name,
                client=self.client,
            )
            for entry in call.superseded:
                self.store.supersede_fact(entry.fact_id, branch_id, position)
                fact = next((f for f in active_before if f.id == entry.fact_id), None)
                if fact is not None:
                    ended.append(fact)
                    result.superseded_claims.append(f"{fact.claim} ({entry.reason})")

            closed = answered_questions(
                text,
                [f for f in active_before if f.kind == "open_question"],
                position=position,
                branch_name=branch.name,
                client=self.client,
            )
            for question, quote in closed:
                self.store.supersede_fact(question.id, branch_id, position)
                ended.append(question)
                result.superseded_claims.append(f"{question.claim} (answered: {quote})")
            result.facts_superseded = len(call.superseded) + len(closed)

            result.findings = unasked_answers(
                [e.fact_id for e in call.superseded],
                all_facts,
                graph,
                branch_id,
                position,
            )
        except Exception as exc:
            logger.warning("Reconciliation failed for part %d: %s", position, exc)
            result.warnings.append(f"Could not update what is still true: {exc}")

        try:
            synopsis, _ = self.store.get_synopsis(branch_id)
            extracted = extract_facts(
                text,
                position=position,
                branch_name=branch.name,
                client=self.client,
                prior_summary=synopsis,
            )
            extracted += extract_open_questions(
                text,
                position=position,
                branch_name=branch.name,
                client=self.client,
            )
            candidates = to_facts(
                extracted,
                project_id=project_id,
                branch_id=branch_id,
                position=position,
                segment_id=segment.id,
            )
            new_facts, restated = drop_restatements(candidates, ended)
            for fact in restated:
                logger.info(
                    "Dropping restatement of a claim this part ended: %s", fact.claim
                )

            # Questions this part just closed stay in the comparison. Halberd
            # Street's part five answers "who called it in?" and asks it again
            # in the same scene, which is ordinary craft; excluding closed
            # questions let the second asking through as a brand-new thread.
            still_open = [f for f in active_before if f.kind == "open_question"]
            new_facts, echoes = drop_asked_again(new_facts, still_open)
            for fact in echoes:
                logger.info("Question already open, not recording again: %s", fact.claim)

            if new_facts:
                vectors = embed_facts(new_facts, client=self.client)
                self.store.add_facts(new_facts, vectors)
                result.facts_added = len(new_facts)
        except Exception as exc:
            logger.warning("Fact extraction failed for part %d: %s", position, exc)
            result.warnings.append(f"Could not read new facts from this part: {exc}")

        if ingest_passages and self.canon is not None:
            try:
                result.passages_ingested = self.canon.ingest_passages(
                    project_id, branch_id, position, chunk_story(text)
                )
            except Exception as exc:
                logger.info("Passage ingest skipped: %s", exc)
                result.warnings.append(
                    "Saved locally. Searchable canon will catch up when "
                    "Databricks is reachable."
                )

        if update_synopsis:
            try:
                refresh_synopsis(
                    store=self.store,
                    project_id=project_id,
                    branch_id=branch_id,
                    new_part=text,
                    position=position,
                    client=self.client,
                )
            except Exception as exc:
                logger.info("Synopsis refresh skipped: %s", exc)

        return result

    # -- checking ----------------------------------------------------------

    def check_draft(
        self,
        project_id: str,
        branch_id: str,
        draft: str,
        *,
        position: int | None = None,
        use_llm: bool = True,
    ) -> ContinuityReport:
        """
        Check a draft against what is true at one point on one timeline.

        `position` matters when the writer is continuing from the middle of a
        branch rather than its end. Defaulting to the tip would check an
        alternative to part 3 against facts established in parts 4 and 5, which
        on the timeline they are about to fork will never have happened.
        """
        graph = self.store.graph(project_id)
        branch = self.store.get_branch(branch_id)
        if position is None:
            position = self.store.next_position(project_id, branch_id)

        return check_continuity(
            draft,
            facts=self.store.list_facts_with_embeddings(project_id),
            graph=graph,
            branch_id=branch_id,
            position=position,
            branch_name=branch.name if branch else "",
            client=self._client,
            use_llm=use_llm,
        )

    def context_for(self, project_id: str, branch_id: str, draft: str = ""):
        return build_context(
            store=self.store,
            project_id=project_id,
            branch_id=branch_id,
            position=self.store.next_position(project_id, branch_id),
            draft=draft,
            canon_store=self.canon,
        )

    # -- reading -----------------------------------------------------------

    def story_text(self, project_id: str, branch_id: str) -> str:
        parts = self.store.story_so_far(project_id, branch_id)
        return "\n\n".join(p.text for p in parts)

    def active_facts(self, project_id: str, branch_id: str, position: int | None = None):
        graph = self.store.graph(project_id)
        if position is None:
            from .visibility import UNBOUNDED

            position = UNBOUNDED
        return graph.active_facts(self.store.list_facts(project_id), branch_id, position)
