"""
The project store, the fact ledger, and the plot-hole finder.

The scenario throughout is the one that motivated the design: a door
established as locked, a key found later, and a branch that forked before the
key was found. A checker that ignores time flags the wrong thing on both
timelines.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from conftest import FakeOpenAI
from projects.chunking import chunk_story, dominant_character, split_into_parts
from projects.continuity import check_continuity
from projects.facts import (
    _dedupe,
    drop_restatements,
    is_restatement,
    rank_by_similarity,
    reconcile,
    subjects_mentioned,
    to_facts,
)
from projects.schemas import (
    ContradictionCall,
    ContradictionVerdicts,
    ExtractedFact,
    Fact,
    Reconciliation,
    SupersessionCall,
)
from projects.store import ProjectStore


@pytest.fixture
def store(tmp_path):
    s = ProjectStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def seeded(store):
    """A project with four parts on main, and a branch forked at part 2."""
    project, main = store.create_project("The House on Wexler Street")
    for i in range(4):
        store.add_segment(project.id, main.id, f"Part {i} happens on the main line.")
    alt = store.create_branch(main.id, 2, "What if Dev goes first")
    store.add_segment(project.id, alt.id, "Part 2 happens differently.")
    return SimpleNamespace(project=project, main=main, alt=alt, store=store)


def a_fact(store, project_id, branch_id, position, subject, claim, kind="state"):
    fact = Fact(
        id=f"fct_{subject}_{position}".replace(" ", "_"),
        project_id=project_id,
        subject=subject,
        claim=claim,
        kind=kind,
        established_branch=branch_id,
        established_position=position,
        quote=claim,
    )
    store.add_facts([fact])
    return fact


class TestStore:
    def test_a_new_project_gets_a_main_timeline(self, store):
        project, branch = store.create_project("Wexler Street")
        assert branch.is_root
        assert branch.forked_at is None
        assert [b.id for b in store.list_branches(project.id)] == [branch.id]

    def test_positions_increment_on_a_branch(self, store):
        project, main = store.create_project("P")
        first = store.add_segment(project.id, main.id, "One.")
        second = store.add_segment(project.id, main.id, "Two.")
        assert (first.position, second.position) == (0, 1)

    def test_a_fork_starts_numbering_at_the_fork_point(self, seeded):
        nxt = seeded.store.next_position(seeded.project.id, seeded.alt.id)
        assert nxt == 3, "alt forked at 2 and has written part 2, so 3 is next"

    def test_story_so_far_excludes_the_parents_later_parts(self, seeded):
        parts = seeded.store.story_so_far(seeded.project.id, seeded.alt.id)
        assert [(p.branch_id, p.position) for p in parts] == [
            (seeded.main.id, 0),
            (seeded.main.id, 1),
            (seeded.alt.id, 2),
        ]

    def test_the_parent_is_unaffected_by_the_branch(self, seeded):
        parts = seeded.store.story_so_far(seeded.project.id, seeded.main.id)
        assert [p.position for p in parts] == [0, 1, 2, 3]
        assert all(p.branch_id == seeded.main.id for p in parts)

    def test_the_main_timeline_cannot_be_deleted(self, seeded):
        with pytest.raises(ValueError, match="main timeline"):
            seeded.store.delete_branch(seeded.main.id)

    def test_deleting_a_branch_revives_facts_it_had_ended(self, seeded):
        fact = a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            0,
            "basement door",
            "The basement door is locked.",
        )
        seeded.store.supersede_fact(fact.id, seeded.alt.id, 2)
        seeded.store.delete_branch(seeded.alt.id)

        revived = seeded.store.list_facts(seeded.project.id)[0]
        assert not revived.is_superseded, (
            "the event that ended it no longer happened anywhere"
        )

    def test_a_branch_with_children_cannot_be_deleted(self, seeded):
        seeded.store.create_branch(seeded.alt.id, 3, "deeper")
        with pytest.raises(ValueError, match="branches of its own"):
            seeded.store.delete_branch(seeded.alt.id)

    def test_synopsis_round_trips(self, seeded):
        seeded.store.set_synopsis(seeded.project.id, seeded.main.id, "A house.", 3)
        assert seeded.store.get_synopsis(seeded.main.id) == ("A house.", 3)


class TestTheLockedDoorEndToEnd:
    """
    The false positive this whole design exists to prevent.

    Locked at part 1, unlocked at part 2, branch forked at 2.
    """

    @pytest.fixture
    def door(self, seeded):
        fact = a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            1,
            "basement door",
            "The basement door is locked and only Aldous has the key.",
            kind="constraint",
        )
        seeded.store.supersede_fact(fact.id, seeded.main.id, 2)
        return fact

    def test_the_door_is_open_on_main(self, seeded, door):
        graph = seeded.store.graph(seeded.project.id)
        live = graph.active_facts(
            seeded.store.list_facts(seeded.project.id), seeded.main.id, 9
        )
        assert door.id not in {f.id for f in live}

    def test_the_door_is_still_locked_on_the_branch(self, seeded, door):
        graph = seeded.store.graph(seeded.project.id)
        live = graph.active_facts(
            seeded.store.list_facts(seeded.project.id), seeded.alt.id, 9
        )
        assert door.id in {f.id for f in live}, (
            "the branch forked at 2 and never saw the key being found"
        )

    def test_the_checker_stays_silent_on_main(self, seeded, door):
        """No candidates means no LLM call and no finding. The bug, prevented."""
        client = FakeOpenAI([])
        report = check_continuity(
            "Priya walks into the basement and turns on the light.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=9,
            client=client,
            use_llm=True,
        )
        assert report.facts_checked == 0
        assert report.is_clean
        assert client.calls == [], "a superseded fact must never reach the adjudicator"

    def test_the_checker_does_flag_it_on_the_branch(self, seeded, door):
        client = FakeOpenAI(
            [
                ContradictionVerdicts(
                    verdicts=[
                        ContradictionCall(
                            fact_id=door.id,
                            fact_is_a_standing_condition=True,
                            draft_depicts_the_change=False,
                            sentence_copied_from_the_draft="Priya walks into the basement",
                            contradicts=True,
                            what="Priya walks through a door established as locked.",
                            suggestion="Show her finding the key first.",
                        )
                    ]
                )
            ]
        )
        report = check_continuity(
            "Priya walks into the basement and turns on the light.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.alt.id,
            position=9,
            client=client,
            use_llm=True,
        )
        assert report.facts_checked == 1
        contradictions = report.by_kind("contradiction")
        assert len(contradictions) == 1
        assert contradictions[0].established_part == 1


class TestPrematureReference:
    def test_reaching_forward_on_your_own_timeline_says_nothing(self, seeded):
        """
        The check no longer speaks about a single timeline, on purpose.

        Writing forward, a reference to your own future cannot be seen: the
        later fact does not exist yet. The case only arises when an earlier
        part is examined again after the story has moved on, and graded there
        it was wrong far more often than right, because a part that introduces
        a thing also mentions it and mention is all this check can see. Kestrel
        part three was told it referred to the Kestrel Light "which is not
        established until part four".
        """
        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            3,
            "brass key",
            "Dev finds the brass key in the drawer.",
        )
        report = check_continuity(
            "Dev turns the brass key in the lock, as if he had always had it.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=1,
            use_llm=False,
        )
        assert report.by_kind("premature_reference") == []

    def test_a_fork_referring_to_the_parents_later_material_is_flagged(self, seeded):
        """
        The failure a writer actually hits after switching branches.

        `alt` forked from main at part 2, so main's part 4 never happened on
        it. Reaching for that material is the real cross-timeline mistake, and
        it is caught because main is an *ancestor* of the fork.

        This test previously ran the other way round — a fact on `alt`, checked
        from main — and asserted it was flagged. That was the bug: `alt` is a
        descendant of main, so main was being warned about an alternative
        written after it.
        """
        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            3,
            "brass key",
            "Dev finds the brass key.",
        )
        report = check_continuity(
            "Dev turns the brass key in the lock.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.alt.id,
            position=2,
            use_llm=False,
        )
        findings = report.by_kind("premature_reference")
        assert len(findings) == 1
        assert "brass key" in findings[0].what

    def test_a_fact_the_draft_never_mentions_is_not_flagged(self, seeded):
        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            3,
            "brass key",
            "Dev finds the brass key.",
        )
        report = check_continuity(
            "Meera reads a file at the records office.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=1,
            use_llm=False,
        )
        assert not report.by_kind("premature_reference")


class TestDanglingQuestions:
    def test_an_old_open_question_is_named_not_counted(self, seeded):
        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            0,
            "the trough",
            "Who dug the child-sized trough in the concrete?",
            kind="open_question",
        )
        report = check_continuity(
            "Meera drives home.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=6,
            use_llm=False,
        )
        findings = report.by_kind("dangling_question")
        assert len(findings) == 1
        assert "trough" in findings[0].established
        assert findings[0].established_part == 0

    def test_a_recent_question_is_left_alone(self, seeded):
        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            2,
            "the trough",
            "Who dug the trough?",
            kind="open_question",
        )
        report = check_continuity(
            "Meera drives home.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=3,
            use_llm=False,
        )
        assert not report.by_kind("dangling_question")

    def test_an_answered_question_stops_nagging(self, seeded):
        fact = a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            0,
            "the trough",
            "Who dug the trough?",
            kind="open_question",
        )
        seeded.store.supersede_fact(fact.id, seeded.main.id, 2)
        report = check_continuity(
            "Meera drives home.",
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=9,
            use_llm=False,
        )
        assert not report.by_kind("dangling_question")


class TestReconciliation:
    def test_a_permanent_fact_is_never_superseded_by_inference(self):
        dead = Fact(
            id="f_dead",
            project_id="p",
            subject="Aldous",
            claim="Aldous is dead.",
            kind="permanent",
            established_branch="main",
            established_position=1,
        )
        client = FakeOpenAI(
            [
                Reconciliation(
                    superseded=[
                        SupersessionCall(fact_id="f_dead", reason="He speaks again.")
                    ]
                )
            ]
        )
        result = reconcile(
            "Aldous speaks from the corner.",
            [dead],
            position=5,
            branch_name="main",
            client=client,
        )
        assert result.superseded == [], "a death must not reverse without being shown"

    def test_ordinary_facts_can_be_superseded(self):
        door = Fact(
            id="f_door",
            project_id="p",
            subject="basement door",
            claim="The basement door is locked.",
            kind="constraint",
            established_branch="main",
            established_position=1,
        )
        client = FakeOpenAI(
            [
                Reconciliation(
                    superseded=[
                        SupersessionCall(fact_id="f_door", reason="Dev unlocks it.")
                    ]
                )
            ]
        )
        result = reconcile(
            "Dev turns the key and the door swings open.",
            [door],
            position=5,
            branch_name="main",
            client=client,
        )
        assert [c.fact_id for c in result.superseded] == ["f_door"]

    def test_invented_fact_ids_are_dropped(self):
        client = FakeOpenAI(
            [Reconciliation(superseded=[SupersessionCall(fact_id="nope", reason="x")])]
        )
        real = Fact(
            id="f1",
            project_id="p",
            subject="s",
            claim="c",
            established_branch="main",
            established_position=0,
        )
        result = reconcile(
            "Something happens.", [real], position=2, branch_name="main", client=client
        )
        assert result.superseded == []

    def test_no_active_facts_means_no_model_call(self):
        client = FakeOpenAI([])
        result = reconcile("Text.", [], position=1, branch_name="main", client=client)
        assert result.superseded == []
        assert client.calls == []


class TestRestatements:
    """
    A part that ends a claim must not re-establish it.

    This came out of a live run. Part 2 unlocked the door and, in the same
    breath, the extractor wrote down "Aldous kept the only key" as current
    canon, because that sentence is sitting right there in the text as
    backstory. The draft at part 4 was then flagged for walking through a door
    that had been open for two parts.
    """

    def _fact(self, subject, claim, position=1):
        return Fact(
            id=f"f_{position}_{subject}".replace(" ", "_"),
            project_id="p",
            subject=subject,
            claim=claim,
            established_branch="main",
            established_position=position,
        )

    def test_the_live_false_positive_is_caught(self):
        ended = self._fact("Aldous", "Aldous kept the only key to the basement door on him.", 0)
        restated = self._fact(
            "Aldous",
            "Aldous was the previous owner of the house on Wexler Street and kept "
            "the only key to the basement door with him.",
            1,
        )
        assert is_restatement(restated, ended)

    def test_extra_detail_does_not_defeat_the_match(self):
        ended = self._fact("the padlock", "The padlock is rusted shut.", 0)
        restated = self._fact(
            "the padlock", "The heavy iron padlock on the door is rusted shut.", 1
        )
        assert is_restatement(restated, ended)

    def test_a_genuinely_new_claim_about_the_same_subject_survives(self):
        ended = self._fact("the padlock", "The padlock is rusted shut.", 0)
        fresh = self._fact("the padlock", "The padlock now hangs open on its hasp.", 1)
        assert not is_restatement(fresh, ended)

    def test_a_different_subject_is_never_a_restatement(self):
        ended = self._fact("the padlock", "The padlock is rusted shut.", 0)
        other = self._fact("the window", "The padlock is rusted shut.", 1)
        assert not is_restatement(other, ended)

    def test_drop_restatements_partitions_the_batch(self):
        ended = [self._fact("Aldous", "Aldous kept the only key on him.", 0)]
        candidates = [
            self._fact("Aldous", "Aldous kept the only basement key on him.", 1),
            self._fact("Meera", "Meera has found the brass key.", 1),
        ]
        keep, dropped = drop_restatements(candidates, ended)
        assert [f.subject for f in keep] == ["Meera"]
        assert [f.subject for f in dropped] == ["Aldous"]

    def test_nothing_ended_means_nothing_dropped(self):
        candidates = [self._fact("Meera", "Meera has the key.", 1)]
        keep, dropped = drop_restatements(candidates, [])
        assert len(keep) == 1 and not dropped


class TestFactHelpers:
    def test_duplicate_claims_are_collapsed(self):
        facts = [
            ExtractedFact(subject="Dev", claim="Dev has the key.", kind="possession", quote="q"),
            ExtractedFact(subject="dev", claim="Dev has the key!", kind="possession", quote="q"),
            ExtractedFact(subject="Dev", claim="Dev is afraid.", kind="state", quote="q"),
        ]
        assert len(_dedupe(facts)) == 2

    def test_extracted_facts_are_stamped_with_where_and_when(self):
        facts = to_facts(
            [ExtractedFact(subject="Dev", claim="Dev has the key.", kind="possession", quote="q")],
            project_id="p",
            branch_id="b",
            position=4,
            segment_id="seg1",
        )
        assert facts[0].established_branch == "b"
        assert facts[0].established_position == 4
        assert facts[0].source_segment_id == "seg1"

    def test_subject_matching_uses_whole_words(self):
        facts = [
            Fact(
                id="f1",
                project_id="p",
                subject="key",
                claim="c",
                established_branch="m",
                established_position=0,
            )
        ]
        assert subjects_mentioned("He turns the key.", facts) == {"key"}
        assert subjects_mentioned("She uses a keyboard.", facts) == set()

    def test_multiword_subjects_match_on_their_significant_words(self):
        facts = [
            Fact(
                id="f1",
                project_id="p",
                subject="the basement door",
                claim="c",
                established_branch="m",
                established_position=0,
            )
        ]
        assert subjects_mentioned("The door to the basement stands open.", facts)

    def test_similarity_ranks_the_nearest_fact_first(self):
        def f(fid):
            return Fact(
                id=fid,
                project_id="p",
                subject=fid,
                claim="c",
                established_branch="m",
                established_position=0,
            )

        ranked = rank_by_similarity(
            [1.0, 0.0], [(f("near"), [0.9, 0.1]), (f("far"), [0.0, 1.0])], floor=0.0
        )
        assert ranked[0][0].id == "near"

    def test_facts_without_embeddings_are_skipped(self):
        f = Fact(
            id="f1",
            project_id="p",
            subject="s",
            claim="c",
            established_branch="m",
            established_position=0,
        )
        assert rank_by_similarity([1.0, 0.0], [(f, None)]) == []


class TestPrematureAtTheSamePosition:
    """
    A fact established at the position being written is not premature.

    Reported from the Kestrel fixture: checking part four produced six
    "happens before it should" findings against facts extracted from part four
    itself, each quoting the draft's own sentence back at the writer. Nothing
    can happen before itself.
    """

    def test_the_drafts_own_facts_are_not_premature(self, seeded):
        from projects.service import StoryService

        store = seeded.store
        project, main = seeded.project, seeded.main

        # Part four has already been finalised once, so its facts sit at
        # position 3 — the same slot the writer is checking.
        a_fact(store, project.id, main.id, 3, "the harbour master",
               "The harbour master came out from Arden with three men.")
        a_fact(store, project.id, main.id, 3, "the ledger",
               "The ledger was missing for eleven hours.")

        service = StoryService(store=store, canon_store=None)
        report = service.check_draft(
            project.id, main.id,
            "By noon the harbour master had come out from Arden with three "
            "men. The ledger was missing for eleven hours.",
            position=3,
            use_llm=False,
        )

        premature = [f for f in report.findings if f.kind == "premature_reference"]
        assert premature == [], f"flagged its own material: {[f.what for f in premature]}"

    def test_something_genuinely_later_is_still_caught(self, seeded):
        """Across the fork, where the check still speaks."""
        from projects.service import StoryService

        a_fact(seeded.store, seeded.project.id, seeded.main.id, 5, "brass key",
               "Meera finds the brass key sewn into the coat.")

        service = StoryService(store=seeded.store, canon_store=None)
        report = service.check_draft(
            seeded.project.id, seeded.alt.id,
            "She turned the brass key over in her hand.",
            position=3,
            use_llm=False,
        )

        premature = [f for f in report.findings if f.kind == "premature_reference"]
        assert len(premature) == 1
        assert "brass key" in premature[0].what

    def test_the_part_number_is_not_off_by_one(self, seeded):
        """
        The finding said "not established until part 3" while the card header
        said "Part 4" for the same fact, because one added one to the
        zero-based position and the other did not.
        """
        from projects.service import StoryService

        a_fact(seeded.store, seeded.project.id, seeded.main.id, 5, "brass key",
               "Meera finds the brass key.")

        service = StoryService(store=seeded.store, canon_store=None)
        report = service.check_draft(
            seeded.project.id, seeded.alt.id,
            "She turned the brass key over in her hand.",
            position=3, use_llm=False,
        )
        finding = [f for f in report.findings if f.kind == "premature_reference"][0]

        assert "part 6" in finding.what, finding.what
        assert finding.established_part == 5


class TestWhatIsNotAContradiction:
    """
    The two exclusions the adjudicator kept agreeing to and then breaking.

    Both came out of grading the fixtures. The prompt already forbade each of
    them in plain words; the model said yes and reported them anyway. So the
    reasoning is now asked for as fields and applied here, in code, where it
    cannot be forgotten.
    """

    def _verdict(self, fact, draft: str, **overrides):
        call = dict(
            fact_id=fact.id,
            fact_is_a_standing_condition=True,
            draft_depicts_the_change=False,
            sentence_copied_from_the_draft=draft,
            contradicts=True,
            what="Something happens.",
            suggestion="Fix it.",
        )
        call.update(overrides)
        payload = ContradictionVerdicts(verdicts=[ContradictionCall(**call)])
        # A missing/wrong quote triggers a focused re-ask; queue the same
        # answer twice so the recovery pass is covered too.
        quote = call.get("sentence_copied_from_the_draft") or ""
        if not quote or quote != draft:
            return FakeOpenAI([payload, payload])
        return FakeOpenAI([payload])

    def _check(self, seeded, draft: str, client):
        return check_continuity(
            draft,
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=3,
            client=client,
        )

    def test_a_change_shown_on_the_page_is_the_story_moving(self, seeded):
        """Nuala giving away the watch is not a clash with her having worn it."""
        draft = "She took the watch off and put it into his hand."
        watch = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "the watch",
            "Nuala wears her father's watch on the inside of her wrist.",
        )
        report = self._check(
            seeded,
            draft,
            self._verdict(
                watch,
                draft,
                draft_depicts_the_change=True,
                sentence_showing_the_change=draft,
            ),
        )
        assert report.by_kind("contradiction") == []

    def test_change_after_the_clash_does_not_clear_it(self, seeded):
        """Unlocking the hold later does not excuse describing its sealed interior."""
        clash = (
            "Cael told him the beef crates inside the still-sealed hold "
            "were stacked three deep."
        )
        change = "Briggs then unlocked the hold in front of him."
        draft = f"{clash} {change}"
        sealed = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "the hold",
            "Tom Briggs has sealed the hold of the coaster with his own padlock.",
        )
        report = self._check(
            seeded,
            draft,
            self._verdict(
                sealed,
                clash,
                draft_depicts_the_change=True,
                sentence_showing_the_change=change,
                contradicts=False,
                what="Cael describes the crates while the hold is still sealed.",
            ),
        )
        assert len(report.by_kind("contradiction")) == 1

    def test_sitting_on_a_sealed_hatch_is_not_opening_it(self, seeded):
        draft = "Cael sat on the hatch of the sealed hold as if it were a bench."
        sealed = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "the hold",
            "Tom Briggs has sealed the hold with his own padlock.",
        )
        report = self._check(
            seeded,
            draft,
            self._verdict(
                sealed,
                draft,
                what="Cael is sitting on the hatch of the sealed hold.",
            ),
        )
        assert report.by_kind("contradiction") == []

    def test_something_that_merely_happened_cannot_be_contradicted(self, seeded):
        """A character sitting in a car does not clash with having climbed stairs."""
        draft = "Dev sits in the car with the rain coming down on the roof."
        stairs = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "Dev",
            "Dev climbed three flights of stairs to Mara's door.",
        )
        report = self._check(
            seeded,
            draft,
            self._verdict(stairs, draft, fact_is_a_standing_condition=False),
        )
        assert report.by_kind("contradiction") == []

    def test_a_verdict_that_quotes_nothing_is_discarded(self, seeded):
        """
        The rule that finally stopped the checker reporting silences.

        Told in prose that a draft failing to mention something is not a
        contradiction, the model produced fourteen findings on one scene, most
        of them phrased as "the draft does not mention this". A sentence that
        was never written cannot be quoted.
        """
        draft = "They ride the lift up together."
        lift = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "the lift",
            "The lift has been out of service since the fire.",
        )
        report = self._check(seeded, draft, self._verdict(lift, draft, sentence_copied_from_the_draft=""))
        assert report.by_kind("contradiction") == []

    def test_a_quote_the_draft_does_not_contain_is_discarded(self, seeded):
        lift = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "the lift",
            "The lift has been out of service since the fire.",
        )
        report = self._check(
            seeded,
            "They ride the lift up together.",
            self._verdict(lift, "", sentence_copied_from_the_draft="She took the stairs instead."),
        )
        assert report.by_kind("contradiction") == []

    def test_a_standing_condition_the_draft_ignores_still_fires(self, seeded):
        draft = "They ride the lift up together."
        lift = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "the lift",
            "The lift has been out of service since the fire.",
        )
        report = self._check(seeded, draft, self._verdict(lift, draft))
        assert len(report.by_kind("contradiction")) == 1

    def test_reflowed_punctuation_does_not_lose_a_real_quote(self, seeded):
        """The model rewraps lines and swaps apostrophes; that must not matter."""
        lift = a_fact(
            seeded.store, seeded.project.id, seeded.main.id, 0, "the lift",
            "The lift has been out of service since the fire.",
        )
        report = self._check(
            seeded,
            "They ride the lift up together,\nsaying nothing until the second floor.",
            self._verdict(
                lift, "", sentence_copied_from_the_draft="they ride the lift up together, saying nothing"
            ),
        )
        assert len(report.by_kind("contradiction")) == 1


class TestOneSentenceAgainstEverything:
    """
    A line offered as the clash for every fact is not evidence.

    Kestrel part three: the model found "Ilse Vary is the fourth keeper" and
    returned it five times, once against each fact it had nothing better to say
    about. The line really is in the draft, so it passes the evidence rule.
    """

    DRAFT = (
        "Ilse Vary is the fourth keeper of the Kestrel Light. "
        "The store below the stair holds what the keepers before her left."
    )

    def _verdicts(self, facts, quote: str):
        return FakeOpenAI(
            [
                ContradictionVerdicts(
                    verdicts=[
                        ContradictionCall(
                            fact_id=f.id,
                            fact_is_a_standing_condition=True,
                            draft_depicts_the_change=False,
                            sentence_copied_from_the_draft=quote,
                            contradicts=True,
                            what=f"The draft clashes with keeper matter {i}.",
                            suggestion="Fix it.",
                        )
                        for i, f in enumerate(facts)
                    ]
                )
            ]
        )

    def _facts(self, seeded, n: int):
        return [
            a_fact(
                seeded.store, seeded.project.id, seeded.main.id, 0,
                f"keeper matter {i}", f"Something about the keeper, number {i}.",
            )
            for i in range(n)
        ]

    def _run(self, seeded, facts, quote):
        return check_continuity(
            self.DRAFT,
            facts=seeded.store.list_facts_with_embeddings(seeded.project.id),
            graph=seeded.store.graph(seeded.project.id),
            branch_id=seeded.main.id,
            position=3,
            client=self._verdicts(facts, quote),
        )

    def test_the_same_line_against_five_facts_is_discarded(self, seeded):
        facts = self._facts(seeded, 5)
        report = self._run(seeded, facts, "Ilse Vary is the fourth keeper")
        assert report.by_kind("contradiction") == []

    def test_two_facts_from_one_line_is_allowed(self, seeded):
        """A single event can honestly clash with two standing conditions."""
        facts = self._facts(seeded, 2)
        report = self._run(seeded, facts, "Ilse Vary is the fourth keeper")
        assert len(report.by_kind("contradiction")) == 2


class TestQuestionsAskedTwice:
    """
    A story reminding the reader of its central mystery is not two mysteries.

    Halberd Street asks "who called it in?" in part one and again in part five.
    Recorded as two threads, it doubled the payoff debt and reported the same
    unanswered question twice in the panel, which reads as two problems.
    """

    def _question(self, claim: str) -> Fact:
        return Fact(
            id=f"fct_{abs(hash(claim))}",
            project_id="prj_x",
            subject="the call",
            claim=claim,
            kind="open_question",
            established_branch="brn_main",
            established_position=0,
        )

    def test_the_same_question_in_other_words_is_not_new(self):
        from projects.facts import drop_asked_again

        already = [self._question("Who called it in three weeks ago is not known.")]
        keep, echoes = drop_asked_again(
            [self._question("Who called it in, three weeks ago, is not known.")],
            already,
        )
        assert keep == []
        assert len(echoes) == 1

    def test_a_different_question_is_kept(self):
        from projects.facts import drop_asked_again

        already = [self._question("Who called it in three weeks ago is not known.")]
        keep, echoes = drop_asked_again(
            [self._question("Where Mrs Paris went in the car is not known.")],
            already,
        )
        assert len(keep) == 1
        assert echoes == []

    def test_ordinary_facts_pass_through_untouched(self):
        from projects.facts import drop_asked_again

        fact = Fact(
            id="fct_1", project_id="prj_x", subject="the lift",
            claim="The lift is out of service.", kind="state",
            established_branch="brn_main", established_position=0,
        )
        keep, echoes = drop_asked_again([fact], [])
        assert keep == [fact] and echoes == []


class TestReconcileCandidates:
    """
    Keeping the supersession list short enough to be read.

    A missed supersession does not stay quiet. The lift in Halberd Street was
    repaired on the page in part four, went unretired, and surfaced two parts
    later as a contradiction against a scene that was perfectly correct.
    """

    WORDS = [
        "lift", "cistern", "manifest", "harbour", "ledger", "padlock", "kestrel",
        "tobacco", "gallery", "brazier", "trawler", "lantern", "ferry", "engine",
        "chapel", "orchard", "wharf", "beacon", "cutter", "granite", "smithy",
        "cellar", "meadow", "belfry", "quarry", "thistle", "anchor", "pantry",
        "gable", "furnace",
    ]

    def _facts(self, n: int) -> list[Fact]:
        return [
            Fact(
                id=f"fct_{i}", project_id="p", subject=f"the {self.WORDS[i]}",
                claim=f"The {self.WORDS[i]} is somewhere.", kind="state",
                established_branch="b", established_position=i,
            )
            for i in range(n)
        ]

    def test_everything_is_shown_when_the_list_is_short(self):
        from projects.facts import _reconcile_candidates

        facts = self._facts(5)
        assert _reconcile_candidates("anything at all", facts, 24) == facts

    def test_what_the_part_names_is_shown_first(self):
        from projects.facts import _reconcile_candidates

        facts = self._facts(30)
        picked = _reconcile_candidates(
            "They mended the lift, and the ledger was found.", facts, 10
        )
        assert {f.subject for f in picked[:2]} == {"the lift", "the ledger"}
        assert len(picked) == 10

    def test_the_rest_is_filled_with_the_most_recent(self):
        from projects.facts import _reconcile_candidates

        picked = _reconcile_candidates("nothing named here", self._facts(30), 3)
        assert [f.subject for f in picked] == [
            "the furnace", "the gable", "the pantry"
        ]


class TestPrematureScope:
    """
    Which timelines and which subjects the premature check may speak about.

    Both cases below came out of running the Kestrel fixture end to end, where
    the main line was warned that its own protagonist belonged to a timeline it
    had never followed.
    """

    def test_a_fork_cannot_incriminate_the_line_it_left(self, seeded):
        from projects.service import StoryService

        store = seeded.store
        fork = store.create_branch(seeded.main.id, 2, "An alternative")
        a_fact(store, seeded.project.id, fork.id, 3, "the brass key",
               "Ilse has the brass key on a cord around her neck.")

        service = StoryService(store=store, canon_store=None)
        report = service.check_draft(
            seeded.project.id, seeded.main.id,
            "She turned the brass key over in her hand.",
            position=4, use_llm=False,
        )

        premature = [f for f in report.findings if f.kind == "premature_reference"]
        assert premature == [], "main was judged against its own descendant"

    def test_a_subject_already_in_the_story_is_not_premature(self, seeded):
        """A later part having more to say about Ilse is not a plot hole."""
        from projects.service import StoryService

        store = seeded.store
        a_fact(store, seeded.project.id, seeded.main.id, 0, "Ilse Vary",
               "Ilse Vary keeps the light on Kestrel Point.")
        a_fact(store, seeded.project.id, seeded.main.id, 5, "Ilse Vary",
               "Ilse Vary swims out to the wreck.")

        service = StoryService(store=store, canon_store=None)
        report = service.check_draft(
            seeded.project.id, seeded.main.id,
            "Ilse Vary climbed the stairs in the dark.",
            position=2, use_llm=False,
        )

        premature = [f for f in report.findings if f.kind == "premature_reference"]
        assert premature == []

    def test_a_genuinely_new_subject_from_later_still_fires(self, seeded):
        from projects.service import StoryService

        a_fact(seeded.store, seeded.project.id, seeded.main.id, 5, "the Marisa",
               "The Marisa comes onto the Skerry.")

        service = StoryService(store=seeded.store, canon_store=None)
        report = service.check_draft(
            seeded.project.id, seeded.alt.id,
            "She heard the Marisa strike from the cottage.",
            position=2, use_llm=False,
        )

        premature = [f for f in report.findings if f.kind == "premature_reference"]
        assert len(premature) == 1
        assert "the Marisa" in premature[0].what


class TestWritingSurface:
    """The editor's helpers, which are what make the format legible."""

    def test_known_speakers_come_from_earlier_parts(self, seeded):
        from dashboard import editor

        store = seeded.store
        store.add_segment(
            seeded.project.id,
            seeded.main.id,
            "INT. WARD - NIGHT\n\nREVATI\nWho was the sixth?\n\n"
            "GANPAT\nThat register should be in the cupboard.\n\n"
            "REVATI\nAnswer me.\n",
        )
        parts = store.story_so_far(seeded.project.id, seeded.main.id)
        names = editor.known_characters(parts)

        assert "REVATI" in names and "GANPAT" in names
        assert names[0] == "REVATI", "the most frequent speaker should come first"

    def test_a_stray_capitalised_slug_is_flagged(self):
        """
        The real footgun, confirmed against the parser rather than assumed.

        A bare all-caps line with nothing beneath it is dropped harmlessly. One
        with a line of action beneath it becomes a character who says that
        action out loud, which is what reaches the audio.
        """
        from audio_engine.synthesizer import parse_script
        from dashboard.editor import _suspicious_cues

        script = (
            "INT. WARD - NIGHT\n\n"
            "ON THE RADIO\nA voice reads out.\n\n"
            "REVATI\nI heard it too. Every night this week, the same broadcast.\n\n"
            "REVATI\nAnd nobody else in this building seems to notice it at all.\n"
        )
        odd = _suspicious_cues(parse_script(script))
        assert "ON THE RADIO" in odd
        assert "REVATI" not in odd, "a real character speaks more than once"

    def test_a_bare_cue_with_no_line_under_it_is_harmless(self):
        from audio_engine.synthesizer import parse_script
        from dashboard.editor import _suspicious_cues

        script = "INT. WARD - NIGHT\n\nON THE RADIO\n\nREVATI\nI heard it.\n"
        assert _suspicious_cues(parse_script(script)) == []

    def test_a_real_character_with_one_short_line_is_left_alone(self):
        """The case that made a line-counting heuristic unusable."""
        from audio_engine.synthesizer import parse_script
        from dashboard.editor import _suspicious_cues

        script = (
            "REVATI\nI heard it too. Every night this week.\n\n"
            "DR. KADAM\nNo.\n\nMATRON SULOCHANA\nGo home.\n"
        )
        assert _suspicious_cues(parse_script(script)) == []

    def test_a_transition_that_speaks_is_flagged(self):
        from audio_engine.synthesizer import parse_script
        from dashboard.editor import _suspicious_cues

        script = "BACK TO THE WARD\nThe lamp is still burning.\n"
        assert _suspicious_cues(parse_script(script)) == ["BACK TO THE WARD"]

    def test_the_narrator_is_never_flagged(self):
        from audio_engine.synthesizer import parse_script
        from dashboard.editor import _suspicious_cues

        script = "NARRATOR (V.O.)\nRain.\n"
        assert _suspicious_cues(parse_script(script)) == []


class TestMidStoryChecking:
    """
    Forking from the middle must not be checked against the branch's tip.

    Otherwise an alternative to part 2 gets measured against facts from parts
    3 and 4, which on the timeline being forked never happen.
    """

    def test_a_mid_story_check_ignores_later_facts(self, seeded):
        from projects.service import StoryService

        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            0,
            "the padlock",
            "The padlock is rusted shut.",
        )
        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            3,
            "brass key",
            "Meera finds the brass key.",
        )
        service = StoryService(store=seeded.store, canon_store=None)

        at_tip = service.check_draft(
            seeded.project.id, seeded.main.id, "Someone waits.", use_llm=False
        )
        mid = service.check_draft(
            seeded.project.id,
            seeded.main.id,
            "Someone waits.",
            position=1,
            use_llm=False,
        )

        assert at_tip.facts_checked == 2
        assert mid.facts_checked == 1, "the part 4 fact has not happened yet"

    def test_the_default_is_still_the_tip(self, seeded):
        from projects.service import StoryService

        a_fact(
            seeded.store,
            seeded.project.id,
            seeded.main.id,
            0,
            "the padlock",
            "The padlock is rusted shut.",
        )
        service = StoryService(store=seeded.store, canon_store=None)
        report = service.check_draft(
            seeded.project.id, seeded.main.id, "Someone waits.", use_llm=False
        )
        assert report.facts_checked == 1


class TestBranchGraphDrawing:
    """
    The picture has to agree with the visibility rule.

    A graph that draws a branch as starting from nothing, or that shows it
    inheriting parts it cannot see, teaches the writer the wrong model of
    their own story.
    """

    @pytest.fixture
    def figure(self, seeded):
        from dashboard import branch_graph

        store = seeded.store
        graph = store.graph(seeded.project.id)
        return branch_graph.build(
            graph,
            store.list_branches(seeded.project.id),
            store.list_segments(seeded.project.id),
            seeded.alt.id,
        )

    def test_every_timeline_gets_a_lane(self, figure):
        labels = [t.replace("<b>", "").replace("</b>", "") for t in figure.layout.yaxis.ticktext]
        assert labels == ["Main timeline", "What if Dev goes first"]

    def test_the_open_timeline_is_marked(self, figure):
        assert any("<b>" in t for t in figure.layout.yaxis.ticktext)

    def test_parts_are_labelled_from_one_not_zero(self, figure):
        assert figure.layout.xaxis.ticktext[0] == "Part 1"

    def test_a_branch_shows_the_parts_it_inherited(self, figure):
        """alt forked at 2, so main's parts at positions 0 and 1 belong to it too."""
        inherited = [
            trace
            for trace in figure.data
            if not trace.meta and list(trace.y or []) == [1, 1] and len(trace.x) == 2
        ]
        assert inherited, "the branch lane should carry its inherited run"
        assert list(inherited[0].x) == [0, 1]

    def test_the_fork_is_an_elbow_not_a_diagonal(self, figure):
        """Collinear diagonals from successive forks merge into one false line."""
        forks = [t for t in figure.data if t.line.dash == "dot"]
        assert len(forks) == 1
        assert len(forks[0].x) == 4, "an elbow needs four points, a diagonal has two"
        assert forks[0].y[0] == 0 and forks[0].y[-1] == 1

    def test_a_branch_carries_its_id_for_click_to_switch(self, figure):
        assert any(t.meta for t in figure.data)

    def test_an_empty_branch_still_gets_a_marker(self, store):
        from dashboard import branch_graph

        project, main = store.create_project("P")
        store.add_segment(project.id, main.id, "One.")
        store.add_segment(project.id, main.id, "Two.")
        empty = store.create_branch(main.id, 1, "Not written yet")

        figure = branch_graph.build(
            store.graph(project.id),
            store.list_branches(project.id),
            store.list_segments(project.id),
            empty.id,
        )
        stub = [t for t in figure.data if t.meta == empty.id]
        assert stub and list(stub[0].x) == [1]

    def test_a_grandchild_sits_below_its_own_parent(self, store):
        """Depth first, so a fork is drawn next to the branch it came from."""
        from dashboard import branch_graph

        project, main = store.create_project("P")
        for _ in range(4):
            store.add_segment(project.id, main.id, "Part.")
        alt = store.create_branch(main.id, 2, "Second")
        store.create_branch(main.id, 3, "Sibling")
        store.create_branch(alt.id, 3, "Child of second")

        figure = branch_graph.build(
            store.graph(project.id),
            store.list_branches(project.id),
            store.list_segments(project.id),
            main.id,
        )
        labels = [t.replace("<b>", "").replace("</b>", "") for t in figure.layout.yaxis.ticktext]
        assert labels.index("Child of second") == labels.index("Second") + 1


class TestChunking:
    def test_screenplay_splits_on_scene_headings(self, screenplay):
        chunks = chunk_story(screenplay)
        assert len(chunks) >= 1
        assert all(text.strip() for _, text in chunks)

    def test_character_attribution_prefers_screenplay_cues(self):
        assert dominant_character("PRIYA\nThat's just the pipes.\n") == "Priya"

    def test_transitions_are_not_mistaken_for_characters(self):
        assert dominant_character("CUT TO:\n\nSomething moves.\n") != "Cut To"

    def test_empty_input_yields_nothing(self):
        assert chunk_story("") == []
        assert split_into_parts("   ") == []

    def test_long_prose_is_divided_into_parts(self, prose):
        long_story = "\n\n".join([prose] * 8)
        parts = split_into_parts(long_story, target_words=300)
        assert len(parts) >= 2
        assert all(p.strip() for p in parts)

    def test_no_text_is_lost_when_splitting_into_parts(self, prose):
        parts = split_into_parts(prose, target_words=100)
        assert sum(len(p.split()) for p in parts) == len(prose.split())
