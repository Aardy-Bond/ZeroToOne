"""
The branch visibility rule.

These tests are the specification for the design. If one of them fails, the
plot-hole finder will report contradictions that are not real, or miss ones
that are, so they are written to be read rather than merely to pass.
"""

from __future__ import annotations

import pytest

from projects.schemas import Branch, Fact, Segment
from projects.visibility import UNBOUNDED, BranchGraph, Event


def branch(bid, parent=None, forked_at=None, name=None):
    return Branch(
        id=bid,
        project_id="prj",
        name=name or bid,
        parent_id=parent,
        forked_at=forked_at,
    )


def fact(fid, est_branch, est_pos, sup_branch=None, sup_pos=None, kind="state"):
    return Fact(
        id=fid,
        project_id="prj",
        subject="basement door",
        claim="The basement door is locked.",
        kind=kind,
        established_branch=est_branch,
        established_position=est_pos,
        superseded_branch=sup_branch,
        superseded_position=sup_pos,
    )


def segment(branch_id, position):
    return Segment(
        id=f"seg_{branch_id}_{position}",
        project_id="prj",
        branch_id=branch_id,
        position=position,
        text=f"Part {position} on {branch_id}.",
    )


@pytest.fixture
def simple_fork():
    """
    main:  0 -- 1 [door locked] -- 2 [key found] -- 3
                  \\
    alt:            fork at 2 -- 2 -- 3
    """
    return BranchGraph([branch("main"), branch("alt", parent="main", forked_at=2)])


class TestLineageCutoff:
    def test_root_sees_all_of_itself(self):
        graph = BranchGraph([branch("main")])
        assert graph.lineage_cutoff("main") == {"main": UNBOUNDED}

    def test_child_sees_itself_unbounded_and_parent_to_the_fork(self, simple_fork):
        cutoff = simple_fork.lineage_cutoff("alt")
        assert cutoff["alt"] == UNBOUNDED
        assert cutoff["main"] == 2

    def test_unknown_branch_has_no_cutoff(self, simple_fork):
        assert simple_fork.lineage_cutoff("nope") == {}

    def test_grandchild_inherits_the_tighter_cap(self):
        """A fork at 2 above a fork at 5 still limits the root to 2."""
        graph = BranchGraph(
            [
                branch("main"),
                branch("mid", parent="main", forked_at=2),
                branch("leaf", parent="mid", forked_at=5),
            ]
        )
        cutoff = graph.lineage_cutoff("leaf")
        assert cutoff["leaf"] == UNBOUNDED
        assert cutoff["mid"] == 5
        assert cutoff["main"] == 2, "the root cap must not widen back out to 5"

    def test_cycles_do_not_hang(self):
        graph = BranchGraph(
            [branch("a", parent="b", forked_at=1), branch("b", parent="a", forked_at=1)]
        )
        assert len(graph.ancestry("a")) <= 2


class TestVisibility:
    def test_own_past_is_visible(self, simple_fork):
        assert simple_fork.visible(Event("main", 1), "main", 3)

    def test_own_future_is_not(self, simple_fork):
        assert not simple_fork.visible(Event("main", 5), "main", 3)

    def test_position_is_exclusive(self, simple_fork):
        assert not simple_fork.visible(Event("main", 3), "main", 3)

    def test_inherited_parent_past_is_visible(self, simple_fork):
        assert simple_fork.visible(Event("main", 1), "alt", 9)

    def test_parent_events_at_the_fork_are_not_inherited(self, simple_fork):
        """The fork point is exclusive, so main's part 2 belongs only to main."""
        assert not simple_fork.visible(Event("main", 2), "alt", 9)

    def test_sibling_branches_cannot_see_each_other(self):
        graph = BranchGraph(
            [
                branch("main"),
                branch("alt-a", parent="main", forked_at=2),
                branch("alt-b", parent="main", forked_at=2),
            ]
        )
        assert not graph.visible(Event("alt-a", 4), "alt-b", 9)
        assert not graph.visible(Event("alt-b", 4), "alt-a", 9)

    def test_a_missing_event_is_never_visible(self, simple_fork):
        assert not simple_fork.visible(None, "main", 3)


class TestTheLockedDoor:
    """
    The case that motivates the whole design.

    A locked door established at main position 1, unlocked at main position 2.
    A branch forked at 2 never saw the key being found, so the door must still
    be locked there while it is open on main.
    """

    @pytest.fixture
    def door(self):
        return fact("f_door", "main", 1, sup_branch="main", sup_pos=2)

    def test_open_on_the_line_where_the_key_was_found(self, simple_fork, door):
        assert not simple_fork.is_active(door, "main", 9)

    def test_still_locked_on_the_branch_that_forked_before(self, simple_fork, door):
        assert simple_fork.is_active(door, "alt", 9), (
            "the supersession happened on main at the fork point, so the branch "
            "never saw it and the door is still locked there"
        )

    def test_locked_on_main_before_the_key_is_found(self, simple_fork, door):
        assert simple_fork.is_active(door, "main", 2)

    def test_not_yet_established_at_the_very_start(self, simple_fork, door):
        assert not simple_fork.is_active(door, "main", 1)


class TestActiveFacts:
    def test_never_superseded_facts_stay_active(self, simple_fork):
        f = fact("f1", "main", 0)
        assert simple_fork.is_active(f, "main", 9)

    def test_active_facts_are_returned_in_story_order(self, simple_fork):
        facts = [fact("f3", "main", 3), fact("f1", "main", 0), fact("f2", "main", 1)]
        order = [f.id for f in simple_fork.active_facts(facts, "main", 9)]
        assert order == ["f1", "f2", "f3"]

    def test_a_branch_sees_inherited_and_own_facts_together(self, simple_fork):
        facts = [fact("inherited", "main", 0), fact("own", "alt", 4)]
        live = {f.id for f in simple_fork.active_facts(facts, "alt", 9)}
        assert live == {"inherited", "own"}

    def test_supersession_on_a_branch_does_not_affect_the_parent(self, simple_fork):
        f = fact("f1", "main", 0, sup_branch="alt", sup_pos=4)
        assert simple_fork.is_active(f, "main", 9), (
            "a branch ending a fact must not end it on the timeline it forked from"
        )
        assert not simple_fork.is_active(f, "alt", 9)


class TestPrematureReference:
    def test_later_facts_on_the_same_branch_are_flagged(self, simple_fork):
        facts = [fact("early", "main", 0), fact("later", "main", 7)]
        unknown = {
            f.id for f in simple_fork.established_elsewhere(facts, "main", 3)
        }
        assert unknown == {"later"}

    def test_sibling_branch_facts_are_flagged(self):
        graph = BranchGraph(
            [
                branch("main"),
                branch("alt-a", parent="main", forked_at=2),
                branch("alt-b", parent="main", forked_at=2),
            ]
        )
        facts = [fact("on_a", "alt-a", 3)]
        unknown = {f.id for f in graph.established_elsewhere(facts, "alt-b", 9)}
        assert unknown == {"on_a"}


class TestPositions:
    def test_a_new_branch_starts_numbering_at_its_fork(self, simple_fork):
        segments = [segment("main", 0), segment("main", 1), segment("main", 2)]
        assert simple_fork.next_position("alt", segments) == 2

    def test_an_existing_branch_continues_from_its_last_part(self, simple_fork):
        segments = [segment("main", 0), segment("alt", 2), segment("alt", 3)]
        assert simple_fork.next_position("alt", segments) == 4

    def test_an_empty_root_starts_at_zero(self, simple_fork):
        assert simple_fork.next_position("main", []) == 0

    def test_story_so_far_stitches_the_lineage_in_reading_order(self, simple_fork):
        segments = [
            segment("main", 0),
            segment("main", 1),
            segment("main", 2),  # after the fork: must not appear
            segment("alt", 2),
            segment("alt", 3),
        ]
        visible = simple_fork.visible_segments(segments, "alt")
        assert [(s.branch_id, s.position) for s in visible] == [
            ("main", 0),
            ("main", 1),
            ("alt", 2),
            ("alt", 3),
        ]
