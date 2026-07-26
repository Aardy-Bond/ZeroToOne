"""
The one rule that orders a branching story.

Everything the canon knows is an *event* with a branch and a position: a part
being written, a fact being established, a fact being superseded. A single
predicate decides whether an event is visible from where the writer is
standing, and every other question in this package reduces to it.

Collapsing branching and supersession into one rule is what keeps this small.
The alternative -- filtering parts by branch in one place and expiring facts by
timestamp in another -- gets the interesting case wrong, because whether a fact
has expired *depends on which timeline you are standing on*.

Worked example, the one that motivates the design:

    main:  pos 0 ... pos 1 [door locked] ... pos 2 [key found] ... pos 3
                            \\
    alt-A:                    fork at 2 ... pos 2 ... pos 3

The "door is locked" fact is established at main position 1 and superseded at
main position 2.

    On main at position 3   establishment visible, supersession visible
                            -> inactive, the door is open
    On alt-A at position 3  establishment visible, supersession NOT visible,
                            because alt-A forked at 2 and the supersession
                            happened at 2
                            -> still active, the door is still locked

The same fact, opposite truth values, from one comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

# A branch can always see its own events, however far along they are. Using an
# int rather than math.inf keeps every comparison in integer space.
UNBOUNDED = 1 << 62


@dataclass(frozen=True)
class Event:
    """Anything that happened somewhere on the branch tree."""

    branch_id: str
    position: int


class BranchGraph:
    """
    The shape of a project's timelines, and the questions asked of that shape.

    Built once from the branch rows and then queried, so the recursive parent
    walk happens on construction rather than on every fact comparison.
    """

    def __init__(self, branches) -> None:
        self._parent: dict[str, str | None] = {}
        self._forked_at: dict[str, int | None] = {}
        self._name: dict[str, str] = {}

        for branch in branches:
            self._parent[branch.id] = branch.parent_id
            self._forked_at[branch.id] = branch.forked_at
            self._name[branch.id] = branch.name

    def __contains__(self, branch_id: str) -> bool:
        return branch_id in self._parent

    def name_of(self, branch_id: str) -> str:
        return self._name.get(branch_id, branch_id)

    def ancestry(self, branch_id: str) -> list[str]:
        """This branch, then its parent, and so on up to the root."""
        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = branch_id

        while current is not None and current in self._parent:
            if current in seen:  # defensive: a cycle would hang the walk
                break
            chain.append(current)
            seen.add(current)
            current = self._parent[current]

        return chain

    def lineage_cutoff(self, branch_id: str) -> dict[str, int]:
        """
        How far this branch can see into each branch of its ancestry.

        The branch itself is unbounded. Each ancestor is capped at the fork
        point of the descendant that came from it -- and capped again by any
        tighter cap further down the chain, so a fork at 2 below a fork at 5
        still yields 2.
        """
        cutoff: dict[str, int] = {}
        limit = UNBOUNDED

        for current in self.ancestry(branch_id):
            cutoff[current] = limit
            forked_at = self._forked_at.get(current)
            if forked_at is not None:
                limit = min(limit, forked_at)

        return cutoff

    def visible(
        self,
        event: Event | None,
        branch_id: str,
        position: int = UNBOUNDED,
    ) -> bool:
        """
        Can an observer on `branch_id` at `position` see this event?

        A missing event is not visible, which is what makes the fact rule read
        cleanly: a fact that was never superseded has no supersession event,
        so `not visible(None, ...)` is True and the fact stays active.
        """
        if event is None:
            return False

        cutoff = self.lineage_cutoff(branch_id).get(event.branch_id)
        if cutoff is None:
            return False  # a sibling branch, or another project entirely

        return event.position < min(cutoff, position)

    def is_active(self, fact, branch_id: str, position: int = UNBOUNDED) -> bool:
        """Whether a fact still holds, seen from one point on one timeline."""
        established = Event(fact.established_branch, fact.established_position)
        if not self.visible(established, branch_id, position):
            return False

        if fact.superseded_position is None or fact.superseded_branch is None:
            return True

        superseded = Event(fact.superseded_branch, fact.superseded_position)
        return not self.visible(superseded, branch_id, position)

    def active_facts(self, facts, branch_id: str, position: int = UNBOUNDED) -> list:
        """Every fact that still holds here, in the order it was established."""
        live = [f for f in facts if self.is_active(f, branch_id, position)]
        live.sort(key=lambda f: (f.established_position, f.subject.lower()))
        return live

    def established_elsewhere(self, facts, branch_id: str, position: int) -> list:
        """
        Facts that exist in this project but cannot be known here yet.

        Either they happen later on this timeline, or they belong to a branch
        this one never descended from. Referring to one is the "Dev mentions
        the key before he finds it" mistake.
        """
        return [
            f
            for f in facts
            if not self.visible(
                Event(f.established_branch, f.established_position),
                branch_id,
                position,
            )
        ]

    def next_position(self, branch_id: str, segments) -> int:
        """
        Where the next part on this branch goes.

        A new branch starts numbering at its fork point rather than at zero,
        so positions stay comparable across the whole lineage.
        """
        own = [s.position for s in segments if s.branch_id == branch_id]
        if own:
            return max(own) + 1

        forked_at = self._forked_at.get(branch_id)
        return forked_at if forked_at is not None else 0

    def visible_segments(self, segments, branch_id: str, position: int = UNBOUNDED) -> list:
        """The story so far on this timeline, in reading order."""
        keep = [
            s
            for s in segments
            if self.visible(Event(s.branch_id, s.position), branch_id, position)
        ]
        keep.sort(key=lambda s: s.position)
        return keep

    def children_of(self, branch_id: str) -> list[str]:
        return [b for b, parent in self._parent.items() if parent == branch_id]

    def roots(self) -> list[str]:
        return [b for b, parent in self._parent.items() if parent is None]
