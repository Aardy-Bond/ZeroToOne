"""
Projects, branching timelines, and a time-aware canon.

The design rests on one idea: branching and fact supersession are the same
problem. A part being written, a fact being established, and a fact ceasing to
be true are all events with a branch and a position, and a single predicate in
`visibility.py` decides whether any of them is visible from where the writer is
standing.

That is what lets the plot-hole finder answer "is this true *here*" rather than
"does this appear anywhere in the story", which is all a similarity search can
tell you.
"""

from .schemas import (
    Branch,
    ContinuityFinding,
    ContinuityReport,
    Fact,
    Project,
    Segment,
)
from .service import FinaliseResult, StoryService
from .store import ProjectStore
from .visibility import BranchGraph, Event

__all__ = [
    "Branch",
    "BranchGraph",
    "ContinuityFinding",
    "ContinuityReport",
    "Event",
    "Fact",
    "FinaliseResult",
    "Project",
    "ProjectStore",
    "Segment",
    "StoryService",
]
