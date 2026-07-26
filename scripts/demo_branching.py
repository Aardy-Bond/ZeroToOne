"""
End-to-end demonstration of the branching canon.

Builds a three-part story where a door is established as locked and later
unlocked, forks a timeline from before the unlocking, and then runs the same
draft against both timelines.

The point is the asymmetry at the end. One draft, two timelines, opposite
verdicts, because the fact ledger knows *when* the key was found and which
timeline saw it happen.

    python scripts/demo_branching.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from projects.service import StoryService  # noqa: E402
from projects.store import ProjectStore  # noqa: E402

DEMO_DB = PROJECT_ROOT / "projects" / "demo_branching.db"

PART_ONE = """\
INT. WEXLER STREET HOUSE - BASEMENT DOOR - NIGHT

The door at the bottom of the stairs is steel, and it does not belong in a
house like this. PRIYA runs her palm across it. A padlock hangs through the
hasp, rusted shut.

PRIYA
Who puts a lock like this on the inside?

DEV
Aldous did. He kept the only key on him. They never found it when he died.

Priya tests the padlock. It does not move.

DEV (CONT'D)
Nobody has been through that door in eleven years.
"""

PART_TWO = """\
INT. WEXLER STREET HOUSE - KITCHEN - DAY

MEERA empties the last of the drawers onto the counter. Buttons, dead
batteries, a curtain hook. Then something heavier.

A brass key, green at the teeth.

MEERA
Dev. Dev, come here.

She holds it up to the window. Dev takes the stairs two at a time.

DEV
That's it. That's the one.

They go down together. The padlock takes the key the way a lock takes a key it
has been waiting for. The hasp swings free. The basement door stands open for
the first time in eleven years, and the air that comes out of it is colder than
the house.
"""

PART_THREE = """\
INT. WEXLER STREET HOUSE - BASEMENT - CONTINUOUS

Concrete, and a drain in the centre of the floor, and a trough cut into the
slab that is exactly the length of a child.

MEERA
We need to call someone.

DEV
Not yet.
"""

# Written for the branch. On the fork the key was never found, so the door is
# still locked and this draft walks straight through it.
DRAFT = """\
INT. WEXLER STREET HOUSE - BASEMENT - NIGHT

Priya comes down the stairs with a torch, pushes the basement door open, and
steps onto the concrete. The drain is where she remembers it.

PRIYA
Dev, you need to see this.
"""


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def show(report, timeline: str) -> None:
    print(f"\n  On '{timeline}':")
    print(f"    facts considered: {report.facts_checked}")
    if report.is_clean:
        print(f"    verdict: CLEAN — {report.note}")
        return
    for finding in report.findings:
        print(f"    [{finding.severity.upper()}] {finding.label}")
        print(f"      {finding.what}")
        if finding.established:
            part = (
                f"part {finding.established_part + 1}"
                if finding.established_part is not None
                else "canon"
            )
            print(f"      established in {part}: {finding.established}")
        if finding.suggestion:
            print(f"      fix: {finding.suggestion}")


def main() -> int:
    if DEMO_DB.exists():
        DEMO_DB.unlink()

    store = ProjectStore(DEMO_DB)
    service = StoryService(store=store, canon_store=None)

    rule("Building the story")
    project, main_branch = service.create_project(
        "The House on Wexler Street", logline="A locked door, and the key that opens it."
    )
    print(f"  project: {project.title}")

    for index, text in enumerate([PART_ONE, PART_TWO, PART_THREE], start=1):
        result = service.finalise_part(project.id, main_branch.id, text)
        line = f"  part {index}: +{result.facts_added} facts"
        if result.facts_superseded:
            line += f", {result.facts_superseded} no longer true"
        print(line)
        for claim in result.superseded_claims:
            print(f"      ended: {claim}")

    rule("What each timeline holds as true")
    live = service.active_facts(project.id, main_branch.id)
    print(f"\n  Main timeline, {len(live)} facts still true:")
    for fact in live:
        print(f"    ({fact.kind}, part {fact.established_position + 1}) "
              f"{fact.subject}: {fact.claim}")

    superseded = [f for f in store.list_facts(project.id) if f.is_superseded]
    print(f"\n  No longer true on main, {len(superseded)}:")
    for fact in superseded:
        print(f"    {fact.claim}")
        print(f"      established part {fact.established_position + 1}, "
              f"ended at part {fact.superseded_position + 1}")

    rule("Forking a timeline from before the key was found")
    branch = service.create_branch(project.id, main_branch.id, 1, "The key stays lost")
    print(f"  '{branch.name}' forks after part 1.")
    print("  It inherits part 1 and nothing after it.")

    branch_live = service.active_facts(project.id, branch.id)
    print(f"\n  On the branch, {len(branch_live)} facts are true:")
    for fact in branch_live:
        print(f"    ({fact.kind}, part {fact.established_position + 1}) "
              f"{fact.subject}: {fact.claim}")

    revived = [f for f in branch_live if f.is_superseded]
    if revived:
        print("\n  Note: these are superseded on main but still true here,")
        print("  because the branch never saw the part that ended them:")
        for fact in revived:
            print(f"    {fact.claim}")

    rule("The same draft, checked against both timelines")
    print("\n  Draft: Priya pushes the basement door open and walks in.")

    show(service.check_draft(project.id, main_branch.id, DRAFT), "Main timeline")
    show(service.check_draft(project.id, branch.id, DRAFT), branch.name)

    rule("Why this matters")
    print(
        "\n  A plain similarity search over all canon would surface 'the door is\n"
        "  locked' on both timelines, because the sentence exists in part 1 either\n"
        "  way. The ledger knows the key was found at part 2 and that the branch\n"
        "  forked before it, so the door is open on one timeline and shut on the\n"
        "  other, from a single comparison."
    )

    store.close()
    DEMO_DB.unlink(missing_ok=True)
    shutil.rmtree(DEMO_DB.parent / "__pycache__", ignore_errors=True)
    for suffix in ("-wal", "-shm"):
        Path(str(DEMO_DB) + suffix).unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
