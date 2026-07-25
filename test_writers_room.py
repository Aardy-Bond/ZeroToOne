"""
Smoke test for the Project Anubhuti Writers Room.

Feeds a short horror scene to the expert panel and prints the parsed JSON so
we can confirm the structured output matches SceneCritique.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from writers_room.orchestrator import WritersRoomError, analyze_script

DUMMY_SCRIPT = """\
INT. BASEMENT LAUNDRY ROOM - NIGHT

A single bulb swings. PRIYA (28) stands at the dryer, folding a child's
t-shirt. The machine has been off for an hour. The clothes are still warm.

PRIYA
Okay. Okay, that's just... that's the pipes.

She folds another shirt. Then another. She does not turn around.

From the far corner, a slow DRAG of something heavy across concrete.

PRIYA (CONT'D)
Arjun? Sweetheart, if that's you, this isn't funny right now.

The dragging stops. Directly behind her.

PRIYA (CONT'D)
I'm going to count to three, and then I'm going to turn around, and you
are going to be my son. Okay? One.

The bulb dies.

PRIYA (CONT'D)
Two.

Something breathes. It is not a child.

CUT TO BLACK.
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 72)
    print("PROJECT ANUBHUTI — WRITERS ROOM SMOKE TEST")
    print("=" * 72)
    print(DUMMY_SCRIPT)
    print("=" * 72)

    try:
        critique = analyze_script(DUMMY_SCRIPT)
    except (WritersRoomError, ValueError) as exc:
        print(f"\nFAILED: {exc}")
        return 1

    print("\nPARSED JSON OUTPUT")
    print("-" * 72)
    print(critique.model_dump_json(indent=2))

    print("\nSCHEMA CHECK")
    print("-" * 72)
    print(f"  director_notes      : {len(critique.director_notes)} chars")
    print(f"  editor_notes        : {len(critique.editor_notes)} chars")
    print(f"  psychologist_notes  : {len(critique.psychologist_notes)} chars")
    print(f"  continuity_critique : {len(critique.continuity_critique)} chars")
    print(f"  foley_triggers      : {len(critique.foley_triggers)} cues")
    for cue in critique.foley_triggers:
        print(f"      [{cue.timestamp}] {cue.sound_effect}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
