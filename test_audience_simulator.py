"""
Smoke test for the Project Anubhuti Synthetic Audience Simulator.

Runs a multi-minute horror script past all three personas and prints the
aggregated drop-off heatmap.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from audience_simulator.simulator import (
    DROP_OFF_THRESHOLD,
    SCORE_MAX,
    AudienceSimulator,
    AudienceSimulatorError,
    _persona_name,
)

TEST_SCRIPT = """\
INT. BASEMENT LAUNDRY ROOM - NIGHT

A single bulb swings. PRIYA (28) stands at the dryer, folding a child's
t-shirt. The machine has been off for an hour. The clothes are still warm.

PRIYA
Okay. Okay, that's just... that's the pipes.

She folds another shirt. Then another. She does not turn around.

NARRATOR (V.O.)
The house on Wexler Street was built in 1911 by a shipping clerk named
Aldous Renn, who had come into money nobody could account for. The deed
changed hands eleven times before the Kapoors bought it, and each of those
eleven families left in under two years. The realtor mentioned none of this.
She mentioned the crown molding.

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

CUT TO:

INT. KITCHEN - CONTINUOUS

DEV (34) scrolls his phone at the counter. The basement door is open behind
him. He does not look at it.

DEV
Priya? You want tea?

No answer. He sets the phone down.

DEV (CONT'D)
Priya.

He walks to the basement door. The stairs go down into nothing.

DEV (CONT'D)
This isn't funny either, you know. Both of you. This whole family.

He takes the first step. The wood gives under him, wet.

NARRATOR (V.O.)
There is a particular silence that a house makes when it is deciding
something about you. Dev had heard it twice before in his life and had
convinced himself both times that he had imagined it.

DEV
Priya, I swear to God—

A small voice from the dark below. A child's voice.

ARJUN (O.S.)
Dad? Mum's down here. She wants you to come see.

Dev stops. His son is asleep upstairs. He put him there ninety minutes ago.

ARJUN (O.S.) (CONT'D)
Dad. Come see.

The basement door swings shut behind him.

SMASH TO BLACK.
"""


def render_bar(score: float, width: int = 20) -> str:
    filled = int(round((score / SCORE_MAX) * width))
    return "█" * filled + "·" * (width - filled)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 78)
    print("PROJECT ANUBHUTI — SYNTHETIC AUDIENCE SIMULATOR")
    print("=" * 78)

    try:
        simulator = AudienceSimulator()
        report = simulator.simulate_audience(TEST_SCRIPT)
    except (AudienceSimulatorError, ValueError) as exc:
        print(f"\nFAILED: {exc}")
        return 1

    print("\nPERSONA VERDICTS")
    print("-" * 78)
    for key, verdict in report.verdicts.items():
        finish = "would finish" if verdict.would_finish else "DROPS OFF"
        print(f"\n  {_persona_name(key)} — {finish}")
        print(f"    {verdict.overall_summary}")

    if report.failures:
        print("\n  Failed personas:")
        for key, err in report.failures.items():
            print(f"    {_persona_name(key)}: {err}")

    print("\n\nAGGREGATED ENGAGEMENT HEATMAP")
    print("-" * 78)
    print(f"  {'MIN':<5} {'AVG':<6} {'BAR':<22} PERSONA SCORES")
    for entry in report.heatmap:
        flag = "  <-- DROP-OFF" if entry.is_drop_off else ""
        scores = " ".join(
            f"{k.split('_')[0]}={v}" for k, v in entry.persona_scores.items()
        )
        print(
            f"  {entry.minute:<5} {entry.average_score:<6.2f} "
            f"{render_bar(entry.average_score):<22} {scores}{flag}"
        )

    print(f"\n  Overall average engagement: {report.overall_average:.2f}/{SCORE_MAX}")

    print("\n\nDROP-OFF REASONS (score < %d)" % DROP_OFF_THRESHOLD)
    print("-" * 78)
    found = False
    for entry in report.heatmap:
        for persona_key, reason in entry.drop_off_reasons.items():
            found = True
            print(f"  [min {entry.minute}] {_persona_name(persona_key)}")
            print(f"      {reason}")
    if not found:
        print("  No persona dropped below the threshold at any minute.")

    print("\n\nWEAKEST MINUTES")
    print("-" * 78)
    for entry in report.weakest_minutes[:3]:
        print(f"  min {entry.minute}: {entry.average_score:.2f}")

    print("\n\nRAW HEATMAP JSON")
    print("-" * 78)
    print(json.dumps([e.to_dict() for e in report.heatmap], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
