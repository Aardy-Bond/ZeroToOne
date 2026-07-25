"""
End-to-end pipeline test for Project Anubhuti.

Runs a draft script through every stage: canon retrieval, expert critique,
synthetic audience simulation, targeted auto-rewrite, and audio export.

The test script deliberately buries a dry exposition dump inside a working
horror scene so the rewrite loop has something real to fix.
"""

from __future__ import annotations

import argparse
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
)
from audio_engine.synthesizer import AudioSynthesizer, SynthesisError
from writers_room.orchestrator import analyze_script, fetch_canon, format_canon_warnings
from writers_room.rewrite_engine import (
    RewriteEngineError,
    rewrite_weak_segments,
    select_weak_minutes,
)

DRAFT_SCRIPT = """\
INT. BASEMENT LAUNDRY ROOM - NIGHT

A single bulb swings. PRIYA stands at the dryer, folding a child's t-shirt.
The machine has been off for an hour. The clothes are still warm.

PRIYA
Okay. Okay, that's just... that's the pipes.

She folds another shirt. She does not turn around.

NARRATOR (V.O.)
The house on Wexler Street was built in 1911 by a shipping clerk named Aldous
Renn. The deed changed hands eleven times before the Kapoors bought it. The
first owner after Renn was a widow named Cotterell, who held the property for
six years and sold it to a consortium of three brothers from the shipping
trade. The consortium dissolved in 1931 following a dispute over the eastern
easement, which had been surveyed incorrectly in 1908 and would be resurveyed
twice more before the matter was settled in county court. The court records
run to four hundred pages. The relevant finding concerns the drainage rights
beneath the north foundation, which is where the basement sits.

From the far corner, a slow DRAG of something heavy across concrete.

PRIYA (CONT'D)
Arjun? Sweetheart, if that's you, this isn't funny right now.

The dragging stops. Directly behind her.

PRIYA (CONT'D)
I'm going to count to three, and then I'm going to turn around, and you are
going to be my son. Okay? One.

The bulb dies.

PRIYA (CONT'D)
Two.

Something breathes. It is not a child.

CUT TO:

INT. KITCHEN - CONTINUOUS

DEV scrolls his phone at the counter. The basement door is open behind him.

DEV
Priya? You want tea?

No answer. He sets the phone down.

DEV (CONT'D)
Priya.

He walks to the basement door. The stairs go down into nothing.

DEV (CONT'D)
This isn't funny either. Both of you. This whole family.

He takes the first step. The wood gives under him, wet.

ARJUN (O.S.)
Dad? Mum's down here. She wants you to come see.

Dev stops. His son is asleep upstairs. He put him there ninety minutes ago.

ARJUN (O.S.) (CONT'D)
Dad. Come see.

The basement door swings shut behind him.

SMASH TO BLACK.
"""


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_heatmap(report) -> None:
    flagged = {e.minute for e in select_weak_minutes(report.heatmap)}
    print(f"  {'MIN':<5} {'AVG':<7} PERSONA SCORES")
    for entry in report.heatmap:
        flag = "  <-- FLAGGED" if entry.minute in flagged else ""
        scores = " ".join(
            f"{k.split('_')[0]}={v}" for k, v in entry.persona_scores.items()
        )
        print(f"  {entry.minute:<5} {entry.average_score:<7.2f} {scores}{flag}")
    print(f"\n  Overall: {report.overall_average:.2f}/{SCORE_MAX}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Project Anubhuti full pipeline test.")
    parser.add_argument("--skip-lore", action="store_true", help="Skip canon retrieval")
    parser.add_argument("--skip-audio", action="store_true", help="Skip TTS rendering")
    parser.add_argument("--skip-resim", action="store_true", help="Skip re-simulation")
    parser.add_argument("--character-id", default="arjun", help="Bias canon retrieval")
    parser.add_argument("--output-dir", default="output", help="Export directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    for noisy in ("httpx", "databricks.sdk", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    banner("PROJECT ANUBHUTI — FULL PIPELINE")
    print(f"  Draft script: {len(DRAFT_SCRIPT.split())} words")

    # ---- Stage 1: canon retrieval -----------------------------------------
    banner("STAGE 1 — LORE CONTINUITY CHECK")
    canon = []
    if args.skip_lore:
        print("  Skipped (--skip-lore).")
    else:
        canon = fetch_canon(DRAFT_SCRIPT, character_id=args.character_id)
        print(format_canon_warnings(canon))

    # ---- Stage 2: expert panel --------------------------------------------
    banner("STAGE 2 — WRITERS ROOM CRITIQUE")
    try:
        critique = analyze_script(
            DRAFT_SCRIPT,
            character_id=args.character_id,
            use_lore=not args.skip_lore,
        )
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 1

    print(f"  Director    : {critique.director_notes[:140]}...")
    print(f"  Editor      : {critique.editor_notes[:140]}...")
    print(f"  Psychologist: {critique.psychologist_notes[:140]}...")
    print(f"  Continuity  : {critique.continuity_critique[:140]}...")
    print(f"  Foley cues  : {len(critique.foley_triggers)}")

    # ---- Stage 3: synthetic audience --------------------------------------
    banner("STAGE 3 — SYNTHETIC AUDIENCE SIMULATION")
    try:
        simulator = AudienceSimulator()
        report = simulator.simulate_audience(DRAFT_SCRIPT)
    except (AudienceSimulatorError, ValueError) as exc:
        print(f"  FAILED: {exc}")
        return 1

    print_heatmap(report)

    reasons = [
        (e.minute, k, r)
        for e in report.heatmap
        for k, r in e.drop_off_reasons.items()
    ]
    if reasons:
        print("\n  Drop-off reasons:")
        for minute, persona, reason in reasons:
            print(f"    [min {minute}] {persona}: {reason[:110]}")

    # ---- Stage 4: auto-rewrite --------------------------------------------
    banner("STAGE 4 — AUTO-REWRITE LOOP")
    final_script = DRAFT_SCRIPT
    try:
        rewrite = rewrite_weak_segments(
            DRAFT_SCRIPT,
            report.heatmap,
            critique,
            canon=canon,
            threshold=DROP_OFF_THRESHOLD,
        )
    except (RewriteEngineError, ValueError) as exc:
        print(f"  FAILED: {exc}")
        return 1

    if rewrite is None:
        print(f"  No minute flagged against threshold {DROP_OFF_THRESHOLD}. Skipped.")
    else:
        final_script = rewrite.rewritten_script
        print(f"  Rewrote {len(rewrite.change_log)} segment(s):")
        for change in rewrite.change_log:
            print(f"\n    MINUTE {change.minute}")
            print(f"      problem: {change.problem[:120]}")
            print(f"      change : {change.change_made[:120]}")
        print(f"\n  Tone/canon: {rewrite.tone_continuity_note[:200]}")
        print(
            f"\n  Length: {len(DRAFT_SCRIPT.split())} words "
            f"-> {len(final_script.split())} words"
        )

    # ---- Stage 5: verification re-simulation ------------------------------
    if rewrite is not None and not args.skip_resim:
        banner("STAGE 5 — RE-SIMULATION OF REWRITTEN SCRIPT")
        try:
            after = simulator.simulate_audience(final_script)
            print_heatmap(after)
            delta = after.overall_average - report.overall_average
            print(f"\n  Engagement delta: {delta:+.2f}")
        except AudienceSimulatorError as exc:
            print(f"  Re-simulation failed: {exc}")

    # ---- Stage 6: audio export --------------------------------------------
    banner("STAGE 6 — AUDIO SYNTHESIS & PRODUCTION MANIFEST")
    try:
        synthesizer = AudioSynthesizer(output_dir=args.output_dir)
        manifest = synthesizer.synthesize(
            final_script,
            critique.foley_triggers,
            generate_audio=not args.skip_audio,
        )
    except (SynthesisError, ValueError) as exc:
        print(f"  FAILED: {exc}")
        return 1

    print(f"  Chunks         : {manifest['chunk_count']}")
    print(f"  Runtime        : {manifest['total_runtime']}")
    print(f"  Audio rendered : {manifest['audio_generated']}")
    print(f"  Casting        : {json.dumps(manifest['casting'])}")
    print(f"  Manifest       : {manifest['manifest_path']}")
    print(f"  Cue sheet      : {manifest['cue_sheet_path']}")

    if manifest["unassigned_foley"]:
        print(f"\n  Unplaced cues  : {len(manifest['unassigned_foley'])}")
        for cue in manifest["unassigned_foley"]:
            print(f"    [{cue['timestamp']}] {cue['sound_effect']} ({cue['reason']})")

    print("\n  Cue sheet preview:")
    print("  " + "-" * 74)
    preview = Path(manifest["cue_sheet_path"]).read_text(encoding="utf-8")
    for line in preview.splitlines()[:24]:
        print(f"  {line}")

    banner("PIPELINE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
