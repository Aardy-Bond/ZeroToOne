"""
Run the whole Kestrel fixture end to end, without pasting anything by hand.

Creates the project, walks the six parts in order, and before each finalise
runs the story check against canon as it stood at that moment. Then it forks
from part three and shows the same draft being judged differently on the two
timelines, which is the thing branch-scoped canon exists to do.

    python scripts/run_kestrel.py             # local ledger only
    python scripts/run_kestrel.py --databricks  # also ingest passages
    python scripts/run_kestrel.py --keep       # do not delete an earlier run

Every trap it reports is documented in samples/kestrel/EXPECTED.md, so the
output can be graded rather than admired.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from projects.service import StoryService
from projects.store import ProjectStore

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
PARTS = sorted((ROOT / "samples" / "kestrel").glob("part_*.txt"))
TITLE = "The Kestrel Light"

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
RED, AMBER, GREEN, BLUE = "\033[31m", "\033[33m", "\033[32m", "\033[34m"

KIND_LABEL = {
    "contradiction": "CONTRADICTS CANON",
    "premature_reference": "HAPPENS BEFORE IT SHOULD",
    "dangling_question": "ASKED, NOT ANSWERED",
    "unasked_answer": "ANSWERS SOMETHING NOBODY ASKED",
}
KIND_COLOUR = {
    "contradiction": RED,
    "premature_reference": AMBER,
    "dangling_question": BLUE,
    "unasked_answer": BLUE,
}


def rule(text: str = "", char: str = "─") -> None:
    if text:
        print(f"\n{BOLD}{text}{OFF}")
        print(char * 78)
    else:
        print(char * 78)


def show_findings(report) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in report.findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1

    if not report.findings:
        print(f"  {GREEN}clean{OFF} — {report.note or 'nothing conflicts'}")
        return counts

    for finding in report.findings:
        colour = KIND_COLOUR.get(finding.kind, "")
        label = KIND_LABEL.get(finding.kind, finding.kind)
        print(f"  {colour}{label}{OFF}  ({finding.severity})")
        print(f"    {finding.what}")
        if finding.established:
            part = (
                f"part {finding.established_part + 1}"
                if finding.established_part is not None
                else "canon"
            )
            print(f"    {DIM}canon ({part}): {finding.established}{OFF}")
        if finding.quote and finding.quote != finding.established:
            print(f'    {DIM}"{finding.quote[:100]}"{OFF}')
    return counts


def show_facts(service, project_id: str, branch_id: str, position: int) -> None:
    facts = service.active_facts(project_id, branch_id, position)
    states = [f for f in facts if f.kind != "open_question"]
    questions = [f for f in facts if f.kind == "open_question"]

    print(f"  {BOLD}{len(states)}{OFF} facts true here, "
          f"{BOLD}{len(questions)}{OFF} threads open")
    for fact in states[-6:]:
        print(f"    {DIM}·{OFF} {fact.subject}: {fact.claim[:66]}")
    if len(states) > 6:
        print(f"    {DIM}...and {len(states) - 6} earlier{OFF}")
    for fact in questions:
        print(f"    {BLUE}?{OFF} {fact.claim[:70]}")


def render_forecast(story: str, charts: Path) -> None:
    """
    Run the engagement forecast over the finished story and draw its charts.

    The figures come out of the dashboard's own render functions rather than
    being rebuilt here, by intercepting `st.plotly_chart`. Redrawing them would
    mean maintaining a second version that could quietly disagree with what the
    app shows.
    """
    import logging

    logging.getLogger("streamlit").setLevel(logging.ERROR)
    import streamlit as st

    captured: list = []
    st.plotly_chart = lambda figure, **kwargs: captured.append(figure)

    from dashboard import forecast_view
    from retention_engine.orchestrator import run_forecast
    from retention_engine.target_cohort import DEFAULT_COHORT

    forecast = run_forecast(
        story,
        cohort=DEFAULT_COHORT,
        client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        use_llm_deep_dive=False,
    )

    curve = forecast.primary_curve
    print(f"  survival proxy ends at {BOLD}{curve.final_survival:.0f}/100{OFF}")
    print(f"  {BOLD}{forecast.cliffhanger.unlock_pull_index:.0f}{OFF} unlock pull, "
          f"ending reads as {', '.join(forecast.cliffhanger.types) or 'unclassified'}")

    riskiest = sorted(curve.scenes, key=lambda s: s.hazard, reverse=True)[:3]
    print("\n  Where a listener is most likely to leave:")
    for scene in riskiest:
        drivers = ", ".join(c.factor for c in scene.contributions[:3])
        print(f"    scene {scene.scene_index + 1}: hazard {scene.hazard:.2f}  {DIM}{drivers}{OFF}")

    forecast_view.render_survival_curves(forecast)
    forecast_view.render_narrative_ekg(forecast)
    for figure, name in zip(captured, ("kestrel_survival.png", "kestrel_ekg.png")):
        figure.write_image(str(charts / name), width=1100, scale=2)
        print(f"\n  wrote output/charts/{name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--databricks", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--no-fork", action="store_true")
    parser.add_argument("--no-forecast", action="store_true")
    args = parser.parse_args()

    if not PARTS:
        print("samples/kestrel is empty.")
        return 1

    store = ProjectStore()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    canon = None
    if args.databricks:
        from projects.canon_store import CanonStore

        canon = CanonStore()
        print("Databricks passage store enabled.")

    if not args.keep:
        for existing in store.list_projects():
            if existing.title == TITLE:
                store.delete_project(existing.id)
                print(f"{DIM}Removed an earlier run of '{TITLE}'.{OFF}")

    service = StoryService(store=store, canon_store=canon, openai_client=client)

    rule(f"{TITLE} — {len(PARTS)} parts", "═")
    print("Part one becomes the story so far. Every later part is checked "
          "against canon\nbefore it is committed, exactly as the composer does it.")

    project, main = service.create_project(
        TITLE, story_so_far=PARTS[0].read_text(), split_existing=False
    )
    print(f"\n  project {DIM}{project.id}{OFF} on '{main.name}'")
    show_facts(service, project.id, main.id, 1)

    tally: dict[str, int] = {}

    for n, path in enumerate(PARTS[1:], start=2):
        text = path.read_text()
        position = store.next_position(project.id, main.id)

        rule(f"Part {n}  ·  {len(text.split())} words  ·  checked at position {position}")
        report = service.check_draft(
            project.id, main.id, text, position=position, use_llm=True
        )
        for kind, count in show_findings(report).items():
            tally[kind] = tally.get(kind, 0) + count

        result = service.finalise_part(project.id, main.id, text)
        print(f"\n  finalised: +{result.facts_added} facts, "
              f"−{result.facts_superseded} superseded")
        for claim in result.superseded_claims:
            print(f"    {AMBER}no longer true:{OFF} {claim[:66]}")
        for warning in result.warnings:
            print(f"    {AMBER}warning:{OFF} {warning}")

        show_facts(service, project.id, main.id, position + 1)

    # ---- the light, traced through the whole story ----------------------
    rule("The supersession chain", "═")
    print("Trap 1 in EXPECTED.md: the light should hold one state, not four.\n")
    everything = store.list_facts(project.id)
    light = [f for f in everything if "light" in f.subject.lower()
             or "lamp" in f.subject.lower()]
    for fact in sorted(light, key=lambda f: f.established_position):
        ended = (
            f"superseded in part {fact.superseded_position + 1}"
            if fact.superseded_position is not None
            else f"{GREEN}still true{OFF}"
        )
        print(f"  part {fact.established_position + 1}: {fact.claim[:56]:58} {ended}")

    # ---- the fork -------------------------------------------------------
    if not args.no_fork:
        rule("Branching: the same draft, judged on two timelines", "═")
        fork = service.create_branch(
            project.id, main.id, 3, "The lock is changed"
        )
        print(f"Forked '{fork.name}' after part 3.\n")

        divergent = (
            "Ilse took the second key down to the smith at Arden that same "
            "afternoon and had the lamp-room lock drawn and re-cut. She hung "
            "the single new key on the cord around her neck, under her "
            "collar, where nothing could take it without taking her."
        )
        service.finalise_part(project.id, fork.id, divergent)
        print(f"{DIM}Wrote a part 4 on the fork in which the lock is changed.{OFF}\n")

        ending = PARTS[-1].read_text()
        for branch in (main, fork):
            position = store.next_position(project.id, branch.id)
            report = service.check_draft(
                project.id, branch.id, ending, position=position, use_llm=True
            )
            print(f"{BOLD}The part six ending, checked on '{branch.name}':{OFF}")
            show_findings(report)
            print()

    # ---- the pictures ---------------------------------------------------
    charts = ROOT / "output" / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    try:
        from dashboard import branch_graph

        figure = branch_graph.build(
            store.graph(project.id),
            store.list_branches(project.id),
            store.list_segments(project.id),
        )
        figure.write_image(str(charts / "kestrel_branches.png"), width=1000, scale=2)
        print(f"\nBranch map: output/charts/kestrel_branches.png")
    except Exception as exc:
        print(f"{DIM}Branch map not rendered: {exc}{OFF}")

    if not args.no_forecast:
        rule("Reading the whole story for engagement risk", "═")
        print(f"{DIM}Part 3 is the planted exposition sag; see EXPECTED.md.{OFF}\n")
        render_forecast(service.story_text(project.id, main.id), charts)

    rule("What the checks found across the whole story", "═")
    if tally:
        for kind, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {count:2}  {KIND_LABEL.get(kind, kind)}")
    else:
        print("  nothing at all, which for this fixture means something is off")
    print(f"\n{DIM}Grade this against samples/kestrel/EXPECTED.md.{OFF}\n")

    print(f"Open the dashboard and pick '{TITLE}' to see it in the UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
