"""
Score the story checker against fixtures whose faults are known in advance.

A plot-hole finder cannot be judged by reading its output and nodding. Every
fixture under samples/fixtures/ declares what should be found and, just as
importantly, what should not be, and this grades one against the other.

    python scripts/grade_fixtures.py                 # everything
    python scripts/grade_fixtures.py ardmore halberd # named fixtures
    python scripts/grade_fixtures.py --no-recheck    # skip the second pass

Two phases per fixture:

  sequential  each part is checked against canon as it stood before it, which
              is what the composer does.
  recheck     every part is checked again once the whole story is on the page,
              which is the only way a reference to something established later
              can be seen at all.

Grading is strict on purpose. A finding that matches no declared trap counts
against precision, because a warning a writer has to dismiss is worse than no
warning: it teaches them to stop reading the panel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from projects.service import StoryService
from projects.store import ProjectStore

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "samples" / "fixtures"

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
RED, AMBER, GREEN = "\033[31m", "\033[33m", "\033[32m"


@dataclass
class Score:
    hits: list[str] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)
    wrong_kind: list[str] = field(default_factory=list)
    ledger: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        wanted = len(self.hits) + len(self.misses)
        return len(self.hits) / wanted if wanted else 1.0

    @property
    def precision(self) -> float:
        raised = len(self.hits) + len(self.false_positives)
        return len(self.hits) / raised if raised else 1.0


def load(name: str) -> tuple[dict, list[Path]]:
    spec = json.loads((FIXTURES / name / "expectations.json").read_text())
    where = FIXTURES / name
    if spec.get("parts_from"):
        where = (where / spec["parts_from"]).resolve()
    return spec, sorted(where.glob("part_*.txt"))


def matches(finding, trap: dict) -> bool:
    """A trap is met when the right kind of finding names the right thing."""
    if finding.kind != trap["expect"]:
        return False
    blob = " ".join(
        [finding.what, finding.established, finding.quote]
    ).lower()
    return any(word.lower() in blob for word in trap["match"])


def grade_part(findings, part: int, spec: dict, phase: str, score: Score) -> None:
    traps = [
        t
        for t in spec["traps"]
        if t["part"] == part and t.get("phase", "sequential") == phase
    ]
    forbidden = [t for t in spec.get("must_not_fire", []) if t["part"] == part]

    claimed: set[int] = set()
    for trap in traps:
        for i, finding in enumerate(findings):
            if i not in claimed and matches(finding, trap):
                claimed.add(i)
                score.hits.append(f"part {part}: {trap['id']}")
                break
        else:
            near = [f for f in findings if f.kind != trap["expect"]
                    and any(w.lower() in f.what.lower() for w in trap["match"])]
            if near:
                score.wrong_kind.append(
                    f"part {part}: {trap['id']} — found as "
                    f"{near[0].kind}, wanted {trap['expect']}"
                )
                claimed.add(findings.index(near[0]))
                score.hits.append(f"part {part}: {trap['id']} (wrong kind)")
            else:
                score.misses.append(f"part {part}: {trap['id']} — {trap['why']}")

    # Some fixtures are mysteries, and a mystery owes its reader open threads.
    # Counting each one as a false positive would score the checker down for
    # reading the story correctly, so a fixture may declare how many of a kind
    # it expects to be told about.
    allowance = {t["kind"]: t["up_to"] for t in spec.get("tolerate", [])}

    for i, finding in enumerate(findings):
        if i in claimed:
            continue
        if allowance.get(finding.kind, 0) > 0:
            allowance[finding.kind] -= 1
            continue
        why = forbidden[0]["why"] if forbidden else "nothing here is wrong"
        score.false_positives.append(
            f"part {part} [{finding.kind}] {finding.what[:88]}\n"
            f"      {DIM}should not fire: {why}{OFF}"
        )


def show(findings) -> None:
    for f in findings:
        print(f"    {DIM}· [{f.kind}] {f.what[:78]}{OFF}")


def run_fixture(name: str, service: StoryService, store: ProjectStore,
                recheck: bool) -> Score:
    spec, parts = load(name)
    score = Score()

    print(f"\n{BOLD}{'═' * 78}{OFF}")
    print(f"{BOLD}{spec['title']}{OFF}  ·  {len(parts)} parts  ·  {spec['form']}")
    print(f"{DIM}{spec['purpose']}{OFF}")
    print("═" * 78)

    for existing in store.list_projects():
        if existing.title == spec["title"]:
            store.delete_project(existing.id)

    project, main = service.create_project(
        spec["title"], story_so_far=parts[0].read_text(), split_existing=False
    )

    for n, path in enumerate(parts[1:], start=2):
        text = path.read_text()
        position = store.next_position(project.id, main.id)
        report = service.check_draft(
            project.id, main.id, text, position=position, use_llm=True
        )
        print(f"\n  part {n}: {len(report.findings)} findings")
        show(report.findings)
        grade_part(report.findings, n, spec, "sequential", score)
        service.finalise_part(project.id, main.id, text)

    if recheck:
        print(f"\n  {DIM}— rechecking every part against the finished story —{OFF}")
        for n, path in enumerate(parts, start=1):
            wanted = [
                t for t in spec["traps"]
                if t.get("phase") == "recheck" and t["part"] == n
            ]
            if not wanted:
                continue
            report = service.check_draft(
                project.id, main.id, path.read_text(),
                position=n - 1, use_llm=True,
            )
            print(f"\n  part {n} rechecked: {len(report.findings)} findings")
            show(report.findings)
            grade_part(report.findings, n, spec, "recheck", score)

    check_ledger(spec, service, store, project.id, main.id, score)
    return score


def check_ledger(spec, service, store, project_id, branch_id, score: Score) -> None:
    """Supersession and thread-closing are graded off the ledger, not findings."""
    facts = service.active_facts(project_id, branch_id)
    live = " ".join(f"{f.subject} {f.claim}".lower() for f in facts
                    if f.kind != "open_question")
    threads = " ".join(f.claim.lower() for f in facts if f.kind == "open_question")

    for want in spec.get("expect_superseded", []):
        about = want["about"].lower()
        stale = [
            f for f in facts
            if about in f"{f.subject} {f.claim}".lower() and f.kind != "open_question"
        ]
        score.ledger.append(
            f"{GREEN}ok{OFF}   supersession about '{about}': "
            f"{len(stale)} live claim(s)"
            if len(stale) <= 2
            else f"{AMBER}warn{OFF} '{about}' still has {len(stale)} live claims; "
                 f"expected the old ones to be retired"
        )

    for want in spec.get("expect_threads_closed", []):
        about = want["about"].lower()
        if about in threads:
            score.ledger.append(
                f"{RED}fail{OFF} thread about '{about}' is still open, "
                f"though {want['why']}"
            )
        else:
            score.ledger.append(f"{GREEN}ok{OFF}   thread about '{about}' closed")


def report(name: str, score: Score) -> None:
    print(f"\n{BOLD}  {name}: {len(score.hits)} found, {len(score.misses)} missed, "
          f"{len(score.false_positives)} false{OFF}")
    print(f"  recall {score.recall:.0%}   precision {score.precision:.0%}")

    for hit in score.hits:
        print(f"    {GREEN}found{OFF}   {hit}")
    for note in score.wrong_kind:
        print(f"    {AMBER}kind{OFF}    {note}")
    for miss in score.misses:
        print(f"    {RED}MISSED{OFF}  {miss}")
    for fp in score.false_positives:
        print(f"    {RED}FALSE{OFF}   {fp}")
    for line in score.ledger:
        print(f"    {line}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--no-recheck", action="store_true")
    args = parser.parse_args()

    names = args.names or sorted(
        p.parent.name for p in FIXTURES.glob("*/expectations.json")
    )
    store = ProjectStore()
    service = StoryService(
        store=store,
        canon_store=None,
        openai_client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    )

    scores = {n: run_fixture(n, service, store, not args.no_recheck) for n in names}

    print(f"\n{BOLD}{'═' * 78}{OFF}")
    print(f"{BOLD}Scoreboard{OFF}")
    print("═" * 78)
    for name, score in scores.items():
        report(name, score)

    found = sum(len(s.hits) for s in scores.values())
    missed = sum(len(s.misses) for s in scores.values())
    false = sum(len(s.false_positives) for s in scores.values())
    print(f"\n{BOLD}Overall: {found} found, {missed} missed, {false} false "
          f"positives{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
