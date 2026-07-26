"""
Render every dashboard page with seeded data and report any exception.

Catches what an HTTP 200 cannot: errors inside the page functions themselves.
Run after any UI change; the offline suite cannot reach page code.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))

from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard.state import DB_PATH  # noqa: E402
from projects.store import ProjectStore  # noqa: E402

DRAFT = "INT. BASEMENT - NIGHT\n\nShe opens the door and steps onto the concrete."

HARNESS = '''
import sys
sys.path.insert(0, {src!r})

import streamlit as st
from dashboard import state, theme
from dashboard.views import {module}

theme.apply()
state.init()
st.session_state["project_id"] = {project!r}
st.session_state["branch_id"] = {branch!r}
st.session_state["draft"] = {draft!r}
st.session_state["_pages"] = {{}}
st.session_state["start_from"] = {start!r}

{module}.render()
'''


def main() -> int:
    store = ProjectStore(DB_PATH)
    for stale in [p for p in store.list_projects() if p.title == "Smoke Test Story"]:
        store.delete_project(stale.id)

    project, main_branch = store.create_project(
        "Smoke Test Story", "Seeded so every page has something to draw."
    )
    store.add_segment(project.id, main_branch.id, "Part one. The door is locked.")
    store.add_segment(project.id, main_branch.id, "Part two. The key turns.")
    branch = store.create_branch(main_branch.id, 1, "The key stays lost")
    store.add_segment(project.id, branch.id, "Part two, differently.")
    store.close()

    # The workspace has two modes and a fork variant, and each renders
    # different code, so each is exercised rather than only the first screen.
    at_tip = {
        "branch_id": main_branch.id,
        "position": 1,
        "will_fork": False,
        "fork_name": "",
    }
    mid_story = {
        "branch_id": main_branch.id,
        "position": 0,
        "will_fork": True,
        "fork_name": "An alternative",
    }

    cases = [
        ("library", None),
        ("workspace", None),  # choosing where to continue
        ("workspace", at_tip),  # the desk, extending this timeline
        ("workspace", mid_story),  # the desk, about to fork
        ("timeline", None),
        ("production", at_tip),
        ("settings", None),
    ]

    failures = 0
    for module, start in cases:
        code = HARNESS.format(
            src=str(SRC),
            module=module,
            project=project.id,
            branch=main_branch.id,
            draft=DRAFT,
            start=start,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(code)
            path = handle.name

        app = AppTest.from_file(path, default_timeout=120)
        app.run()

        if start is None:
            label = module
        elif start["will_fork"]:
            label = f"{module} (forking)"
        else:
            label = f"{module} (writing)"

        if app.exception:
            failures += 1
            print(f"  FAIL {label}: {str(app.exception[0].value)[:300]}")
        else:
            print(
                f"  ok   {label:21} {len(app.markdown):3} markdown, "
                f"{len(app.button):2} buttons, {len(app.expander):2} expanders"
            )

        Path(path).unlink(missing_ok=True)

    store = ProjectStore(DB_PATH)
    store.delete_project(project.id)
    store.close()

    print("\nall pages render" if not failures else f"\n{failures} page(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
