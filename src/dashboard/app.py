"""
Project Anubhuti — Pocket FM Writers Room.

A writing room for serialised stories: keep a project, write the next part,
check it against what the story has already established, and branch when you
want to try a different direction.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
SRC_DIR = DASHBOARD_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
LOGO_PATH = DASHBOARD_DIR / "assets" / "pocket-fm.jpg"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from dashboard import brand, state, theme  # noqa: E402
from dashboard.views import library, production, settings, timeline, workspace  # noqa: E402


def render_sidebar() -> None:
    """
    The sidebar answers two questions and nothing else: which story am I in,
    and where am I in it.
    """
    store = state.get_store()
    brand.sidebar_brand()

    project = state.current_project()
    branch = state.current_branch()

    if project is not None:
        start = st.session_state.get("start_from")
        if start is not None:
            where = f"Writing part {start['position'] + 2}"
        else:
            where = "Choosing where to continue"

        st.sidebar.markdown(
            f"<div class='anu-meta'>Now writing</div>"
            f"<div style='font-size:1.02rem;font-weight:650;color:{theme.INK};"
            f"letter-spacing:-.02em;margin:.15rem 0 .1rem'>{project.title}</div>"
            f"<div style='font-size:.8rem;color:{theme.ACCENT};font-weight:550'>"
            f"{branch.name if branch else ''}</div>"
            f"<div style='font-size:.74rem;color:{theme.FAINT};margin:.25rem 0 .85rem'>"
            f"{where}</div>",
            unsafe_allow_html=True,
        )
        st.sidebar.divider()

    projects = store.list_projects()
    if projects:
        st.sidebar.markdown(
            "<div class='anu-meta' style='margin-bottom:.35rem'>Switch story</div>",
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Each keeps its own canon and timelines.")

        current = st.session_state.get("project_id")
        for entry in projects[:12]:
            selected = entry.id == current
            if st.sidebar.button(
                entry.title,
                key=f"nav_{entry.id}",
                width="stretch",
                type="primary" if selected else "secondary",
                disabled=selected,
            ):
                state.open_project(entry.id)
                st.switch_page(WORKSPACE_PAGE)

    st.sidebar.divider()
    st.sidebar.markdown(
        "<div class='anu-meta' style='margin-bottom:.45rem'>Studio status</div>",
        unsafe_allow_html=True,
    )

    chips = []
    for label, status, detail in state.connection_status():
        chips.append(
            f'<span class="pfm-status {status}">'
            f'<span class="dot"></span>{label}'
            f'<span class="detail">{detail}</span></span>'
        )
    st.sidebar.markdown(
        f'<div class="pfm-status-row" style="flex-direction:column;'
        f'align-items:stretch">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div style="margin-top:1.4rem;padding-top:.9rem;border-top:1px solid {theme.RULE}">
          <div style="display:flex;align-items:center;gap:.55rem;opacity:.92">
            {brand.mark_img(22)}
            <div style="font-size:.68rem;color:{theme.FAINT};line-height:1.35;
                 letter-spacing:.02em">
              Built for Pocket FM story teams
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Every view's entry point is called `render`, so the URL for each has to be
# given explicitly. Streamlit would otherwise infer all five from the callable
# name and reject the duplicates.
LIBRARY_PAGE = st.Page(
    library.render, title="Library", icon=":material/library_books:", default=True
)
WORKSPACE_PAGE = st.Page(
    workspace.render, title="Write", icon=":material/edit_note:", url_path="write"
)
TIMELINE_PAGE = st.Page(
    timeline.render,
    title="Timelines",
    icon=":material/account_tree:",
    url_path="timelines",
)
PRODUCTION_PAGE = st.Page(
    production.render,
    title="Production",
    icon=":material/graphic_eq:",
    url_path="production",
)
SETTINGS_PAGE = st.Page(
    settings.render, title="Settings", icon=":material/tune:", url_path="settings"
)


def main() -> None:
    page_icon = str(LOGO_PATH) if LOGO_PATH.exists() else ":material/auto_stories:"
    st.set_page_config(
        page_title="Anubhuti · Pocket FM",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.apply()
    state.init()

    # Pages are registered here so a view can send the writer somewhere else
    # without importing this module and creating a cycle.
    st.session_state["_pages"] = {
        "library": LIBRARY_PAGE,
        "workspace": WORKSPACE_PAGE,
        "timeline": TIMELINE_PAGE,
        "production": PRODUCTION_PAGE,
        "settings": SETTINGS_PAGE,
    }

    navigation = st.navigation(
        {
            "Story": [LIBRARY_PAGE, WORKSPACE_PAGE, TIMELINE_PAGE, PRODUCTION_PAGE],
            "Studio": [SETTINGS_PAGE],
        }
    )

    render_sidebar()
    navigation.run()


if __name__ == "__main__":
    main()
