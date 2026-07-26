"""
Shared resources and session state for the dashboard.

The store, the OpenAI client, and the Databricks canon store are all cached
across reruns. Streamlit re-executes the whole script on every interaction, so
building a WorkspaceClient in a page function would mean a network round trip
each time the writer types.

The canon store is allowed to fail. Databricks being unreachable costs the
passage layer, not the app: the fact ledger lives in SQLite and continuity
checking keeps working.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st

from projects.service import StoryService
from projects.store import ProjectStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "projects" / "anubhuti.db"

SESSION_DEFAULTS: dict = {
    "project_id": None,
    "branch_id": None,
    # Where in the story the writer is continuing from. None means they have
    # not chosen yet, which is what puts the map in front of the desk.
    "start_from": None,
    "draft": "",
    "draft_version": 0,
    "continuity": None,
    "genre_rewrite": None,
    "readthrough": None,
    "readthrough_baseline": None,
    "comparison": None,
    "preview_manifest": None,
    "producer_plan": None,
    "producer_lines": None,
    "producer_manifest": None,
    "producer_error": "",
    "open_producer": False,
    "flash": "",
    "error": "",
    "busy": "",
}


@st.cache_resource(show_spinner=False)
def get_store() -> ProjectStore:
    return ProjectStore(DB_PATH)


@st.cache_resource(show_spinner=False)
def get_openai():
    from projects.facts import build_client

    return build_client()


@st.cache_resource(show_spinner=False)
def get_canon_store():
    """
    The Databricks passage layer, or None.

    Returning None rather than raising is deliberate. A writer should be able
    to work on a train.
    """
    try:
        from projects.canon_store import CanonStore

        store = CanonStore(openai_client=get_openai())
        store.bootstrap()
        return store
    except Exception as exc:
        logger.info("Canon store unavailable, running local-only: %s", exc)
        return None


def get_service() -> StoryService:
    return StoryService(
        store=get_store(),
        openai_client=_safe_openai(),
        canon_store=get_canon_store() if _databricks_configured() else None,
    )


def _safe_openai():
    try:
        return get_openai()
    except Exception:
        return None


def _databricks_configured() -> bool:
    return bool(os.getenv("DATABRICKS_HOST", "").strip())


def init() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_project():
    pid = st.session_state.get("project_id")
    return get_store().get_project(pid) if pid else None


def current_branch():
    bid = st.session_state.get("branch_id")
    return get_store().get_branch(bid) if bid else None


def open_project(project_id: str, branch_id: str | None = None) -> None:
    """Switch projects, clearing everything that described the old one."""
    store = get_store()
    st.session_state.project_id = project_id

    if branch_id is None:
        branches = store.list_branches(project_id)
        branch_id = branches[0].id if branches else None

    st.session_state.branch_id = branch_id
    st.session_state.start_from = None
    reset_work()


def switch_branch(branch_id: str) -> None:
    st.session_state.branch_id = branch_id
    st.session_state.start_from = None
    reset_work()


def set_start(
    branch_id: str, position: int, *, will_fork: bool, fork_name: str = ""
) -> None:
    """
    Record where the writer is continuing from.

    `position` is the part they are continuing *after*, so the new part lands
    at `position + 1`. `will_fork` is decided here rather than at save time so
    the interface can say what finalising will do before a word is written.
    """
    st.session_state.start_from = {
        "branch_id": branch_id,
        "position": position,
        "will_fork": will_fork,
        "fork_name": fork_name,
    }
    reset_work()


def clear_start() -> None:
    st.session_state.start_from = None


def reset_work() -> None:
    """Clear results that no longer describe what is on screen."""
    st.session_state.draft = ""
    st.session_state.draft_version += 1
    st.session_state.continuity = None
    st.session_state.genre_rewrite = None
    st.session_state.readthrough = None
    st.session_state.readthrough_baseline = None
    st.session_state.comparison = None
    st.session_state.preview_manifest = None
    st.session_state.error = ""


def flash(message: str) -> None:
    st.session_state.flash = message


def fail(message: str) -> None:
    st.session_state.error = message


def drain_messages() -> None:
    if st.session_state.get("flash"):
        st.success(st.session_state.flash)
        st.session_state.flash = ""
    if st.session_state.get("error"):
        st.error(st.session_state.error)
        st.session_state.error = ""


def go(page: str) -> None:
    """
    Send the writer to another page.

    Pages are looked up from a registry the app populates at startup rather
    than imported, since a view importing the app module would be a cycle.
    """
    target = st.session_state.get("_pages", {}).get(page)
    if target is not None:
        st.switch_page(target)


def connection_status() -> list[tuple[str, str, str]]:
    """(label, state, detail) for the sidebar footer."""
    rows: list[tuple[str, str, str]] = []

    if os.getenv("OPENAI_API_KEY", "").strip():
        rows.append(("Writing assistant", "ok", "connected"))
    else:
        rows.append(("Writing assistant", "off", "add OPENAI_API_KEY to .env"))

    host = os.getenv("DATABRICKS_HOST", "").strip()
    if not host:
        rows.append(("Searchable canon", "off", "local only"))
    elif get_canon_store() is not None:
        rows.append(("Searchable canon", "ok", host.replace("https://", "")))
    else:
        rows.append(("Searchable canon", "warn", "unreachable, using local"))

    return rows
