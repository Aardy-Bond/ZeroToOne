"""
Library: every project, and the door into a new one.

Starting a project is where the story so far gets ingested. That is the slow
step -- reading an existing back-catalogue means one extraction pass per part --
so it reports progress rather than spinning silently.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard import brand, language, state

SAMPLES = Path(__file__).resolve().parents[3] / "samples"


def render() -> None:
    state.drain_messages()
    store = state.get_store()
    projects = store.list_projects()

    brand.page_hero(
        "Your stories",
        "Each project keeps its own canon. Write the next part, and Anubhuti "
        "checks it against everything that timeline has already established.",
        eyebrow="Pocket FM · Library",
    )

    actions = st.columns([1.2, 1.2, 3.6])
    with actions[0]:
        if st.button("New story", type="primary", width="stretch"):
            new_story_dialog()
    with actions[1]:
        if projects and st.button("Open latest", width="stretch"):
            state.open_project(projects[0].id)
            state.go("workspace")

    if projects:
        total_parts = 0
        total_branches = 0
        for project in projects:
            branches = store.list_branches(project.id)
            total_branches += len(branches)
            if branches:
                total_parts += len(store.story_so_far(project.id, branches[0].id))
        brand.metric_strip(
            [
                ("Stories", str(len(projects)), "in this studio"),
                ("Parts", str(total_parts), "across main lines"),
                ("Timelines", str(total_branches), "including forks"),
            ]
        )

    if not projects:
        _empty_state()
        return

    brand.section_label("All projects")
    columns = st.columns(3, gap="medium")

    for index, project in enumerate(projects):
        branches = store.list_branches(project.id)
        parts = store.story_so_far(project.id, branches[0].id) if branches else []
        words = sum(p.word_count for p in parts)
        meta = f"{language.plural(len(parts), 'part')} · {words:,} words"

        with columns[index % 3]:
            st.markdown(
                brand.project_tile(
                    project.title,
                    project.logline or "No logline yet — open it and start writing.",
                    meta,
                    timelines=len(branches),
                ),
                unsafe_allow_html=True,
            )
            buttons = st.columns([3, 1])
            with buttons[0]:
                if st.button("Open", key=f"open_{project.id}", width="stretch"):
                    state.open_project(project.id)
                    state.go("workspace")
            with buttons[1]:
                if st.button("⋯", key=f"more_{project.id}", width="stretch"):
                    manage_dialog(project.id)


def _empty_state() -> None:
    brand.empty_panel(
        "Nothing here yet",
        "Start a story from scratch, or paste in one you have already begun "
        "and Anubhuti will read it into canon.",
    )
    middle = st.columns([1, 1.2, 1])
    with middle[1]:
        if st.button("Start your first story", type="primary", width="stretch"):
            new_story_dialog()


@st.dialog("New story", width="large")
def new_story_dialog() -> None:
    title = st.text_input("Title", placeholder="The House on Wexler Street")
    logline = st.text_input(
        "Logline", placeholder="One line, so you recognise it in the list."
    )

    st.markdown("**The story so far**")
    st.caption(
        "Paste what you have already written, if anything. Anubhuti reads it "
        "into canon so later parts can be checked against it. Leave it empty to "
        "start from a blank page."
    )

    source = st.radio(
        "Source",
        ["Paste it", "Upload a file", "Use the sample story", "Start empty"],
        horizontal=True,
        label_visibility="collapsed",
    )

    story = ""
    if source == "Paste it":
        story = st.text_area(
            "Story so far", height=240, label_visibility="collapsed"
        )
    elif source == "Upload a file":
        uploaded = st.file_uploader("Upload", type=["txt", "md"], label_visibility="collapsed")
        if uploaded is not None:
            story = uploaded.read().decode("utf-8", errors="replace")
            st.caption(language.word_count_line(story))
    elif source == "Use the sample story":
        sample = SAMPLES / "wexler_street_continuation.txt"
        if sample.exists():
            story = sample.read_text(encoding="utf-8")
            st.caption(f"Sample loaded · {language.word_count_line(story)}")

    split = True
    if story.strip():
        split = st.toggle(
            "Split it into separate parts",
            value=True,
            help=(
                "Recommended. Separate parts let you branch from any point in "
                "the story rather than only from the end."
            ),
        )

    st.write("")
    actions = st.columns([1, 1])
    with actions[0]:
        if st.button("Cancel", width="stretch"):
            st.rerun()
    with actions[1]:
        if st.button("Create", type="primary", width="stretch"):
            if not title.strip():
                st.error("Give the story a title.")
                return
            _create(title, logline, story, split)


def _create(title: str, logline: str, story: str, split: bool) -> None:
    service = state.get_service()

    if not story.strip():
        project, _branch = service.create_project(title, logline=logline)
        state.open_project(project.id)
        state.flash(f"'{project.title}' is ready. Write the opening.")
        st.rerun()
        return

    bar = st.progress(0.0, text="Reading your story...")

    def progress(done: int, total: int, message: str) -> None:
        bar.progress(min(1.0, done / max(1, total)), text=message)

    try:
        project, _branch = service.create_project(
            title,
            story,
            logline=logline,
            split_existing=split,
            progress=progress,
        )
    except Exception as exc:
        bar.empty()
        st.error(f"Could not read the story in: {exc}")
        return

    bar.empty()
    state.open_project(project.id)

    facts = service.active_facts(project.id, st.session_state.branch_id)
    state.flash(
        f"'{title}' is ready. Anubhuti noted "
        f"{language.plural(len(facts), 'fact')} it will hold you to."
    )
    st.rerun()


@st.dialog("Story settings")
def manage_dialog(project_id: str) -> None:
    store = state.get_store()
    project = store.get_project(project_id)
    if project is None:
        st.error("That story no longer exists.")
        return

    title = st.text_input("Title", value=project.title)
    logline = st.text_input("Logline", value=project.logline)

    if st.button("Save", type="primary", width="stretch"):
        store.rename_project(project_id, title, logline)
        state.flash("Saved.")
        st.rerun()

    st.divider()
    st.caption("Deleting a story removes every timeline and everything written in it.")
    if st.button("Delete this story", width="stretch"):
        if st.session_state.get("confirm_delete") == project_id:
            canon = state.get_canon_store()
            if canon is not None:
                try:
                    canon.delete_project(project_id)
                except Exception:
                    pass
            store.delete_project(project_id)
            if st.session_state.project_id == project_id:
                st.session_state.project_id = None
                st.session_state.branch_id = None
            state.flash(f"Deleted '{project.title}'.")
            st.rerun()
        else:
            st.session_state.confirm_delete = project_id
            st.warning("Press again to confirm.")
