"""
The workbench, in two steps.

Opening a project asks one question before anything else: **where in the story
are you continuing from?** Only once that is answered does the desk appear.

That order is deliberate. In a branching story "where" is not obvious and it
changes everything downstream — which parts you inherit, which facts are true,
and whether finalising extends this timeline or starts a new one. Dropping
someone straight into a text box forces them to infer all of that from a
sidebar, and they will get it wrong.

Once writing, the page is a desk rather than a report: recap and canon sit
behind chips you open when you want them, the writing surface is the only thing
with real estate, and results appear underneath only after you ask for them.
"""

from __future__ import annotations

import streamlit as st

from dashboard import brand, editor, language, readthrough, state, theme
from dashboard.views import timeline
from writers_room.genre_rewrite import (
    GENRE_REWRITE_TARGETS,
    GenreRewriteError,
    rewrite_as_genre,
)


def render() -> None:
    state.drain_messages()

    project = state.current_project()
    branch = state.current_branch()
    if project is None or branch is None:
        _no_project()
        return

    if st.session_state.get("start_from") is None:
        _choose_start(project)
    else:
        _workbench(project)


def _no_project() -> None:
    brand.page_hero(
        "Nothing open",
        "Pick a story from the Library, or start a new one.",
        eyebrow="Pocket FM · Write",
    )
    if st.button("Go to the Library", type="primary"):
        state.go("library")


# ---------------------------------------------------------------------------
# Step one: where are you continuing from?
# ---------------------------------------------------------------------------


def _choose_start(project) -> None:
    store = state.get_store()
    branches = store.list_branches(project.id)
    segments = store.list_segments(project.id)

    brand.page_hero(
        project.title,
        "Pick the part you are continuing from. Everything before it is the "
        "story you are writing into; everything after it is set aside.",
        eyebrow="Pocket FM · Choose a starting point",
    )

    if not segments:
        brand.callout(
            "calm",
            "This will be the opening",
            "Nothing is on the page yet. Starting here creates Part 1 on the main timeline.",
        )
        if st.button("Start writing", type="primary"):
            main = branches[0]
            state.set_start(main.id, -1, will_fork=False)
            st.rerun()
        return

    brand.section_label("Story map")

    timeline.render_graph(
        store.graph(project.id), branches, segments, key="start_graph"
    )

    branch = state.current_branch()
    visible = store.story_so_far(project.id, branch.id)
    tip = max((s.position for s in visible), default=-1)

    options = {s.position: s for s in visible}
    default = list(options).index(tip) if tip in options else len(options) - 1

    st.write("")
    picker = st.columns([2, 3])
    with picker[0]:
        chosen = st.selectbox(
            "Continue after",
            options=list(options),
            format_func=lambda p: language.part_label(p),
            index=default,
            key="start_pick",
        )
    with picker[1]:
        st.write("")
        st.caption(
            f"On **{branch.name}**. Click another timeline in the map above to "
            "continue from that version instead."
        )

    will_fork = chosen < tip
    _start_card(project, branch, chosen, tip, will_fork)


def _start_card(project, branch, chosen: int, tip: int, will_fork: bool) -> None:
    """Say plainly what pressing the button will do."""
    next_part = chosen + 2  # zero-based position to one-based part number

    if not will_fork:
        brand.callout(
            "calm",
            f"Part {next_part} of {branch.name}",
            f"This continues the timeline you are on. Finalising adds "
            f"Part {next_part} to the end of it.",
        )
        fork_name = ""
    else:
        brand.callout(
            "warn",
            "This starts a new timeline",
            f"{branch.name} already continues past Part {chosen + 1}. "
            f"Nothing is overwritten — when you finalise, this becomes Part "
            f"{next_part} of a new timeline that shares everything up to Part "
            f"{chosen + 1}.",
        )
        fork_name = st.text_input(
            "Call the new timeline",
            placeholder="She never gets the key",
            help="You can rename it later on the Timelines page.",
            key="start_fork_name",
        )

    if st.button("Start writing", type="primary"):
        if will_fork and not fork_name.strip():
            st.error("Give the new timeline a name so you can tell them apart.")
            return
        state.set_start(
            branch.id, chosen, will_fork=will_fork, fork_name=fork_name.strip()
        )
        st.rerun()


# ---------------------------------------------------------------------------
# Step two: the desk
# ---------------------------------------------------------------------------


def _workbench(project) -> None:
    store = state.get_store()
    service = state.get_service()

    start = st.session_state.start_from
    branch = store.get_branch(start["branch_id"])
    if branch is None:
        state.clear_start()
        st.rerun()
        return

    upto = start["position"]
    part_number = upto + 2
    parts = store.story_so_far(project.id, branch.id, upto + 1)

    _bar(project, branch, part_number, start)
    _context_chips(service, project, branch, parts, upto + 1)

    st.write("")
    characters = editor.known_characters(parts)
    draft = editor.render(characters)

    st.write("")
    _actions(service, project, branch, draft, start)

    _results()


def _bar(project, branch, part_number: int, start) -> None:
    left, mid, right = st.columns([4.2, 1.6, 1.6])

    with left:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.35rem">
              {brand.mark_img(26)}
              <div style="font-size:.78rem;color:{theme.FAINT};letter-spacing:.04em;
                   text-transform:uppercase;font-weight:600">
                Pocket FM · Desk
              </div>
            </div>
            <div style="font-size:.82rem;margin-bottom:.15rem;color:{theme.MUTED}">
              <span style="color:{theme.FAINT}">{project.title}</span>
              <span style="color:{theme.FAINT}"> / </span>
              <span>{branch.name}</span>
            </div>
            <div style="font-size:1.7rem;font-weight:650;color:{theme.INK};
                 letter-spacing:-.035em;line-height:1.1">Part {part_number}</div>
            """,
            unsafe_allow_html=True,
        )
        if start["will_fork"]:
            st.markdown(
                theme.pill(
                    f"Finalising starts a new timeline: {start['fork_name']}", "accent"
                ),
                unsafe_allow_html=True,
            )

    with mid:
        st.write("")
        st.caption("Context chips below open only when you need them.")

    with right:
        st.write("")
        if st.button("Change start point", width="stretch"):
            state.clear_start()
            st.rerun()


def _context_chips(service, project, branch, parts, upto: int) -> None:
    """
    Recap, canon, and open threads, behind chips.

    All three are needed occasionally and none is needed constantly. On the
    page at all times they push the writing surface below the fold, which is
    the wrong trade for something you glance at twice an hour.
    """
    store = state.get_store()
    facts = service.active_facts(project.id, branch.id, upto)
    established = [f for f in facts if f.kind != "open_question"]
    questions = [f for f in facts if f.kind == "open_question"]

    synopsis, _ = store.get_synopsis(branch.id)
    words = sum(p.word_count for p in parts)

    row = st.columns([1.1, 1.1, 1.1, 2.7])

    with row[0]:
        with st.popover(
            f"Story so far · {language.plural(len(parts), 'part')}", width="stretch"
        ):
            st.markdown("#### Where the story stands")
            if synopsis:
                st.markdown(
                    f"<div class='anu-story' style='font-size:.94rem'>{synopsis}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No recap yet. It is written when you finalise a part.")

            st.caption(f"{language.plural(len(parts), 'part')} · {words:,} words")
            st.divider()
            for part in reversed(parts):
                with st.expander(language.part_label(part.position)):
                    st.markdown(
                        f"<div class='anu-story'>{_escape(part.text)}</div>",
                        unsafe_allow_html=True,
                    )

    with row[1]:
        with st.popover(
            f"Established · {len(established)}", width="stretch"
        ):
            st.markdown("#### True at this point")
            st.caption(language.CANON_NOTE)
            if not established:
                st.caption("Nothing recorded yet.")
            by_kind: dict[str, list] = {}
            for fact in established:
                by_kind.setdefault(fact.kind, []).append(fact)
            for kind, group in by_kind.items():
                st.markdown(
                    f"<div class='anu-meta' style='margin-top:.5rem'>{kind}</div>",
                    unsafe_allow_html=True,
                )
                for fact in group:
                    st.markdown(
                        f"<div style='font-size:.85rem;margin:.15rem 0'>"
                        f"<strong>{_escape(fact.subject)}</strong> — "
                        f"{_escape(fact.claim)}"
                        f"<span style='color:{theme.FAINT}'> · part "
                        f"{fact.established_position + 1}</span></div>",
                        unsafe_allow_html=True,
                    )

    with row[2]:
        label = f"Open threads · {len(questions)}"
        with st.popover(label, width="stretch"):
            st.markdown("#### Raised and not answered")
            if not questions:
                st.caption("Nothing outstanding.")
            for fact in questions:
                st.markdown(
                    f"<div style='font-size:.88rem;margin:.3rem 0'>"
                    f"{_escape(fact.claim)}"
                    f"<span style='color:{theme.FAINT}'> · opened in part "
                    f"{fact.established_position + 1}</span></div>",
                    unsafe_allow_html=True,
                )

    with row[3]:
        if questions:
            st.markdown(
                f"<div style='padding-top:.5rem;font-size:.8rem;color:{theme.FAINT}'>"
                f"{language.plural(len(questions), 'thread')} waiting on an answer"
                f"</div>",
                unsafe_allow_html=True,
            )


def _actions(service, project, branch, draft: str, start) -> None:
    row = st.columns([1, 1, 1, 2])
    empty = not draft.strip()

    with row[0]:
        if st.button(
            "Check the story",
            width="stretch",
            disabled=empty,
            help="Anything that contradicts what this timeline has established.",
        ):
            _check(service, project.id, branch.id, draft, start["position"] + 1)
            st.rerun()

    with row[1]:
        if st.button(
            "Read it through",
            width="stretch",
            disabled=empty,
            help="Where a reader is likely to drift, and how hard the ending pulls.",
        ):
            readthrough.run(draft, is_revision=st.session_state.readthrough is not None)
            st.rerun()

    with row[2]:
        label = "Finalise and branch" if start["will_fork"] else "Finalise this part"
        if st.button(label, type="primary", width="stretch", disabled=empty):
            _finalise(service, project, branch, draft, start)
            st.rerun()

    with row[3]:
        if start["will_fork"]:
            st.markdown(
                f"<div style='padding-top:.55rem;font-size:.78rem;color:{theme.MUTED}'>"
                f"Finalising creates <strong>{_escape(start['fork_name'])}</strong> "
                f"and puts this part on it.</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    genre_row = st.columns([1.4, 1.2, 2.4])
    with genre_row[0]:
        target = st.selectbox(
            "Rewrite as",
            options=list(GENRE_REWRITE_TARGETS),
            format_func=lambda g: g.replace("_", " ").title(),
            key="genre_rewrite_target",
            help=(
                "Restyle the whole draft into this genre or style. Plot and "
                "established facts stay fixed."
            ),
            disabled=empty,
        )
    with genre_row[1]:
        st.write("")
        if st.button(
            "Rewrite as…",
            width="stretch",
            disabled=empty,
            help="Transform tone and style while preserving the core plot.",
        ):
            _genre_rewrite(
                service, project.id, branch.id, draft, start, target
            )
            st.rerun()
    with genre_row[2]:
        st.write("")
        st.caption(
            "Genre rewrite restyles the draft. It is not the weak-minute "
            "punch-up on Production."
        )


def _results() -> None:
    have_genre = st.session_state.get("genre_rewrite") is not None
    have_check = st.session_state.continuity is not None
    have_read = st.session_state.readthrough is not None
    if not (have_check or have_read or have_genre):
        return

    st.write("")
    st.divider()

    if have_genre:
        _genre_rewrite_result(st.session_state.genre_rewrite)

    if have_check and have_read:
        tabs = st.tabs(["Story check", "Read-through"])
        with tabs[0]:
            readthrough.render_story_check(st.session_state.continuity)
        with tabs[1]:
            readthrough.render_readthrough(
                st.session_state.readthrough,
                baseline=st.session_state.readthrough_baseline,
                comparison=st.session_state.comparison,
            )
    elif have_check:
        readthrough.render_story_check(st.session_state.continuity)
    elif have_read:
        readthrough.render_readthrough(
            st.session_state.readthrough,
            baseline=st.session_state.readthrough_baseline,
            comparison=st.session_state.comparison,
        )


def _genre_rewrite_result(result) -> None:
    label = (result.target_genre or "").replace("_", " ").title()
    with st.expander(f"Genre rewrite · {label}", expanded=True):
        st.write(result.plot_preservation_note)
        if result.change_log:
            st.markdown("**What changed**")
            for entry in result.change_log:
                st.markdown(f"- **{entry.aspect}** — {entry.change_made}")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _check(service, project_id: str, branch_id: str, draft: str, position: int) -> None:
    with st.spinner("Checking against the story so far..."):
        try:
            st.session_state.continuity = service.check_draft(
                project_id, branch_id, draft, position=position
            )
        except Exception as exc:
            state.fail(f"The story check could not run: {exc}")


def _genre_rewrite(
    service, project_id: str, branch_id: str, draft: str, start, target: str
) -> None:
    """Restyle the draft, replace the composer, then re-check continuity."""
    position = start["position"] + 1
    with st.spinner(f"Rewriting as {target}..."):
        try:
            pack = service.context_for(project_id, branch_id, draft=draft)
            context_text = pack.to_prompt() if pack is not None else ""
            result = rewrite_as_genre(
                draft,
                target,
                context_pack=context_text,
            )
        except (GenreRewriteError, ValueError) as exc:
            state.fail(f"Genre rewrite failed: {exc}")
            return
        except Exception as exc:
            state.fail(f"Genre rewrite failed: {exc}")
            return

    st.session_state.genre_rewrite = result
    st.session_state.draft = result.rewritten_script
    st.session_state.draft_version += 1
    # Stale forecast / comparison describe the previous draft.
    st.session_state.readthrough = None
    st.session_state.readthrough_baseline = None
    st.session_state.comparison = None
    st.session_state.preview_manifest = None

    _check(service, project_id, branch_id, result.rewritten_script, position)
    state.flash(
        f"Draft restyled as {target}. Story check ran against the rewrite."
    )


def _finalise(service, project, branch, draft: str, start) -> None:
    """
    Commit the part, forking first when the writer chose a mid-story point.

    The fork is created here rather than when they picked the point, so that
    abandoning a draft leaves no empty timeline behind.
    """
    target = branch

    if start["will_fork"]:
        with st.spinner("Starting the new timeline..."):
            try:
                target = service.create_branch(
                    project.id, branch.id, start["position"] + 1, start["fork_name"]
                )
            except Exception as exc:
                state.fail(f"Could not start the new timeline: {exc}")
                return

    with st.spinner("Adding it to the story..."):
        try:
            result = service.finalise_part(project.id, target.id, draft)
        except Exception as exc:
            state.fail(f"Could not save the part: {exc}")
            return

    st.session_state.branch_id = target.id
    state.clear_start()
    state.reset_work()

    message = result.summary
    if start["will_fork"]:
        message = f"'{target.name}' started. " + message
    if result.superseded_claims:
        message += " Some earlier facts no longer hold."
    state.flash(message)

    for warning in result.warnings:
        st.warning(warning)


def _escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
