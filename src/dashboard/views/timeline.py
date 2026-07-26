"""
Timelines: the branch tree, and what diverges between two of them.

The compare view is the payoff of the fact ledger. Two timelines that share an
opening can hold contradictory truths, and seeing exactly which facts differ is
more useful than diffing the prose, because it is the facts the checker will
hold the writer to.
"""

from __future__ import annotations

import streamlit as st

from dashboard import branch_graph, brand, language, state, theme


def render() -> None:
    state.drain_messages()

    project = state.current_project()
    if project is None:
        brand.page_hero(
            "Timelines",
            "Open a story first to see how its versions diverge.",
            eyebrow="Pocket FM · Timelines",
        )
        return

    store = state.get_store()
    branches = store.list_branches(project.id)
    graph = store.graph(project.id)
    segments = store.list_segments(project.id)
    facts = store.list_facts(project.id)

    brand.page_hero(
        "Timelines",
        f"{project.title} · {language.plural(len(branches), 'timeline')}. "
        "A branch inherits everything before its fork and nothing after it.",
        eyebrow="Pocket FM · Timelines",
    )

    if not segments:
        st.info("Nothing written yet. Write a part and it will appear here.")
        return

    render_graph(graph, branches, segments, key="timeline_graph")
    st.divider()
    _lane_details(graph, branches, segments, facts)

    if len(branches) > 1:
        st.divider()
        _compare(graph, branches, segments, facts)


def render_graph(graph, branches, segments, *, key: str) -> None:
    """Draw the branch graph and switch timelines when a part is clicked."""
    current = st.session_state.get("branch_id")
    figure = branch_graph.build(graph, branches, segments, current)

    event = st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": False},
    )

    st.markdown(branch_graph.legend(), unsafe_allow_html=True)

    chosen = branch_graph.clicked_branch(event, figure)
    if chosen and chosen != current:
        state.switch_branch(chosen)
        st.rerun()


def _lane_details(graph, branches, segments, facts) -> None:
    current = st.session_state.get("branch_id")

    for branch in branches:
        own = [s for s in segments if s.branch_id == branch.id]
        visible = graph.visible_segments(segments, branch.id)
        live = graph.active_facts(facts, branch.id)
        questions = [f for f in live if f.kind == "open_question"]
        selected = branch.id == current

        fork_note = (
            f"branches after part {branch.forked_at}"
            if branch.forked_at is not None
            else "the original"
        )

        columns = st.columns([5, 1, 1])
        with columns[0]:
            open_tag = (
                f"<span style='color:{theme.ACCENT};font-size:.78rem'> · open</span>"
                if selected
                else ""
            )
            st.markdown(
                f"<div class='anu-card'>"
                f"<div class='anu-meta'>{fork_note}</div>"
                f"<h4>{branch.name}{open_tag}</h4>"
                f"<p>{language.plural(len(visible), 'part')} to read · "
                f"{language.plural(len(own), 'part')} written here · "
                f"{language.plural(len(live), 'fact')} true · "
                f"{language.plural(len(questions), 'question')} open</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with columns[1]:
            st.write("")
            if not selected and st.button(
                "Open", key=f"tl_open_{branch.id}", width="stretch"
            ):
                state.switch_branch(branch.id)
                st.rerun()
        with columns[2]:
            st.write("")
            if not branch.is_root and st.button(
                "Delete", key=f"tl_del_{branch.id}", width="stretch"
            ):
                _delete(branch.id)


def _delete(branch_id: str) -> None:
    store = state.get_store()
    branch = store.get_branch(branch_id)
    try:
        store.delete_branch(branch_id)
    except ValueError as exc:
        state.fail(str(exc))
        st.rerun()
        return

    canon = state.get_canon_store()
    if canon is not None and branch is not None:
        try:
            canon.delete_branch(branch.project_id, branch_id)
        except Exception:
            pass

    if st.session_state.branch_id == branch_id:
        remaining = store.list_branches(branch.project_id)
        state.switch_branch(remaining[0].id if remaining else None)

    state.flash("Timeline deleted.")
    st.rerun()


def _compare(graph, branches, segments, facts) -> None:
    st.markdown("### Compare two timelines")
    st.caption(
        "What each one holds as true. These are the facts the story check will "
        "hold you to, so a difference here changes what counts as a plot hole."
    )

    names = {b.id: b.name for b in branches}
    picker = st.columns(2)
    with picker[0]:
        left_id = st.selectbox(
            "First", list(names), format_func=lambda b: names[b], key="cmp_left"
        )
    with picker[1]:
        default = 1 if len(names) > 1 else 0
        right_id = st.selectbox(
            "Second",
            list(names),
            format_func=lambda b: names[b],
            index=default,
            key="cmp_right",
        )

    if left_id == right_id:
        st.caption("Pick two different timelines.")
        return

    left = {f.id: f for f in graph.active_facts(facts, left_id)}
    right = {f.id: f for f in graph.active_facts(facts, right_id)}

    only_left = [f for fid, f in left.items() if fid not in right]
    only_right = [f for fid, f in right.items() if fid not in left]
    shared = len(set(left) & set(right))

    st.write("")
    summary = st.columns(3)
    summary[0].metric("Shared", shared)
    summary[1].metric(f"Only in {names[left_id]}", len(only_left))
    summary[2].metric(f"Only in {names[right_id]}", len(only_right))

    if not only_left and not only_right:
        st.info("These timelines hold exactly the same facts as true.")
        return

    columns = st.columns(2)
    for column, name, group in (
        (columns[0], names[left_id], only_left),
        (columns[1], names[right_id], only_right),
    ):
        with column:
            st.markdown(f"**True only on {name}**")
            if not group:
                st.caption("Nothing unique.")
                continue
            for fact in group:
                st.markdown(
                    f"<div style='font-size:.86rem;margin:.25rem 0'>"
                    f"<strong>{_escape(fact.subject)}</strong> — {_escape(fact.claim)}"
                    f"<span style='color:{theme.FAINT}'> · part "
                    f"{fact.established_position + 1}</span></div>",
                    unsafe_allow_html=True,
                )


def _escape(text: str) -> str:
    return (text or "").replace("<", "&lt;").replace(">", "&gt;")
