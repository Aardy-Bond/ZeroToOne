"""
Settings: who you are writing for, and what is connected.

The target reader lives here rather than in the sidebar because it is a
decision a writer makes once per story, not per draft.
"""

from __future__ import annotations

import os

import streamlit as st

from dashboard import brand, language, state
from dashboard.forecast_view import AGE_BANDS, GENRES, get_quality_proxy
from retention_engine.schemas import CohortProfile


def _cohort_controls() -> CohortProfile:
    """
    The same fields as the original sidebar control, in the main area.

    Kept here rather than added to `forecast_view` so that module's existing
    sidebar entry point stays exactly as it was.
    """
    saved = st.session_state.get("cohort")

    def index(options: list, value, fallback: int = 0) -> int:
        return options.index(value) if value in options else fallback

    columns = st.columns(2)
    with columns[0]:
        genre = st.selectbox(
            "Kind of story", GENRES, index=index(GENRES, getattr(saved, "genre_affinity", None))
        )
        pace = st.select_slider(
            "Pace they enjoy",
            options=["slow", "balanced", "fast"],
            value=getattr(saved, "pace_preference", "balanced"),
        )
        complexity = st.select_slider(
            "How much they will follow at once",
            options=["low", "medium", "high"],
            value=getattr(saved, "complexity_tolerance", "medium"),
        )
        age_band = st.selectbox(
            "Age band",
            AGE_BANDS,
            index=index(AGE_BANDS, getattr(saved, "age_band", None)),
            help="Context you are choosing, not a measured statistic.",
        )
    with columns[1]:
        emotional = st.selectbox(
            "What they come for",
            ["dread", "mystery", "action", "romance", "warmth"],
            index=index(
                ["dread", "mystery", "action", "romance", "warmth"],
                getattr(saved, "emotional_preference", None),
            ),
        )
        content = st.select_slider(
            "How dark they will go",
            options=["low", "medium", "high"],
            value=getattr(saved, "content_boundary", "medium"),
        )
        mode = st.selectbox(
            "How they listen",
            ["commute", "binge", "casual"],
            index=index(
                ["commute", "binge", "casual"], getattr(saved, "listening_mode", None)
            ),
        )

    proxy = get_quality_proxy()
    if proxy.available:
        meta = proxy.metadata
        st.caption(
            f"Prose is scored against {meta.get('sample_count', '?')} rated stories "
            f"({proxy.model_version}). A writing-quality reference, not a "
            "retention measurement."
        )
    else:
        st.caption(
            "Prose scoring is off. Run `python scripts/train_quality_proxy.py` "
            "to switch it on."
        )

    return CohortProfile(
        label="Primary target cohort",
        genre_affinity=genre,
        pace_preference=pace,
        complexity_tolerance=complexity,
        emotional_preference=emotional,
        content_boundary=content,
        listening_mode=mode,
        age_band=age_band,
        is_counterfactual=False,
    )


def render() -> None:
    state.drain_messages()
    brand.page_hero(
        "Settings",
        "Who you are writing for, how thorough the analysis should be, and "
        "what this studio is connected to.",
        eyebrow="Pocket FM · Studio",
    )

    brand.section_label("Audience")
    st.markdown("### Who you are writing for")
    st.caption(
        "Anubhuti reads the story as this reader would. It is your description "
        "of an intended audience, not a measurement of a real one."
    )
    st.session_state.cohort = _cohort_controls()

    st.divider()

    brand.section_label("Analysis")
    st.markdown("### How thorough")
    st.session_state.deep_dive = st.toggle(
        "Explain the weak spots in detail",
        value=st.session_state.get("deep_dive", True),
        help=(
            "Adds a written explanation for up to three risky scenes. The "
            "structural evidence is shown either way."
        ),
    )

    st.divider()

    brand.section_label("Connections")
    st.markdown("### Studio status")
    brand.status_row(list(state.connection_status()))

    if state.get_canon_store() is None and os.getenv("DATABRICKS_HOST", "").strip():
        st.caption(
            "Searchable canon is unavailable, so passage recall is off. The fact "
            "ledger is local, so story checks still run."
        )

    with st.expander("Where your work is stored"):
        st.markdown(
            f"- Stories, timelines, and the fact ledger: `{state.DB_PATH}`\n"
            f"- Searchable passages: Databricks Vector Search, when connected\n"
            f"- Audio and manifests: `output/`"
        )
        st.caption(
            "Everything except the searchable passages is on this machine. "
            "Delete the database file to start over."
        )

    st.divider()
    st.caption(language.DISCLAIMER)
