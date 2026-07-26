"""
Production: the expert panel, the audience read, and voiced audio.

This is the original Writers Room pipeline, unchanged underneath and re-homed
as a page. It now reads the draft from the workspace instead of its own editor,
so a writer moves between writing and production without copying text around.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from audience_simulator.simulator import (
    DROP_OFF_THRESHOLD,
    SCORE_MAX,
    AudienceSimulator,
    AudienceSimulatorError,
    _persona_name,
)
from audio_engine.synthesizer import (
    AudioSynthesizer,
    SynthesisError,
    format_timestamp,
)
from dashboard import brand, forecast_view, language, producer_view, state, theme
from writers_room.genre_rewrite import (
    GENRE_REWRITE_TARGETS,
    GenreRewriteError,
    rewrite_as_genre,
)
from writers_room.orchestrator import analyze_script, fetch_canon, format_canon_warnings
from writers_room.rewrite_engine import (
    RewriteEngineError,
    rewrite_weak_segments,
    select_weak_minutes,
)
from writers_room.schemas import FoleyTrigger

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULTS = {
    "critique": None,
    "report": None,
    "canon": [],
    "weak_minutes": [],
    "rewrite": None,
    "genre_rewrite": None,
    "manifest": None,
}


def render() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    state.drain_messages()

    brand.page_hero(
        "Production",
        "Let the AI Producer cast and direct from your retention read-through, "
        "then convene the specialist panel and render audio.",
        eyebrow="Pocket FM · Production",
    )

    script = st.session_state.get("draft", "").strip()
    if not script:
        st.info(
            "Nothing to work on yet. Write a part on the Write page, then come "
            "back here."
        )
        _sample_offer()
        return

    st.caption(language.word_count_line(script))

    settings = _settings()

    if st.session_state.pop("open_producer", False) and not st.session_state.get(
        "producer_plan"
    ):
        _run_producer(script, settings)
        st.rerun()

    producer_primary = st.session_state.get("readthrough") is not None
    actions = st.columns([1.15, 1, 1, 1])
    with actions[0]:
        if st.button(
            "Run AI Producer",
            type="primary" if producer_primary else "secondary",
            width="stretch",
            help=(
                "Uses the retention read-through (survival, EKG signals, "
                "cliffhanger) to cast voices and direct delivery. Runs a "
                "read-through first if you have not yet."
            ),
        ):
            _run_producer(script, settings)
            st.rerun()
    with actions[1]:
        if st.button(
            "Run the panel",
            type="primary" if not producer_primary else "secondary",
            width="stretch",
        ):
            _analyse(script, settings)
            st.rerun()
    with actions[2]:
        if st.button(
            "Rewrite the weak minutes",
            width="stretch",
            disabled=st.session_state.report is None or not st.session_state.weak_minutes,
        ):
            _rewrite(script, settings)
            st.rerun()
    with actions[3]:
        if st.button(
            "Render audio",
            width="stretch",
            disabled=st.session_state.critique is None,
        ):
            _synthesise(script, settings)
            st.rerun()

    st.write("")
    st.markdown("#### Genre rewrite")
    st.caption(
        "Restyle the whole draft into another genre or style while keeping "
        "the core plot and project canon. Separate from rewriting weak minutes."
    )
    genre_cols = st.columns([1.4, 1.2, 2.4])
    with genre_cols[0]:
        genre_target = st.selectbox(
            "Target genre or style",
            options=list(GENRE_REWRITE_TARGETS),
            format_func=lambda g: g.replace("_", " ").title(),
            key="production_genre_rewrite_target",
        )
    with genre_cols[1]:
        st.write("")
        if st.button("Rewrite as genre", width="stretch"):
            _genre_rewrite(script, genre_target)
            st.rerun()

    if st.session_state.get("producer_error"):
        st.error(st.session_state.producer_error)
        st.session_state.producer_error = ""

    if st.session_state.get("producer_plan") is not None:
        st.divider()
        producer_view.render_producer(
            st.session_state.producer_plan,
            st.session_state.get("producer_lines") or [],
            forecast=st.session_state.get("readthrough"),
            manifest=st.session_state.get("producer_manifest"),
        )
        hear = st.columns([1, 1, 2])
        with hear[0]:
            if st.button(
                "Hear this plan",
                width="stretch",
                type="primary",
                disabled=st.session_state.producer_plan is None,
            ):
                _hear_producer_plan(script, settings)
                st.rerun()
        with hear[1]:
            if st.session_state.get("producer_plan"):
                st.download_button(
                    "Download plan JSON",
                    producer_view.plan_as_download_bytes(st.session_state.producer_plan),
                    file_name="producer_plan.json",
                    mime="application/json",
                    width="stretch",
                )

    if st.session_state.report is not None:
        st.divider()
        _heatmap(st.session_state.report, settings["threshold"], st.session_state.weak_minutes)

    if st.session_state.critique is not None:
        st.divider()
        st.markdown("### The panel")
        _panel(st.session_state.critique, st.session_state.canon)

    if st.session_state.rewrite is not None:
        st.divider()
        _rewrite_result(st.session_state.rewrite)

    if st.session_state.get("genre_rewrite") is not None:
        st.divider()
        _genre_rewrite_result(st.session_state.genre_rewrite)

    if st.session_state.manifest is not None:
        st.divider()
        st.markdown("### Audio")
        _audio(st.session_state.manifest)

    if st.session_state.get("preview_manifest"):
        st.divider()
        forecast_view.render_preview(st.session_state.preview_manifest)


def _run_producer(script: str, settings: dict) -> None:
    """Forecast if needed, then ask the AI Producer for a full plan."""
    from producer.agent import run_producer

    st.session_state.producer_error = ""
    forecast = st.session_state.get("readthrough")
    if forecast is None:
        from dashboard import readthrough

        with st.spinner("Reading it through so the producer has scores to work from..."):
            readthrough.run(script, is_revision=False)
        forecast = st.session_state.get("readthrough")
        if forecast is None:
            st.session_state.producer_error = (
                "The read-through did not finish, so the producer could not run."
            )
            return

    with st.spinner("AI Producer is casting and marking the script..."):
        try:
            plan, lines = run_producer(script, forecast)
        except (ValueError, RuntimeError) as exc:
            st.session_state.producer_error = str(exc)
            return

    st.session_state.producer_plan = plan
    st.session_state.producer_lines = lines
    st.session_state.producer_manifest = None
    state.flash("AI Producer plan ready — review the annotated script below.")


def _hear_producer_plan(script: str, settings: dict) -> None:
    from producer.script_align import chunks_from_lines
    from producer.to_directing_sheet import casting_overrides, plan_to_directing_sheet

    plan = st.session_state.get("producer_plan")
    lines = st.session_state.get("producer_lines") or []
    forecast = st.session_state.get("readthrough")
    if plan is None or not lines or forecast is None:
        st.session_state.producer_error = "Run the AI Producer before hearing the plan."
        return

    sheet = plan_to_directing_sheet(plan, lines, forecast)
    chunks = chunks_from_lines(lines)
    triggers = _producer_foley(plan)

    label = (
        "Rendering directed audio from the producer plan..."
        if settings["generate_audio"]
        else "Building the producer manifest..."
    )
    with st.spinner(label):
        try:
            manifest = AudioSynthesizer(
                output_dir=PROJECT_ROOT / "output",
            ).synthesize_directed_script(
                script,
                triggers,
                sheet,
                generate_audio=settings["generate_audio"],
                chunks=chunks,
                voice_overrides=casting_overrides(plan),
                forecast_summary={
                    "primary_cohort_survival_proxy": forecast.overall_survival,
                    "unlock_pull_index": forecast.cliffhanger.unlock_pull_index,
                    "strategy": plan.strategy,
                    "source": "ai_producer",
                },
            )
        except (SynthesisError, ValueError) as exc:
            st.session_state.producer_error = f"Directed render failed: {exc}"
            return

    st.session_state.producer_manifest = manifest
    st.session_state.preview_manifest = manifest
    state.flash("Directed audio from the AI Producer is ready.")


def _producer_foley(plan) -> list[FoleyTrigger]:
    """Sound board → FoleyTrigger list (beat refs when no clock time)."""
    triggers: list[FoleyTrigger] = []
    critique = st.session_state.get("critique")
    if critique is not None:
        triggers.extend(critique.foley_triggers)
    for cue in plan.sound_cues:
        triggers.append(
            FoleyTrigger(
                timestamp=f"on line {cue.line_index}",
                sound_effect=cue.effect,
            )
        )
    return triggers


def _sample_offer() -> None:
    sample = PROJECT_ROOT / "samples" / "wexler_street_continuation.txt"
    if not sample.exists():
        return
    if st.button("Load the sample script into the composer"):
        st.session_state.draft = sample.read_text(encoding="utf-8")
        st.session_state.draft_version += 1
        state.flash("Sample loaded into the composer.")
        st.rerun()


def _settings() -> dict:
    with st.expander("Settings"):
        columns = st.columns(2)
        with columns[0]:
            use_lore = st.toggle(
                "Check against the older lore database",
                value=False,
                help=(
                    "The original Databricks lore table, kept for the earlier "
                    "pipeline. Project canon on the Write page is separate."
                ),
            )
            character = st.text_input("Bias canon toward character", value="")
            generate_audio = st.toggle("Actually render the audio", value=True)
        with columns[1]:
            threshold = st.slider(
                "Drop-off threshold",
                1.0,
                9.0,
                float(DROP_OFF_THRESHOLD),
                0.5,
            )
            min_dropped = st.slider("Listeners needed to flag a minute", 1, 3, 2)

    return {
        "use_lore": use_lore,
        "character_id": character.strip() or None,
        "threshold": threshold,
        "min_dropped": min_dropped,
        "generate_audio": generate_audio,
    }


def _analyse(script: str, settings: dict) -> None:
    st.session_state.update({k: v for k, v in DEFAULTS.items()})

    if settings["use_lore"]:
        with st.spinner("Retrieving established canon..."):
            try:
                st.session_state.canon = fetch_canon(
                    script, character_id=settings["character_id"]
                )
            except Exception as exc:
                st.warning(f"Canon retrieval skipped: {exc}")

    with st.spinner("Convening the panel..."):
        try:
            st.session_state.critique = analyze_script(
                script,
                character_id=settings["character_id"],
                use_lore=settings["use_lore"],
            )
        except Exception as exc:
            state.fail(f"The panel could not run: {exc}")
            return

    with st.spinner("Playing it to a synthetic audience..."):
        try:
            st.session_state.report = AudienceSimulator().simulate_audience(script)
        except (AudienceSimulatorError, ValueError) as exc:
            state.fail(f"Audience simulation failed: {exc}")
            return

    st.session_state.weak_minutes = [
        entry.minute
        for entry in select_weak_minutes(
            st.session_state.report.heatmap,
            threshold=settings["threshold"],
            min_dropped_personas=settings["min_dropped"],
        )
    ]


def _rewrite(script: str, settings: dict) -> None:
    with st.spinner("Rewriting the flagged minutes..."):
        try:
            result = rewrite_weak_segments(
                script,
                st.session_state.report.heatmap,
                st.session_state.critique,
                canon=st.session_state.canon,
                threshold=settings["threshold"],
                min_dropped_personas=settings["min_dropped"],
            )
        except (RewriteEngineError, ValueError) as exc:
            state.fail(f"Rewrite failed: {exc}")
            return

    if result is None:
        state.fail("No minute was flagged, so nothing was rewritten.")
        return

    st.session_state.rewrite = result
    st.session_state.draft = result.rewritten_script
    st.session_state.draft_version += 1
    # The old read describes the previous draft.
    st.session_state.report = None
    st.session_state.weak_minutes = []
    st.session_state.manifest = None
    state.flash("The draft in the composer has been replaced with the rewrite.")


def _genre_rewrite(script: str, target: str) -> None:
    """Restyle the draft using project canon when a project is open."""
    context_parts: list[str] = []
    project = state.current_project()
    branch = state.current_branch()
    service = None
    if project is not None and branch is not None:
        service = state.get_service()
        try:
            pack = service.context_for(project.id, branch.id, draft=script)
            text = pack.to_prompt() if pack is not None else ""
            if text.strip():
                context_parts.append(text.strip())
        except Exception as exc:
            st.warning(f"Project canon skipped: {exc}")

    lore = st.session_state.get("canon") or []
    if lore:
        context_parts.append(format_canon_warnings(lore))

    context_text = "\n\n".join(context_parts)

    with st.spinner(f"Rewriting as {target}..."):
        try:
            result = rewrite_as_genre(
                script,
                target,
                context_pack=context_text,
            )
        except (GenreRewriteError, ValueError) as exc:
            state.fail(f"Genre rewrite failed: {exc}")
            return

    st.session_state.genre_rewrite = result
    st.session_state.draft = result.rewritten_script
    st.session_state.draft_version += 1
    st.session_state.report = None
    st.session_state.weak_minutes = []
    st.session_state.manifest = None
    st.session_state.critique = None
    st.session_state.readthrough = None
    st.session_state.readthrough_baseline = None
    st.session_state.comparison = None

    if service is not None and project is not None and branch is not None:
        start = st.session_state.get("start_from")
        position = (
            start["position"] + 1
            if start is not None
            else None
        )
        with st.spinner("Checking the rewrite against the story..."):
            try:
                st.session_state.continuity = service.check_draft(
                    project.id,
                    branch.id,
                    result.rewritten_script,
                    position=position,
                )
            except Exception as exc:
                st.warning(f"Story check after genre rewrite skipped: {exc}")

    state.flash(
        f"Draft restyled as {target}. Open Write to see the story check."
    )


def _synthesise(script: str, settings: dict) -> None:
    label = "Rendering audio..." if settings["generate_audio"] else "Building the manifest..."
    with st.spinner(label):
        try:
            st.session_state.manifest = AudioSynthesizer(
                output_dir=PROJECT_ROOT / "output"
            ).synthesize(
                script,
                st.session_state.critique.foley_triggers,
                generate_audio=settings["generate_audio"],
            )
        except (SynthesisError, ValueError) as exc:
            state.fail(f"Synthesis failed: {exc}")


def _heatmap(report, threshold: float, weak_minutes: list[int]) -> None:
    st.markdown("### Where the audience drifts")

    minutes = [e.minute for e in report.heatmap]
    scores = [e.average_score for e in report.heatmap]

    top = st.columns(3)
    top[0].metric("Overall engagement", f"{report.overall_average:.2f} / {SCORE_MAX}")
    top[1].metric("Minutes analysed", len(report.heatmap))
    top[2].metric("Flagged minutes", len(weak_minutes))

    if HAS_PLOTLY:
        colours = [theme.RISK if m in weak_minutes else theme.CALM for m in minutes]
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=minutes,
                y=scores,
                mode="lines+markers",
                line={"color": theme.ACCENT, "width": 3},
                marker={"size": 13, "color": colours},
                hovertemplate="Minute %{x}<br>Score %{y:.2f}<extra></extra>",
            )
        )
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=theme.RISK,
            annotation_text=f"drop-off threshold ({threshold})",
            annotation_position="bottom right",
        )
        figure.update_layout(
            height=330,
            margin={"l": 10, "r": 10, "t": 26, "b": 10},
            yaxis={"range": [0, SCORE_MAX + 0.5], "title": "Engagement"},
            xaxis={"title": "Minute", "dtick": 1},
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, width="stretch")
    else:
        st.line_chart({"engagement": scores})

    if weak_minutes:
        st.warning(
            "Flagged minutes: " + ", ".join(str(m) for m in sorted(weak_minutes))
        )

    with st.expander("Per-listener scores and reasons"):
        for entry in report.heatmap:
            flag = "  ·  flagged" if entry.minute in weak_minutes else ""
            line = " · ".join(
                f"{_persona_name(k)}: {v}" for k, v in entry.persona_scores.items()
            )
            st.markdown(f"**Minute {entry.minute}** — avg {entry.average_score:.2f}{flag}  \n{line}")
            for persona, reason in entry.drop_off_reasons.items():
                st.caption(f"{_persona_name(persona)}: {reason}")

    with st.expander("What each listener said overall"):
        for key, verdict in report.verdicts.items():
            status = "would finish" if verdict.would_finish else "drops off"
            st.markdown(f"**{_persona_name(key)}** — {status}")
            st.write(verdict.overall_summary)


def _panel(critique, canon: list[dict]) -> None:
    tabs = st.tabs(["Director", "Editor", "Psychologist", "Sound", "Continuity"])

    with tabs[0]:
        st.write(critique.director_notes)
    with tabs[1]:
        st.write(critique.editor_notes)
    with tabs[2]:
        st.write(critique.psychologist_notes)
    with tabs[3]:
        if critique.foley_triggers:
            st.dataframe(
                [
                    {"Timestamp": t.timestamp, "Sound effect": t.sound_effect}
                    for t in critique.foley_triggers
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("The panel returned no sound cues for this scene.")
    with tabs[4]:
        st.write(critique.continuity_critique)
        if canon:
            st.divider()
            st.caption("Canon supplied to the panel")
            st.code(format_canon_warnings(canon), language="text")

    with st.expander("Raw panel output"):
        st.json(json.loads(critique.model_dump_json()))


def _rewrite_result(result) -> None:
    st.markdown("### The rewrite")
    if getattr(result, "change_log", None):
        minutes = [
            str(entry.minute)
            for entry in result.change_log
            if getattr(entry, "minute", None) is not None
        ]
        if minutes:
            st.caption("Minutes rewritten: " + ", ".join(minutes))
        for entry in result.change_log:
            st.markdown(
                f"- **Minute {entry.minute}** — {entry.change_made} "
                f"({entry.problem})"
            )
    note = getattr(result, "tone_continuity_note", None) or getattr(
        result, "summary", ""
    )
    if note:
        st.write(note)


def _genre_rewrite_result(result) -> None:
    label = (result.target_genre or "").replace("_", " ").title()
    st.markdown(f"### Genre rewrite · {label}")
    st.write(result.plot_preservation_note)
    if result.change_log:
        st.markdown("**What changed**")
        for entry in result.change_log:
            st.markdown(f"- **{entry.aspect}** — {entry.change_made}")


def _audio(manifest: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("Chunks", manifest["chunk_count"])
    columns[1].metric("Runtime", manifest["total_runtime"])
    columns[2].metric("Sound cues", manifest["foley_cue_count"])
    columns[3].metric(
        "Audio", "rendered" if manifest["audio_generated"] else "manifest only"
    )

    audio_dir = Path(manifest["audio_dir"])
    for chunk in manifest["chunks"]:
        st.markdown(
            f"**[{chunk['start']}] {chunk['character']}** "
            f"({chunk['voice']}, {chunk['duration_seconds']:.1f}s)"
        )
        st.caption(chunk["text"])

        if manifest["audio_generated"] and chunk["audio_file"]:
            path = audio_dir / chunk["audio_file"]
            if path.exists():
                st.audio(str(path))

        for cue in chunk["foley"]:
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;&nbsp;`[{format_timestamp(cue['at_seconds'])} "
                f"- FX: {cue['sound_effect']}]`",
                unsafe_allow_html=True,
            )
        st.divider()

    if manifest.get("unassigned_foley"):
        st.warning("Cues that could not be pinned to a spoken line")
        st.dataframe(
            [
                {"Timestamp": c["timestamp"], "Sound effect": c["sound_effect"]}
                for c in manifest["unassigned_foley"]
            ],
            width="stretch",
            hide_index=True,
        )
