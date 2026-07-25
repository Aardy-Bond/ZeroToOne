"""
Project Anubhuti — Interactive Creator Dashboard.

Streamlit front end wiring together the Lore Engine, Writers Room, Audience
Simulator, Rewrite Engine, and Audio Synthesizer.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
SRC_DIR = DASHBOARD_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from audience_simulator.simulator import (  # noqa: E402
    DROP_OFF_THRESHOLD,
    SCORE_MAX,
    AudienceSimulator,
    AudienceSimulatorError,
    _persona_name,
)
from audio_engine.synthesizer import (  # noqa: E402
    AudioSynthesizer,
    SynthesisError,
    format_timestamp,
)
from writers_room.orchestrator import (  # noqa: E402
    analyze_script,
    fetch_canon,
    format_canon_warnings,
)
from writers_room.rewrite_engine import (  # noqa: E402
    RewriteEngineError,
    rewrite_weak_segments,
    select_weak_minutes,
)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover - chart degrades to st.line_chart
    HAS_PLOTLY = False


DEFAULT_SCRIPT = """\
INT. BASEMENT LAUNDRY ROOM - NIGHT

A single bulb swings. PRIYA stands at the dryer, folding a child's t-shirt.
The machine has been off for an hour. The clothes are still warm.

PRIYA
Okay. Okay, that's just... that's the pipes.

She folds another shirt. She does not turn around.

NARRATOR (V.O.)
The house on Wexler Street was built in 1911 by a shipping clerk named Aldous
Renn. The deed changed hands eleven times before the Kapoors bought it. The
consortium that held it in the twenties dissolved in 1931 following a dispute
over the eastern easement, which had been surveyed incorrectly in 1908 and
would be resurveyed twice more before the matter was settled in county court.

From the far corner, a slow DRAG of something heavy across concrete.

PRIYA (CONT'D)
Arjun? Sweetheart, if that's you, this isn't funny right now.

The dragging stops. Directly behind her.

PRIYA (CONT'D)
I'm going to count to three, and then I'm going to turn around, and you are
going to be my son. Okay? One.

The bulb dies.

PRIYA (CONT'D)
Two.

Something breathes. It is not a child.

CUT TO:

INT. KITCHEN - CONTINUOUS

DEV scrolls his phone at the counter. The basement door is open behind him.

DEV
Priya? You want tea?

No answer. He sets the phone down.

DEV (CONT'D)
This isn't funny either. Both of you. This whole family.

He takes the first step. The wood gives under him, wet.

ARJUN (O.S.)
Dad? Mum's down here. She wants you to come see.

Dev stops. His son is asleep upstairs. He put him there ninety minutes ago.

SMASH TO BLACK.
"""

SESSION_DEFAULTS = {
    "script_text": DEFAULT_SCRIPT,
    "editor_version": 0,
    "critique": None,
    "report": None,
    "canon": [],
    "weak_minutes": [],
    "rewrite": None,
    "manifest": None,
    "last_error": "",
}


def init_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_downstream() -> None:
    """Clear results that no longer describe the current script."""
    st.session_state.critique = None
    st.session_state.report = None
    st.session_state.canon = []
    st.session_state.weak_minutes = []
    st.session_state.rewrite = None
    st.session_state.manifest = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def render_sidebar() -> dict:
    st.sidebar.title("Project Anubhuti")
    st.sidebar.caption("AI Writers Room control panel")

    st.sidebar.subheader("Connections")

    if os.getenv("OPENAI_API_KEY", "").strip():
        st.sidebar.success("OpenAI — connected")
    else:
        st.sidebar.error("OpenAI — OPENAI_API_KEY not set")

    host = os.getenv("DATABRICKS_HOST", "").strip()
    if host:
        st.sidebar.success(f"Databricks — {host.replace('https://', '')}")
    else:
        st.sidebar.warning("Databricks — DATABRICKS_HOST not set")

    warehouse = os.getenv("DATABRICKS_WAREHOUSE_ID", "").strip()
    if warehouse:
        st.sidebar.success(f"SQL Warehouse — {warehouse}")
    else:
        st.sidebar.info("SQL Warehouse — auto-discover")

    st.sidebar.divider()
    st.sidebar.subheader("Analysis settings")

    use_lore = st.sidebar.toggle(
        "Check lore continuity",
        value=bool(host),
        help="Retrieves canon from Databricks Vector Search. Requires a running SQL warehouse.",
    )
    character_id = st.sidebar.text_input(
        "Bias canon toward character",
        value="arjun",
        help="Optional. Leave blank to search all characters.",
    )
    threshold = st.sidebar.slider(
        "Drop-off threshold",
        min_value=1.0,
        max_value=9.0,
        value=float(DROP_OFF_THRESHOLD),
        step=0.5,
    )
    min_dropped = st.sidebar.slider(
        "Personas needed to flag a minute",
        min_value=1,
        max_value=3,
        value=2,
        help="Flags a minute even when the average stays above threshold.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Audio")
    generate_audio = st.sidebar.toggle(
        "Render TTS audio",
        value=True,
        help="Turn off to build the manifest and cue sheet without spending TTS credits.",
    )

    return {
        "use_lore": use_lore,
        "character_id": character_id.strip() or None,
        "threshold": threshold,
        "min_dropped": min_dropped,
        "generate_audio": generate_audio,
    }


# --------------------------------------------------------------------------
# Pipeline actions
# --------------------------------------------------------------------------


def run_full_analysis(script_text: str, settings: dict) -> None:
    st.session_state.last_error = ""
    reset_downstream()

    if settings["use_lore"]:
        with st.spinner("Retrieving established canon from Databricks..."):
            st.session_state.canon = fetch_canon(
                script_text, character_id=settings["character_id"]
            )

    with st.spinner("Convening the expert panel (gpt-4o)..."):
        try:
            st.session_state.critique = analyze_script(
                script_text,
                character_id=settings["character_id"],
                use_lore=settings["use_lore"],
            )
        except Exception as exc:
            st.session_state.last_error = f"Writers Room failed: {exc}"
            return

    with st.spinner("Running synthetic audience (3 personas in parallel)..."):
        try:
            st.session_state.report = AudienceSimulator().simulate_audience(script_text)
        except (AudienceSimulatorError, ValueError) as exc:
            st.session_state.last_error = f"Audience simulation failed: {exc}"
            return

    st.session_state.weak_minutes = [
        entry.minute
        for entry in select_weak_minutes(
            st.session_state.report.heatmap,
            threshold=settings["threshold"],
            min_dropped_personas=settings["min_dropped"],
        )
    ]


def run_rewrite(script_text: str, settings: dict) -> None:
    st.session_state.last_error = ""

    with st.spinner("Rewriting the flagged minutes..."):
        try:
            result = rewrite_weak_segments(
                script_text,
                st.session_state.report.heatmap,
                st.session_state.critique,
                canon=st.session_state.canon,
                threshold=settings["threshold"],
                min_dropped_personas=settings["min_dropped"],
            )
        except (RewriteEngineError, ValueError) as exc:
            st.session_state.last_error = f"Rewrite failed: {exc}"
            return

    if result is None:
        st.session_state.last_error = "No minute was flagged, so nothing was rewritten."
        return

    st.session_state.rewrite = result
    st.session_state.script_text = result.rewritten_script
    # Force a fresh text_area so the editor picks up the rewritten script.
    st.session_state.editor_version += 1
    # The old critique and heatmap describe the previous draft.
    st.session_state.report = None
    st.session_state.weak_minutes = []
    st.session_state.manifest = None


def run_synthesis(script_text: str, settings: dict) -> None:
    st.session_state.last_error = ""

    label = "Rendering scratch audio..." if settings["generate_audio"] else "Building manifest..."
    with st.spinner(label):
        try:
            st.session_state.manifest = AudioSynthesizer(
                output_dir=PROJECT_ROOT / "output"
            ).synthesize(
                script_text,
                st.session_state.critique.foley_triggers,
                generate_audio=settings["generate_audio"],
            )
        except (SynthesisError, ValueError) as exc:
            st.session_state.last_error = f"Synthesis failed: {exc}"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_heatmap(report, threshold: float, weak_minutes: list[int]) -> None:
    minutes = [e.minute for e in report.heatmap]
    scores = [e.average_score for e in report.heatmap]

    top = st.columns(3)
    top[0].metric("Overall engagement", f"{report.overall_average:.2f} / {SCORE_MAX}")
    top[1].metric("Minutes analysed", len(report.heatmap))
    top[2].metric("Flagged minutes", len(weak_minutes))

    if HAS_PLOTLY:
        colors = ["#e03131" if m in weak_minutes else "#2f9e44" for m in minutes]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=minutes,
                y=scores,
                mode="lines+markers",
                name="Average engagement",
                line={"color": "#4c6ef5", "width": 3},
                marker={"size": 14, "color": colors},
                hovertemplate="Minute %{x}<br>Score %{y:.2f}<extra></extra>",
            )
        )
        figure.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#e03131",
            annotation_text=f"drop-off threshold ({threshold})",
            annotation_position="bottom right",
        )
        figure.update_layout(
            height=340,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis={"range": [0, SCORE_MAX + 0.5], "title": "Engagement"},
            xaxis={"title": "Minute", "dtick": 1},
            showlegend=False,
        )
        st.plotly_chart(figure, width="stretch")
    else:
        st.line_chart({"engagement": scores})

    if weak_minutes:
        st.warning(f"Flagged minutes: {', '.join(str(m) for m in sorted(weak_minutes))}")

    with st.expander("Per-persona scores and drop-off reasons"):
        for entry in report.heatmap:
            flag = "  🔴" if entry.minute in weak_minutes else ""
            scores_text = " · ".join(
                f"{_persona_name(k)}: {v}" for k, v in entry.persona_scores.items()
            )
            st.markdown(
                f"**Minute {entry.minute}** — avg {entry.average_score:.2f}{flag}  \n{scores_text}"
            )
            for persona, reason in entry.drop_off_reasons.items():
                st.caption(f"↳ {_persona_name(persona)}: {reason}")

    with st.expander("Persona verdicts"):
        for key, verdict in report.verdicts.items():
            status = "would finish" if verdict.would_finish else "DROPS OFF"
            st.markdown(f"**{_persona_name(key)}** — {status}")
            st.write(verdict.overall_summary)


def render_panel_tabs(critique, canon: list[dict]) -> None:
    tabs = st.tabs(
        ["Director", "Editor", "Psychologist", "Sound Producer", "Continuity (Lore)"]
    )

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
            st.info("The panel returned no foley cues for this scene.")

    with tabs[4]:
        st.write(critique.continuity_critique)
        st.divider()
        st.caption("Canon supplied to the panel")
        st.code(format_canon_warnings(canon), language="text")

    with st.expander("Raw SceneCritique JSON"):
        st.json(json.loads(critique.model_dump_json()))


def render_audio(manifest: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Chunks", manifest["chunk_count"])
    cols[1].metric("Runtime", manifest["total_runtime"])
    cols[2].metric("Foley cues", manifest["foley_cue_count"])
    cols[3].metric("Audio", "rendered" if manifest["audio_generated"] else "manifest only")

    st.caption(f"Casting: {json.dumps(manifest['casting'])}")

    audio_dir = Path(manifest["audio_dir"])
    for chunk in manifest["chunks"]:
        header = (
            f"**[{chunk['start']}] {chunk['character']}** "
            f"({chunk['voice']}, {chunk['duration_seconds']:.1f}s)"
        )
        st.markdown(header)
        st.caption(chunk["text"])

        if manifest["audio_generated"] and chunk["audio_file"]:
            path = audio_dir / chunk["audio_file"]
            if path.exists():
                st.audio(str(path))
            else:
                st.caption(f"Missing audio file: {path}")

        for cue in chunk["foley"]:
            st.markdown(
                f"&nbsp;&nbsp;&nbsp;&nbsp;`[{format_timestamp(cue['at_seconds'])} "
                f"- FX: {cue['sound_effect']}]`",
                unsafe_allow_html=True,
            )
        st.divider()

    if manifest["unassigned_foley"]:
        st.warning("Cues that could not be pinned to a spoken line")
        st.dataframe(
            [
                {
                    "Timestamp": cue["timestamp"],
                    "Sound effect": cue["sound_effect"],
                    "Reason": cue["reason"],
                }
                for cue in manifest["unassigned_foley"]
            ],
            width="stretch",
            hide_index=True,
        )

    with st.expander("production_manifest.json"):
        st.json(manifest)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Project Anubhuti: AI Writers Room",
        page_icon="🎬",
        layout="wide",
    )
    init_state()

    st.title("Project Anubhuti: AI Writers Room")
    st.caption(
        "Lore continuity · expert critique · synthetic audience · auto-rewrite · audio export"
    )

    settings = render_sidebar()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    left, right = st.columns([2, 3], gap="large")

    # ---- Left: script editor ---------------------------------------------
    with left:
        st.subheader("Script editor")

        edited = st.text_area(
            "Draft script",
            value=st.session_state.script_text,
            height=560,
            key=f"editor_{st.session_state.editor_version}",
            label_visibility="collapsed",
        )
        st.session_state.script_text = edited

        st.caption(f"{len(edited.split())} words · ~{len(edited.split()) / 150:.1f} min runtime")

        if st.button("Run Full Analysis", type="primary", width="stretch"):
            if not edited.strip():
                st.session_state.last_error = "Paste a script before running analysis."
            else:
                run_full_analysis(edited, settings)
            st.rerun()

        if st.session_state.rewrite is not None:
            st.success("Script updated by the rewrite engine.")
            with st.expander("Rewrite change log", expanded=True):
                for change in st.session_state.rewrite.change_log:
                    st.markdown(f"**Minute {change.minute}**")
                    st.caption(f"Problem: {change.problem}")
                    st.caption(f"Change: {change.change_made}")
                st.info(st.session_state.rewrite.tone_continuity_note)

    # ---- Right: results ---------------------------------------------------
    with right:
        if st.session_state.report is None and st.session_state.critique is None:
            st.info(
                "Paste a script on the left and press **Run Full Analysis** to pull "
                "canon, convene the expert panel, and simulate the audience."
            )
            return

        st.subheader("Engagement heatmap")
        if st.session_state.report is not None:
            render_heatmap(
                st.session_state.report,
                settings["threshold"],
                st.session_state.weak_minutes,
            )
        else:
            st.info("Re-run the analysis to score the rewritten script.")

        st.divider()
        st.subheader("Expert panel")
        if st.session_state.critique is not None:
            render_panel_tabs(st.session_state.critique, st.session_state.canon)

        st.divider()
        st.subheader("Actions")

        action_left, action_right = st.columns(2)

        with action_left:
            can_rewrite = bool(st.session_state.weak_minutes)
            if st.button(
                "Auto-Rewrite Weak Minutes",
                disabled=not can_rewrite,
                width="stretch",
            ):
                run_rewrite(st.session_state.script_text, settings)
                st.rerun()
            if not can_rewrite:
                st.caption("Enabled once the audience flags at least one minute.")

        with action_right:
            if st.button(
                "Generate Scratch Audio",
                disabled=st.session_state.critique is None,
                width="stretch",
            ):
                run_synthesis(st.session_state.script_text, settings)
                st.rerun()
            if not settings["generate_audio"]:
                st.caption("TTS disabled in the sidebar; manifest only.")

        if st.session_state.manifest is not None:
            st.divider()
            st.subheader("Production export")
            render_audio(st.session_state.manifest)


if __name__ == "__main__":
    main()
