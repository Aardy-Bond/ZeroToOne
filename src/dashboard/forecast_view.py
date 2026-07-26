"""
Engagement Survival Forecast UI.

Rendering and actions for the forecast tab, kept out of app.py so the existing
Writers Room flow stays readable. Every number shown here is a relative proxy;
the disclaimer is rendered from a single shared constant so it cannot drift out
of sync with what the engine actually does.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from audio_engine.synthesizer import (
    AudioSynthesizer,
    SynthesisError,
    parse_script,
)
from retention_engine.directing_sheet import (
    assign_scene_indices,
    build_directing_sheet,
    select_preview_scenes,
)
from retention_engine.emotion_signals import emotion_source_note
from retention_engine.engagement_forecast import risk_band
from retention_engine.orchestrator import ForecastError, compare_forecasts, run_forecast
from retention_engine.quality_proxy import get_quality_proxy
from retention_engine.target_cohort import to_vector
from retention_engine.schemas import (
    EMOTION_DISCLAIMER,
    FORECAST_DISCLAIMER,
    CohortProfile,
    ForecastResult,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

GENRES = [
    "horror", "thriller", "romance", "drama", "fantasy",
    "sci_fi", "mystery", "comedy", "action", "literary",
]
AGE_BANDS = ["unspecified", "13-17", "18-24", "25-34", "35-44", "45+"]

RISK_COLORS = {"low": "#2f9e44", "elevated": "#f08c00", "high": "#e03131"}

FORECAST_STATE = {
    "cohort": None,
    "baseline_forecast": None,
    "current_forecast": None,
    "comparison": None,
    "preview_manifest": None,
    "forecast_error": "",
}


def init_forecast_state() -> None:
    for key, value in FORECAST_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------
# Sidebar: target cohort
# --------------------------------------------------------------------------


def render_cohort_sidebar() -> CohortProfile:
    """
    Collect the writer's declared target audience.

    This is stated intent, not a measured population, and the caption says so
    where the writer will actually read it.
    """
    st.sidebar.divider()
    st.sidebar.subheader("Target cohort")
    st.sidebar.caption(
        "Who you are writing for. A declared target, not a measured audience."
    )

    genre = st.sidebar.selectbox("Primary genre affinity", GENRES, index=0)
    pace = st.sidebar.select_slider(
        "Preferred pace", options=["slow", "balanced", "fast"], value="balanced"
    )
    complexity = st.sidebar.select_slider(
        "Complexity tolerance", options=["low", "medium", "high"], value="medium"
    )
    emotional = st.sidebar.selectbox(
        "Emotional preference", ["dread", "mystery", "action", "romance", "warmth"], index=0
    )
    content = st.sidebar.select_slider(
        "Violence / gore / dark-theme boundary",
        options=["low", "medium", "high"],
        value="medium",
    )
    mode = st.sidebar.selectbox("Listening mode", ["commute", "binge", "casual"], index=0)
    age_band = st.sidebar.selectbox(
        "Age band (writer-selected context only)", AGE_BANDS, index=0
    )

    st.sidebar.divider()
    st.sidebar.subheader("Forecast evidence")

    proxy = get_quality_proxy()
    if proxy.available:
        meta = proxy.metadata
        st.sidebar.success(f"Quality prior — {proxy.model_version}")
        st.sidebar.caption(
            f"Trained on {meta.get('sample_count', '?')} rated stories · "
            f"R² {meta.get('r2', '?')} · story-quality proxy, not retention."
        )
    else:
        st.sidebar.warning("Quality prior — not trained")
        st.sidebar.caption("Run `python scripts/train_quality_proxy.py` to enable.")

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


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


def run_forecast_action(
    script_text: str,
    cohort: CohortProfile,
    *,
    use_deep_dive: bool,
    is_revision: bool,
) -> None:
    """Run a forecast, promoting the previous one to baseline on a revision."""
    st.session_state.forecast_error = ""

    label = "Re-running forecast on the revision..." if is_revision else (
        "Measuring Scene DNA and building the survival proxy..."
    )
    with st.spinner(label):
        try:
            result = run_forecast(
                script_text,
                cohort=cohort,
                use_llm_deep_dive=use_deep_dive,
                use_emotion_classifier=True,
            )
        except (ForecastError, ValueError, Exception) as exc:
            st.session_state.forecast_error = f"Forecast failed: {exc}"
            return

    if is_revision and st.session_state.current_forecast is not None:
        st.session_state.baseline_forecast = st.session_state.current_forecast
        st.session_state.current_forecast = result
        st.session_state.comparison = compare_forecasts(
            st.session_state.baseline_forecast, result
        )
    else:
        st.session_state.baseline_forecast = None
        st.session_state.current_forecast = result
        st.session_state.comparison = None

    st.session_state.preview_manifest = None


def run_preview_action(
    forecast: ForecastResult,
    project_root: Path,
    *,
    generate_audio: bool,
    foley_triggers,
) -> None:
    """Render the retention-directed preview of the approved ending."""
    st.session_state.forecast_error = ""

    selected = select_preview_scenes(
        forecast,
        spoken_word_count=lambda scene: sum(
            chunk.word_count for chunk in parse_script(scene.text)
        ),
    )
    if not selected:
        st.session_state.forecast_error = "No scenes available for a preview."
        return

    preview_text = "\n\n".join(a.scene.text for a in selected)
    chunks = parse_script(preview_text)
    if not chunks:
        st.session_state.forecast_error = (
            "The selected ending has no spoken lines. Add character cues or a "
            "NARRATOR (V.O.) block so the preview has something to voice."
        )
        return

    scene_indices = assign_scene_indices(
        chunks, [a.scene for a in selected], parse_script
    )
    sheet = build_directing_sheet(chunks, forecast, scene_indices)

    summary = {
        "overall_survival_proxy": round(forecast.overall_survival, 1),
        "primary_cohort_survival_proxy": round(forecast.primary_curve.final_survival, 1),
        "unlock_pull_index": forecast.cliffhanger.unlock_pull_index,
        "hook_strength": forecast.cliffhanger.hook_strength,
        "cliffhanger_types": forecast.cliffhanger.types,
        "preview_scenes": [a.scene.index for a in selected],
        "disclaimer": FORECAST_DISCLAIMER,
    }

    label = "Rendering directed preview..." if generate_audio else "Building preview manifest..."
    with st.spinner(label):
        try:
            st.session_state.preview_manifest = AudioSynthesizer(
                output_dir=project_root / "output"
            ).synthesize_directed_preview(
                preview_text,
                foley_triggers,
                sheet,
                generate_audio=generate_audio,
                forecast_summary=summary,
            )
        except (SynthesisError, ValueError) as exc:
            st.session_state.forecast_error = f"Preview synthesis failed: {exc}"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_headline(forecast: ForecastResult) -> None:
    st.info(FORECAST_DISCLAIMER, icon="ℹ️")

    cols = st.columns(4)
    cols[0].metric(
        "Engagement Survival Proxy",
        f"{forecast.overall_survival:.0f} / 100",
        help="Relative proxy across the primary and what-if cohorts. Not a listener percentage.",
    )
    cols[1].metric(
        "Primary cohort",
        f"{forecast.primary_curve.final_survival:.0f} / 100",
    )
    cols[2].metric(
        "Unlock Pull Index",
        f"{forecast.cliffhanger.unlock_pull_index:.0f} / 100",
        help="Relative narrative pull toward the next episode. Not a conversion rate.",
    )
    high = sum(1 for s in forecast.primary_curve.scenes if risk_band(s.hazard) != "low")
    cols[3].metric("Risk scenes", high)


def render_survival_curves(forecast: ForecastResult) -> None:
    """
    The survival proxy, with the spread across cohorts as its own reference.

    Four separate curves was the wrong picture. They all start at 100 and all
    slope down, so the eye reads four near-identical lines and takes away
    nothing. What is actually informative is where they *diverge*: that gap is
    how much the result depends on assumptions about the audience rather than
    on the writing. So the counterfactuals collapse into a shaded band and the
    target cohort is the only line, which turns the chart into one claim —
    here is the forecast, and here is how much the audience assumption moves
    it.
    """
    st.markdown("#### Engagement Survival Proxy")
    st.caption(
        "Starts at 100 and decays by the hazard assigned to each scene. The "
        "band is the range across the what-if cohorts, so a wide band means "
        "the result depends heavily on who you think is listening."
    )

    if not HAS_PLOTLY:
        st.line_chart({"primary": forecast.primary_curve.curve})
        return

    primary = forecast.primary_curve.curve
    x = list(range(len(primary)))
    figure = go.Figure()

    others = [c.curve for c in forecast.counterfactual_curves]
    if others:
        lower = [min(c[i] for c in others) for i in x]
        upper = [max(c[i] for c in others) for i in x]
        figure.add_trace(
            go.Scatter(
                x=x + x[::-1], y=upper + lower[::-1],
                fill="toself", fillcolor="rgba(230,26,75,0.10)",
                mode="lines", line={"width": 0}, hoverinfo="skip",
                name="Range across what-if cohorts",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=x, y=primary, mode="lines+markers",
            name=f"{forecast.cohort.label}",
            line={"color": "#E61A4B", "width": 4},
            marker={"size": 9},
            hovertemplate="After scene %{x}<br>Survival proxy %{y:.1f}<extra></extra>",
        )
    )

    for scene in forecast.primary_curve.scenes:
        band = risk_band(scene.hazard)
        if band == "low":
            continue
        figure.add_vrect(
            x0=scene.scene_index - 0.5, x1=scene.scene_index + 0.5,
            fillcolor=RISK_COLORS[band], opacity=0.12, line_width=0,
        )

    # The single most useful thing on this chart is where the floor gave way.
    drops = [(primary[i - 1] - primary[i], i) for i in range(1, len(primary))]
    if drops:
        worst, at = max(drops)
        if worst >= 3:
            figure.add_annotation(
                x=at, y=primary[at],
                text=f"biggest loss<br>scene {at}, −{worst:.0f}",
                showarrow=True, arrowhead=2, ax=34, ay=38,
                font={"size": 11, "color": "#E61A4B"},
                bgcolor="rgba(255,255,255,0.85)", bordercolor="#E61A4B",
                borderwidth=1, borderpad=4,
            )

    floor = min([min(c) for c in others] + [min(primary)]) if others else min(primary)
    figure.update_layout(
        height=400,
        margin={"l": 10, "r": 20, "t": 30, "b": 10},
        yaxis={
            "range": [max(0, floor - 12), 103],
            "title": "Relative survival proxy",
            "automargin": True,
        },
        xaxis={
            "title": "Scene", "dtick": 1, "automargin": True, "zeroline": False,
        },
        legend={"orientation": "h", "y": -0.2},
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")

    if others:
        finals = [c[-1] for c in others] + [primary[-1]]
        st.caption(
            f"Ends at **{primary[-1]:.0f}** for your target cohort, and between "
            f"**{min(finals):.0f}** and **{max(finals):.0f}** across the what-if "
            f"cohorts. Only the shape and the gaps carry meaning here; the "
            f"absolute number is not a percentage of anybody."
        )


def render_narrative_ekg(forecast: ForecastResult) -> None:
    """
    Three stacked readings instead of one tangle.

    The previous chart drew six 0-1 series on a single axis, and measurement on
    a real episode showed why that reads as noise: the quality prior moved
    0.01 across the whole episode (a dead flat line), tension and emotional
    intensity tracked each other almost exactly (the same line drawn twice),
    cliffhanger strength sat at zero for most scenes, and everything lived in
    the lower half of the axis while the top half stayed empty.

    The deeper problem was that a number between 0 and 1 means nothing without
    something to compare it against. Is tempo 0.22 slow? Only relative to what
    this audience wants. So each row now carries the cohort's own target as a
    reference line, the series are separated so each gets its own scale, and
    the flat and duplicated ones are gone.
    """
    st.markdown("#### Narrative EKG")
    st.caption(
        "Each row is measured against what your target cohort wants, shown as "
        "the dashed line. Distance from that line is the reading. "
        + emotion_source_note([a.emotion for a in forecast.scenes])
    )

    scenes = forecast.scenes
    indices = [a.scene.index for a in scenes]
    vector = to_vector(forecast.cohort)

    if not HAS_PLOTLY:
        st.line_chart(
            {
                "tension": [a.dna.tension for a in scenes],
                "scene tempo": [a.dna.scene_tempo for a in scenes],
                "exposition": [a.dna.exposition_ratio for a in scenes],
            }
        )
        return

    figure = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        row_heights=[0.36, 0.32, 0.32],
        specs=[[{}], [{}], [{"secondary_y": True}]],
        subplot_titles=(
            "Pace — how fast each scene moves, against your cohort's preference",
            "Tension and emotional intensity",
            "Exposition against tolerance, and threads left open",
        ),
    )

    # ---- Row 1: pace against the cohort's target ------------------------
    # The target is drawn first so the tempo trace can fill down to it. The
    # shaded gap is the actual reading: how far this scene sits from the pace
    # the cohort came for.
    tempo = [a.dna.scene_tempo for a in scenes]
    figure.add_trace(
        go.Scatter(
            x=indices, y=[vector.pace_target] * len(indices),
            mode="lines", name="Pace your cohort wants",
            line={"color": "#424245", "width": 1.5, "dash": "dash"},
            opacity=0.6, hovertemplate="Wants %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=indices, y=tempo, mode="lines+markers", name="Scene tempo",
            line={"color": "#424245", "width": 3},
            fill="tonexty", fillcolor="rgba(66,66,69,0.12)",
            hovertemplate="Tempo %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ---- Row 2: tension and feeling -------------------------------------
    # Emotional intensity is a filled area rather than a line because it
    # tracks tension closely: as a line it vanishes underneath wherever the
    # two agree, which looks like a broken chart. As a band, agreement reads
    # as the line sitting on top of its own shading, and the scenes where
    # emotion runs ahead of tension are the ones that stand out.
    figure.add_trace(
        go.Scatter(
            x=indices, y=[a.emotion.arousal for a in scenes],
            mode="lines", name="Emotional intensity",
            line={"color": "#f76707", "width": 1.5},
            fill="tozeroy", fillcolor="rgba(247,103,7,0.13)",
            hovertemplate="Intensity %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=indices, y=[a.dna.tension for a in scenes],
            mode="lines+markers", name="Tension",
            line={"color": "#e03131", "width": 3},
            hovertemplate="Tension %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )

    hooks = [(i, a.dna.cliffhanger_strength) for i, a in zip(indices, scenes)
             if a.dna.cliffhanger_strength > 0.05]
    if hooks:
        figure.add_trace(
            go.Scatter(
                x=[h[0] for h in hooks], y=[h[1] for h in hooks],
                mode="markers", name="Hook at scene end",
                marker={"color": "#1D1D1F", "size": 13, "symbol": "diamond"},
                hovertemplate="Hook %{y:.2f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # ---- Row 3: exposition against tolerance, plus open threads ---------
    exposition = [a.dna.exposition_ratio for a in scenes]
    figure.add_trace(
        go.Bar(
            x=indices, y=[a.structural.payoff_debt for a in scenes],
            name="Threads left open",
            marker={"color": "rgba(134,142,150,0.30)"},
            hovertemplate="%{y} open<extra></extra>",
        ),
        row=3, col=1, secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(
            x=indices, y=exposition, mode="lines+markers", name="Exposition",
            line={"color": "#495057", "width": 3},
            hovertemplate="Exposition %{y:.2f}<extra></extra>",
        ),
        row=3, col=1, secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=indices, y=[vector.exposition_tolerance] * len(indices),
            mode="lines", name="Exposition your cohort tolerates",
            line={"color": "#E61A4B", "width": 1.5, "dash": "dash"},
            opacity=0.7, hovertemplate="Tolerates %{y:.2f}<extra></extra>",
        ),
        row=3, col=1, secondary_y=False,
    )

    _annotate_ekg(figure, forecast, vector)

    figure.update_layout(
        height=660,
        margin={"l": 10, "r": 10, "t": 56, "b": 10},
        legend={"orientation": "h", "y": -0.10},
        hovermode="x unified",
        barmode="overlay",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    for row in (1, 2, 3):
        figure.update_xaxes(
            dtick=1, showgrid=False, zeroline=False, row=row, col=1,
        )
        # Without an explicit format Plotly strips the leading zero and shows
        # 0.6 as "6", which reads as a completely different quantity.
        figure.update_yaxes(
            tickformat=".1f", automargin=True, nticks=4, row=row, col=1,
        )
    figure.update_xaxes(title_text="Scene", automargin=True, row=3, col=1)

    # Each row autoscales to its own data with headroom, so a series that only
    # ever moves between 0.13 and 0.33 is still legible instead of being
    # flattened against the floor of a shared 0-1 axis.
    figure.update_yaxes(
        range=_padded(tempo + [vector.pace_target]), row=1, col=1, title_text="",
    )
    # The hook diamonds have to be inside the range or they sit clipped on the
    # floor of the row, which is where they were.
    figure.update_yaxes(
        range=_padded(
            [a.dna.tension for a in scenes]
            + [a.emotion.arousal for a in scenes]
            + [h[1] for h in hooks]
            + [0.0]
        ),
        row=2, col=1, title_text="",
    )
    figure.update_yaxes(
        range=_padded(exposition + [vector.exposition_tolerance]),
        row=3, col=1, secondary_y=False, title_text="",
    )
    figure.update_yaxes(
        showgrid=False, row=3, col=1, secondary_y=True, title_text="open",
        rangemode="tozero",
    )
    for note in figure.layout.annotations[:3]:
        note.font.size = 12
        note.x = 0
        note.xanchor = "left"

    st.plotly_chart(figure, width="stretch")

    prior = [a.quality.score for a in scenes]
    if prior and max(prior) - min(prior) < 0.05:
        # Kept off the chart deliberately: a line that never moves reads as a
        # signal that says nothing, when the honest reading is that this layer
        # does not discriminate at scene length.
        st.caption(
            f"Narrative quality prior sits at {sum(prior) / len(prior):.2f} "
            f"across every scene (spread {max(prior) - min(prior):.2f}), so it "
            f"is reported as one number rather than drawn as a line. It was "
            f"trained on complete stories and separates them poorly at the "
            f"length of a single scene."
        )


def _padded(values: list[float], pad: float = 0.10) -> list[float]:
    """A y-range that frames the data instead of squashing it against 0-1."""
    if not values:
        return [0, 1]
    low, high = min(values), max(values)
    if high - low < 0.15:  # near-flat: give it a window so it is not a wire
        middle = (high + low) / 2
        low, high = middle - 0.12, middle + 0.12
    return [max(0.0, low - pad), min(1.0, high + pad)]


def _annotate_ekg(figure, forecast: ForecastResult, vector) -> None:
    """Mark the moments a writer should look at first."""
    scenes = forecast.scenes

    for scene in forecast.primary_curve.scenes:
        band = risk_band(scene.hazard)
        if band == "low":
            continue
        for row in (1, 2, 3):
            figure.add_vrect(
                x0=scene.scene_index - 0.45, x1=scene.scene_index + 0.45,
                fillcolor=RISK_COLORS[band], opacity=0.10, line_width=0,
                row=row, col=1,
            )

    slowest = min(scenes, key=lambda a: a.dna.scene_tempo)
    if vector.pace_target - slowest.dna.scene_tempo > 0.2:
        figure.add_annotation(
            x=slowest.scene.index, y=slowest.dna.scene_tempo,
            text="slowest, well under target", showarrow=True, arrowhead=2, ay=26,
            font={"size": 10, "color": "#424245"}, row=1, col=1,
        )

    strongest = max(scenes, key=lambda a: a.dna.cliffhanger_strength)
    if strongest.dna.cliffhanger_strength > 0.3:
        figure.add_annotation(
            x=strongest.scene.index, y=strongest.dna.cliffhanger_strength,
            text="strongest hook", showarrow=True, arrowhead=2, ay=-26,
            font={"size": 10, "color": "#1D1D1F"}, row=2, col=1,
        )

    # Skipped where it would land on a hook diamond, which is a marker the
    # writer is more likely to be looking for.
    flattest = min(scenes, key=lambda a: a.dna.tension)
    hooked = {a.scene.index for a in scenes if a.dna.cliffhanger_strength > 0.05}
    if flattest.dna.tension < 0.35 and flattest.scene.index not in hooked:
        figure.add_annotation(
            x=flattest.scene.index, y=flattest.dna.tension,
            text="flattest scene", showarrow=True, arrowhead=2, ay=32,
            font={"size": 10, "color": "#e03131"}, row=2, col=1,
        )

    over = [a for a in scenes
            if a.dna.exposition_ratio > vector.exposition_tolerance]
    if over:
        worst = max(over, key=lambda a: a.dna.exposition_ratio)
        figure.add_annotation(
            x=worst.scene.index, y=worst.dna.exposition_ratio,
            text="over tolerance", showarrow=True, arrowhead=2, ay=-24,
            font={"size": 10, "color": "#E61A4B"}, row=3, col=1,
        )


def render_cliffhanger_lab(forecast: ForecastResult) -> None:
    report = forecast.cliffhanger

    st.markdown("#### Cliffhanger Lab")
    st.caption(
        "Analysis of the episode ending. The Unlock Pull Index is a relative "
        "measure of narrative pull, not a predicted conversion rate."
    )

    top = st.columns(3)
    top[0].metric("Ending type", " + ".join(t.replace("_", " ") for t in report.types))
    top[1].metric("Hook strength", f"{report.hook_strength:.0f} / 100")
    top[2].metric("Unlock Pull Index", f"{report.unlock_pull_index:.0f} / 100")

    components = [
        ("Stakes", report.stakes, False),
        ("Information gap", report.information_gap, False),
        ("Surprise / reversal", report.surprise, False),
        ("Emotional investment", report.emotional_investment, False),
        ("Novelty vs earlier hooks", report.novelty, False),
        ("Payoff debt", report.payoff_debt_penalty, True),
        ("False-resolution risk", report.false_resolution_risk, True),
    ]

    for label, value, is_penalty in components:
        bar_col, value_col = st.columns([5, 1])
        with bar_col:
            st.caption(("− " if is_penalty else "+ ") + label)
            st.progress(min(1.0, max(0.0, value)))
        value_col.caption(f"{value:.2f}")

    if "weak_resolved" in report.types:
        st.warning(report.recommendation)
    elif report.unlock_pull_index >= 60:
        st.success(report.recommendation)
    else:
        st.info(report.recommendation)


def render_risk_evidence(forecast: ForecastResult) -> None:
    st.markdown("#### Risk scenes and likely contributors")

    ranked = [s for s in forecast.risk_ranking if risk_band(s.hazard) != "low"]
    if not ranked:
        st.success("No scene reached an elevated hazard band for this cohort.")
        ranked = forecast.risk_ranking[:1]

    explanations = {e.scene_index: e for e in forecast.risk_explanations}

    for scene in ranked[:4]:
        analysis = forecast.scene_by_index(scene.scene_index)
        if analysis is None:
            continue

        band = risk_band(scene.hazard)
        with st.expander(
            f"Scene {scene.scene_index} — {band.upper()} risk · "
            f"survival proxy {scene.survival:.0f}/100 · {analysis.scene.heading[:44]}",
            expanded=(band == "high"),
        ):
            st.markdown("**Likely engagement contributors**")
            st.dataframe(
                [
                    {
                        "Factor": c.factor.replace("_", " "),
                        "Effect on hazard": f"{c.delta:+.3f}",
                        "Evidence": c.detail,
                    }
                    for c in scene.contributions
                ],
                width="stretch",
                hide_index=True,
            )

            explanation = explanations.get(scene.scene_index)
            if explanation is not None:
                source = (
                    "LLM deep dive" if explanation.source == "llm_deep_dive"
                    else "deterministic evidence"
                )
                st.caption(f"Explanation source: {source}")
                st.markdown("**Why this scene is risky**")
                st.write(explanation.why_risky)
                st.markdown("**What the target cohort expects**")
                st.write(explanation.cohort_expectation)
                st.markdown("**Surgical fix**")
                st.success(explanation.surgical_fix)
                st.markdown("**Trade-off**")
                st.warning(explanation.trade_off)

            with st.popover("Scene text"):
                st.code(analysis.scene.text, language="text")


def render_comparison(comparison) -> None:
    st.markdown("#### Before / after comparison")

    headline_metric = comparison.metrics[0] if comparison.metrics else None
    if headline_metric is None or not headline_metric.significant:
        st.info(comparison.headline)
    elif headline_metric.improved:
        st.success(comparison.headline)
    else:
        st.error(comparison.headline)

    st.caption(comparison.disclaimer)

    st.dataframe(
        [
            {
                "Metric": m.label,
                "Before": m.before,
                "After": m.after,
                "Change": f"{m.delta:+.1f}",
                "Reading": (
                    "moved" if m.significant else f"within noise (±{m.noise_band:.1f})"
                ),
            }
            for m in comparison.metrics
        ],
        width="stretch",
        hide_index=True,
    )

    if any(not m.significant and m.noise_band for m in comparison.metrics):
        st.caption(
            "Re-measuring an unchanged episode moves these numbers, so a change "
            "smaller than the stated band is sensor noise rather than an effect "
            "of the revision."
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**Primary evidence**")
        for line in comparison.primary_evidence:
            st.markdown(f"- {line}")
    with right:
        st.markdown("**Risk scene movement**")
        for line in comparison.risk_scene_movement:
            st.markdown(f"- {line}")


def render_ekg_overlay(baseline: ForecastResult, revised: ForecastResult) -> None:
    """Before/after survival curves on one set of axes."""
    if not HAS_PLOTLY:
        return

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(range(len(baseline.primary_curve.curve))),
            y=baseline.primary_curve.curve,
            mode="lines+markers", name="Before",
            line={"color": "#adb5bd", "width": 3, "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=list(range(len(revised.primary_curve.curve))),
            y=revised.primary_curve.curve,
            mode="lines+markers", name="After",
            line={"color": "#E61A4B", "width": 4},
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        yaxis={"range": [0, 105], "title": "Relative survival proxy"},
        xaxis={"title": "Scene", "dtick": 1},
        legend={"orientation": "h", "y": -0.25},
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch")


def render_preview(manifest: dict) -> None:
    st.markdown("#### Retention-directed audio preview")

    summary = manifest.get("forecast_summary", {})
    cols = st.columns(4)
    cols[0].metric("Runtime", manifest["total_runtime"])
    cols[1].metric("Chunks", manifest["chunk_count"])
    cols[2].metric(
        "Survival proxy that directed this",
        f"{summary.get('primary_cohort_survival_proxy', 0):.0f} / 100",
    )
    cols[3].metric("Unlock Pull Index", f"{summary.get('unlock_pull_index', 0):.0f} / 100")

    st.caption(
        f"TTS model: {manifest.get('tts_model') or 'not rendered'} · "
        f"scenes {summary.get('preview_scenes', [])} · "
        f"casting {json.dumps(manifest['casting'])}"
    )

    sheet = {e["chunk_index"]: e for e in manifest.get("retention_directing_sheet", [])}
    audio_dir = Path(manifest["audio_dir"])

    for chunk in manifest["chunks"]:
        entry = sheet.get(chunk["index"], {})
        band = entry.get("engagement_risk", "low")
        dot = {"low": "🟢", "elevated": "🟠", "high": "🔴"}.get(band, "⚪")

        st.markdown(
            f"{dot} **[{chunk['start']}] {chunk['character']}** "
            f"({chunk['voice']}, {chunk['duration_seconds']:.1f}s) — "
            f"*{entry.get('narrative_role', 'n/a').replace('_', ' ')}*"
        )
        st.caption(chunk["text"])

        if manifest["audio_generated"] and chunk["audio_file"]:
            path = audio_dir / chunk["audio_file"]
            if path.exists():
                st.audio(str(path))
            else:
                st.caption(f"Missing audio file: {path}")

        if entry:
            st.caption(
                f"**Direction:** {entry.get('delivery_tempo', '')} @ "
                f"{entry.get('target_speed', 1.0):.2f}x · pause before reveal "
                f"{entry.get('pause_before_reveal_ms', 0)}ms · emotion "
                f"{entry.get('dominant_emotion', '')}"
            )
            st.caption(f"**Note:** {entry.get('instruction', '')}")
            if entry.get("foley"):
                st.caption(f"**Foley:** {'; '.join(entry['foley'])}")
            if entry.get("hook_role"):
                st.caption(f"**Hook role:** {entry['hook_role']}")

        st.divider()

    with st.expander("Retention directing sheet (JSON)"):
        st.json(manifest.get("retention_directing_sheet", []))

    download = st.columns(2)
    manifest_path = Path(manifest["manifest_path"])
    cue_path = Path(manifest["cue_sheet_path"])

    if manifest_path.exists():
        download[0].download_button(
            "Download preview_manifest.json",
            manifest_path.read_bytes(),
            file_name=manifest_path.name,
            mime="application/json",
            width="stretch",
        )
    if cue_path.exists():
        download[1].download_button(
            "Download preview cue sheet",
            cue_path.read_bytes(),
            file_name=cue_path.name,
            mime="text/plain",
            width="stretch",
        )


def render_scene_table(forecast: ForecastResult) -> None:
    with st.expander("All scenes — measurements"):
        if any(a.scene.inferred for a in forecast.scenes):
            st.info(
                "This draft had no slug lines, so scene boundaries were inferred "
                "from paragraph structure. The segments below are what was measured."
            )
        st.dataframe(
            [
                {
                    "Scene": a.scene.index,
                    "Heading": a.scene.heading[:38],
                    "Words": a.scene.word_count,
                    "Hazard": f"{s.hazard:.3f}",
                    "Survival": f"{s.survival:.0f}",
                    "Exposition": f"{a.dna.exposition_ratio:.2f}",
                    "Tension": f"{a.dna.tension:.2f}",
                    "Tempo": f"{a.dna.scene_tempo:.2f}",
                    "Complexity": f"{a.dna.complexity:.2f}",
                    "Cliff": f"{a.dna.cliffhanger_strength:.2f}",
                    "Debt": a.structural.payoff_debt,
                    "Quality": f"{a.quality.score:.2f}" if a.quality.available else "n/a",
                }
                for a, s in zip(forecast.scenes, forecast.primary_curve.scenes)
            ],
            width="stretch",
            hide_index=True,
        )
        st.caption(EMOTION_DISCLAIMER)
