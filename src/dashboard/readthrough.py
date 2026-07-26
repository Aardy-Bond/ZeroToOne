"""
Results, in a writer's language.

Two surfaces. The story check reports contradictions and dangling threads, and
it is the one a writer will use constantly. The read-through reports where
readers drift, and it wraps the retention engine unchanged -- every original
number is still there, one expander away, under "Show the analysis".

The translation is presentational only. Nothing here rounds a proxy into a
claim it cannot support.
"""

from __future__ import annotations

import streamlit as st

from dashboard import forecast_view, language, state, theme
from projects.schemas import ContinuityReport

SEVERITY_ORDER = ["high", "medium", "low"]


# ---------------------------------------------------------------------------
# Story check
# ---------------------------------------------------------------------------


def render_story_check(report: ContinuityReport) -> None:
    st.markdown("### Story check")

    if report.is_clean:
        st.markdown(
            f"<div class='anu-card' style='border-left:3px solid {theme.CALM}'>"
            f"<h4 style='color:{theme.CALM}'>Nothing conflicts</h4>"
            f"<p>{report.note}</p></div>",
            unsafe_allow_html=True,
        )
        _footnote(report)
        return

    counts = {s: len(_of_severity(report, s)) for s in SEVERITY_ORDER}
    pills = ""
    if counts["high"]:
        pills += theme.pill(f"{counts['high']} to fix", "risk")
    if counts["medium"]:
        pills += theme.pill(f"{counts['medium']} to look at", "accent")
    if counts["low"]:
        pills += theme.pill(f"{counts['low']} to note")
    st.markdown(pills, unsafe_allow_html=True)
    st.write("")

    for severity in SEVERITY_ORDER:
        for finding in _of_severity(report, severity):
            detail = ""
            if finding.established:
                part = (
                    f"Part {finding.established_part + 1}: "
                    if finding.established_part is not None
                    else ""
                )
                detail = f"{part}{_escape(finding.established)}"
                if finding.quote and finding.quote.strip() != finding.established.strip():
                    detail += f"<br><span style='opacity:.75'>“{_escape(finding.quote)}”</span>"

            st.markdown(
                theme.finding_card(
                    finding.label,
                    _escape(finding.what),
                    severity,
                    detail,
                    _escape(finding.suggestion),
                ),
                unsafe_allow_html=True,
            )

    _footnote(report)


def _of_severity(report: ContinuityReport, severity: str) -> list:
    return [f for f in report.findings if f.severity == severity]


def _footnote(report: ContinuityReport) -> None:
    bits = [
        f"Checked against {language.plural(report.facts_checked, 'established fact')} "
        f"true on {report.branch_name or 'this timeline'} at this point."
    ]
    if not report.llm_used:
        bits.append("Structural checks only.")
    st.caption(" ".join(bits))

    if report.note and not report.is_clean:
        st.caption(report.note)


# ---------------------------------------------------------------------------
# Read-through
# ---------------------------------------------------------------------------


def run(draft: str, *, is_revision: bool = False) -> None:
    """Run the retention engine over the draft, keeping the previous one as a baseline."""
    from retention_engine.orchestrator import compare_forecasts, run_forecast

    cohort = st.session_state.get("cohort")

    with st.spinner("Reading it the way an audience would..."):
        try:
            result = run_forecast(
                draft,
                cohort=cohort,
                use_llm_deep_dive=st.session_state.get("deep_dive", True),
                use_emotion_classifier=True,
            )
        except Exception as exc:
            state.fail(f"The read-through could not run: {exc}")
            return

    if is_revision and st.session_state.readthrough is not None:
        baseline = st.session_state.readthrough
        st.session_state.readthrough_baseline = baseline
        try:
            st.session_state.comparison = compare_forecasts(baseline, result)
        except Exception:
            st.session_state.comparison = None
    else:
        st.session_state.readthrough_baseline = None
        st.session_state.comparison = None

    st.session_state.readthrough = result


def render_readthrough(forecast, *, baseline=None, comparison=None) -> None:
    st.markdown("### Read-through")
    st.caption(language.DISCLAIMER)

    survival = forecast.overall_survival
    hook = forecast.cliffhanger.unlock_pull_index

    top = st.columns(3)
    top[0].metric("Readers still with you", f"{survival:.0f} / 100")
    top[1].metric("Pull to the next part", f"{hook:.0f} / 100")
    top[2].metric(
        "Prose strength",
        f"{forecast.quality_prior.score:.2f}"
        if forecast.quality_prior.available
        else "—",
    )

    st.markdown(
        f"<p style='color:{theme.MUTED};margin-top:-.4rem'>"
        f"{language.survival_sentence(survival)} {language.hook_sentence(hook)}</p>",
        unsafe_allow_html=True,
    )

    _where_they_drift(forecast)

    if comparison is not None:
        st.write("")
        _comparison(comparison)

    _preview_action(forecast)
    _open_producer_cta()

    with st.expander("Show the analysis"):
        st.caption(
            "The same numbers in the engine's own terms: survival proxy, hazard "
            "per scene, Unlock Pull Index, cohort curves, and the Narrative EKG."
        )
        forecast_view.render_headline(forecast)
        st.divider()
        forecast_view.render_survival_curves(forecast)
        st.divider()
        forecast_view.render_narrative_ekg(forecast)
        st.divider()
        forecast_view.render_cliffhanger_lab(forecast)
        st.divider()
        forecast_view.render_risk_evidence(forecast)
        forecast_view.render_scene_table(forecast)
        if comparison is not None and baseline is not None:
            st.divider()
            forecast_view.render_comparison(comparison)
            forecast_view.render_ekg_overlay(baseline, forecast)


def _preview_action(forecast) -> None:
    """Render the ending as directed audio, so the writer can hear the hook land."""
    from pathlib import Path

    st.write("")
    row = st.columns([2, 1])
    with row[0]:
        st.caption(
            "Hear the ending performed, with the pacing and pauses the analysis "
            "suggests."
        )
    with row[1]:
        if st.button("Hear the ending", width="stretch"):
            critique = st.session_state.get("critique")
            forecast_view.run_preview_action(
                forecast,
                Path(__file__).resolve().parents[2],
                generate_audio=True,
                foley_triggers=critique.foley_triggers if critique else [],
            )
            if st.session_state.get("forecast_error"):
                state.fail(st.session_state.forecast_error)
                st.session_state.forecast_error = ""
            st.rerun()

    if st.session_state.get("preview_manifest"):
        forecast_view.render_preview(st.session_state.preview_manifest)


def _open_producer_cta() -> None:
    """Hand the forecast to Production's AI Producer without cluttering the desk."""
    st.write("")
    row = st.columns([2, 1])
    with row[0]:
        st.caption(
            "Open the AI Producer on Production — casting, line cues, sound, and "
            "a short marketing brief directed by these scores."
        )
    with row[1]:
        if st.button("Open AI Producer", width="stretch"):
            st.session_state.open_producer = True
            state.go("production")


def _where_they_drift(forecast) -> None:
    from retention_engine.engagement_forecast import risk_band

    risky = [
        (s, risk_band(s.hazard))
        for s in forecast.primary_curve.scenes
        if risk_band(s.hazard) != "low"
    ]

    if not risky:
        st.markdown(
            f"<div class='anu-card' style='border-left:3px solid {theme.CALM}'>"
            f"<h4 style='color:{theme.CALM}'>Nothing drags</h4>"
            f"<p>No scene stands out as a place readers are likely to leave.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("**Where readers drift**")

    explanations = {e.scene_index: e for e in forecast.risk_explanations}
    for scene, band in risky:
        explanation = explanations.get(scene.scene_index)
        what = (
            explanation.why_risky
            if explanation
            else "This scene loses momentum against the rest."
        )
        fix = explanation.surgical_fix if explanation else ""

        st.markdown(
            theme.finding_card(
                f"Scene {scene.scene_index} · {language.risk_label(band)}",
                _escape(what),
                "high" if band == "high" else "medium",
                "",
                _escape(fix),
            ),
            unsafe_allow_html=True,
        )


def _comparison(comparison) -> None:
    headline = comparison.metrics[0] if comparison.metrics else None

    if headline is None or not headline.significant:
        st.info(comparison.headline)
    elif headline.improved:
        st.success(comparison.headline)
    else:
        st.error(comparison.headline)

    moved = [m for m in comparison.metrics if m.significant]
    if not moved:
        st.caption(
            "Nothing moved beyond what the reading naturally varies by between runs."
        )
        return

    for metric in moved:
        arrow = "improved" if metric.delta > 0 else "weakened"
        st.markdown(
            f"<div style='font-size:.88rem;color:{theme.MUTED};margin:.15rem 0'>"
            f"{metric.label} {arrow} by {abs(metric.delta):.1f}</div>",
            unsafe_allow_html=True,
        )


def _escape(text: str) -> str:
    return (text or "").replace("<", "&lt;").replace(">", "&gt;")
