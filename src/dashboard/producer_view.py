"""
Sparse AI Producer UI — annotated script without caption walls.

Brief strip → scene headers with defaults → lines with cue pills only when
non-default → marketing callout → optional JSON / cue sheet.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from dashboard import brand, theme
from producer.schemas import LineCue, ProducerPlan, SceneDirection, SpeakableLine


def render_producer(
    plan: ProducerPlan,
    lines: list[SpeakableLine],
    *,
    forecast=None,
    manifest: dict | None = None,
) -> None:
    st.markdown("### AI Producer")
    st.caption(
        "Casting, delivery, and sound directed by the retention read-through — "
        "simulated scores, not live listener data."
    )

    _brief_strip(plan, forecast)
    st.write("")
    _script_flow(plan, lines)

    marketing = plan.marketing
    brand.callout(
        "accent",
        marketing.title_treatment or "Episode marketing",
        _marketing_body(marketing),
    )
    st.caption(marketing.disclaimer or plan.disclaimer)

    if manifest:
        st.write("")
        st.markdown("#### Hear this plan")
        _render_manifest_audio(manifest)

    with st.expander("Full producer plan (JSON)"):
        st.json(plan.model_dump())

    if manifest and manifest.get("cue_sheet_path"):
        cue_path = Path(manifest["cue_sheet_path"])
        if cue_path.exists():
            st.download_button(
                "Download producer cue sheet",
                cue_path.read_bytes(),
                file_name=cue_path.name,
                mime="text/plain",
                width="stretch",
            )


def _brief_strip(plan: ProducerPlan, forecast) -> None:
    survival = f"{forecast.overall_survival:.0f}" if forecast is not None else "—"
    upi = (
        f"{forecast.cliffhanger.unlock_pull_index:.0f}"
        if forecast is not None
        else "—"
    )
    cast_bits = " · ".join(
        f"{html.escape(c.character)}→{html.escape(c.voice)}" for c in plan.casting[:8]
    )
    if len(plan.casting) > 8:
        cast_bits += " …"

    brand.metric_strip(
        [
            ("Survival that directed this", f"{survival} / 100", "proxy"),
            ("Unlock pull", f"{upi} / 100", "ending hook"),
            ("Cued lines", str(len(plan.line_cues)), "selective"),
            ("Sound cues", str(len(plan.sound_cues)), "board"),
        ]
    )
    st.markdown(
        f"<p style='margin:.35rem 0 .15rem;color:{theme.INK};font-weight:550'>"
        f"{html.escape(plan.strategy)}</p>",
        unsafe_allow_html=True,
    )
    if cast_bits:
        st.markdown(
            f"<div style='font-size:.78rem;color:{theme.MUTED};letter-spacing:.01em'>"
            f"{cast_bits}</div>",
            unsafe_allow_html=True,
        )


def _script_flow(plan: ProducerPlan, lines: list[SpeakableLine]) -> None:
    scene_dir = {s.scene_index: s for s in plan.scenes}
    cues = {c.line_index: c for c in plan.line_cues}
    sounds: dict[int, list[str]] = {}
    for cue in plan.sound_cues:
        sounds.setdefault(cue.line_index, []).append(cue.effect)

    current_scene: int | None = None
    for line in lines:
        if line.scene_index != current_scene:
            current_scene = line.scene_index
            scene = scene_dir.get(current_scene) or SceneDirection(scene_index=current_scene)
            st.markdown(
                f"<div class='anu-meta' style='margin:1rem 0 .35rem'>"
                f"Scene {current_scene}</div>"
                f"<div style='font-size:.8rem;color:{theme.MUTED};margin-bottom:.5rem'>"
                f"{html.escape(_scene_default_label(scene))}</div>",
                unsafe_allow_html=True,
            )

        cue = cues.get(line.index)
        fx = sounds.get(line.index, [])
        pills = _cue_pills(cue, scene_dir.get(line.scene_index))
        fx_html = ""
        if fx:
            fx_html = (
                f" <span style='color:{theme.FAINT};font-size:.75rem'>"
                f"[FX: {html.escape(', '.join(fx))}]</span>"
            )

        st.markdown(
            f"<div style='margin:.45rem 0 .15rem'>"
            f"<span style='font-weight:650;color:{theme.INK};font-size:.82rem'>"
            f"{html.escape(line.character)}</span>"
            f"{pills}{fx_html}</div>"
            f"<div style='font-size:.95rem;line-height:1.45;color:{theme.INK};"
            f"margin:0 0 .35rem .1rem'>{html.escape(line.text)}</div>",
            unsafe_allow_html=True,
        )


def _scene_default_label(scene: SceneDirection) -> str:
    bits = [
        scene.tempo,
        scene.loudness,
        scene.energy,
        scene.emotional_color,
    ]
    if scene.note:
        bits.append(scene.note)
    return " · ".join(b for b in bits if b)


def _cue_pills(cue: LineCue | None, scene: SceneDirection | None) -> str:
    if cue is None:
        return ""
    parts: list[str] = []
    default_tempo = scene.tempo if scene else "conversational"
    default_loud = scene.loudness if scene else "full"
    if cue.tempo and cue.tempo != default_tempo:
        parts.append(cue.tempo)
    if cue.loudness and cue.loudness != default_loud:
        parts.append(cue.loudness)
    if cue.pause_before_ms and cue.pause_before_ms >= 400:
        parts.append(f"pause {cue.pause_before_ms}ms")
    if cue.emphasis:
        parts.append(f"land “{cue.emphasis}”")
    if cue.breath_hold:
        parts.append("breath hold")
    if not parts and cue.instruction:
        parts.append("directed")
    if not parts:
        return ""
    return " " + "".join(theme.pill(p, "accent") for p in parts)


def _marketing_body(marketing) -> str:
    bullets = "".join(f" · {b}" for b in (marketing.hook_bullets or [])[:3])
    listener = marketing.target_listener or ""
    return (
        f"{marketing.logline}"
        f"{bullets}"
        + (f" — For: {listener}" if listener else "")
    )


def _render_manifest_audio(manifest: dict) -> None:
    audio_dir = Path(manifest.get("audio_dir", ""))
    if not manifest.get("audio_generated"):
        st.caption("Manifest built; audio was not rendered.")
        return
    for chunk in manifest.get("chunks", []):
        path = audio_dir / chunk["audio_file"] if chunk.get("audio_file") else None
        label = f"[{chunk.get('start', '')}] {chunk.get('character', '')}"
        st.caption(label)
        if path and path.exists():
            st.audio(str(path))


def plan_as_download_bytes(plan: ProducerPlan) -> bytes:
    return json.dumps(plan.model_dump(), indent=2).encode("utf-8")
