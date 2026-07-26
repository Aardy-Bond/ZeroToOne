"""
Pocket FM brand assets and chrome helpers.

The logo is embedded as a data URI so it renders anywhere Streamlit will show
HTML — sidebar, page headers, empty states — without depending on a static-file
server path that differs between local runs and Databricks Apps.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from dashboard import theme

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS / "pocket-fm.jpg"
MARK_PATH = ASSETS / "pocket-fm-mark.jpg"


@lru_cache(maxsize=4)
def _data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def logo_uri() -> str:
    return _data_uri(str(LOGO_PATH))


def mark_uri() -> str:
    path = MARK_PATH if MARK_PATH.exists() else LOGO_PATH
    return _data_uri(str(path))


def logo_img(height: int = 36, *, rounded: bool = True) -> str:
    radius = "10px" if rounded else "0"
    return (
        f'<img src="{logo_uri()}" alt="Pocket FM" '
        f'style="height:{height}px;width:auto;border-radius:{radius};'
        f'display:block;object-fit:cover"/>'
    )


def mark_img(height: int = 28) -> str:
    return (
        f'<img src="{mark_uri()}" alt="Pocket FM" '
        f'style="height:{height}px;width:{height}px;border-radius:9px;'
        f'display:block;object-fit:cover"/>'
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        f"""
        <div class="pfm-sidebar-brand">
          {logo_img(44)}
          <div class="pfm-sidebar-brand-copy">
            <div class="pfm-product">Anubhuti</div>
            <div class="pfm-product-sub">Writers Room · Pocket FM</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_hero(
    title: str,
    subtitle: str = "",
    *,
    eyebrow: str = "Pocket FM · Anubhuti",
    show_logo: bool = True,
) -> None:
    logo = (
        f'<div class="pfm-hero-logo">{logo_img(52)}</div>' if show_logo else ""
    )
    sub = f'<p class="pfm-hero-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="pfm-hero">
          <div class="pfm-hero-copy">
            <div class="pfm-eyebrow">{eyebrow}</div>
            <h1 class="pfm-hero-title">{title}</h1>
            {sub}
          </div>
          {logo}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(
        f'<div class="pfm-section-label">{text}</div>',
        unsafe_allow_html=True,
    )


def empty_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="pfm-empty">
          <div class="pfm-empty-mark">{mark_img(56)}</div>
          <h3>{title}</h3>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def project_tile(
    title: str,
    logline: str,
    meta: str,
    *,
    timelines: int = 1,
) -> str:
    badge = (
        f'<span class="anu-pill accent">{timelines} timelines</span>'
        if timelines > 1
        else f'<span class="anu-pill">Main timeline</span>'
    )
    return f"""
    <div class="pfm-project">
      <div class="pfm-project-top">
        {mark_img(22)}
        {badge}
      </div>
      <h3 class="pfm-project-title">{title}</h3>
      <p class="pfm-project-logline">{logline}</p>
      <div class="pfm-project-meta">{meta}</div>
    </div>
    """


def status_row(items: list[tuple[str, str, str]]) -> None:
    """items: (label, status ok|warn|off, detail)"""
    chips = []
    for label, status, detail in items:
        chips.append(
            f'<span class="pfm-status {status}">'
            f'<span class="dot"></span>{label}'
            f'<span class="detail">{detail}</span></span>'
        )
    st.markdown(
        f'<div class="pfm-status-row">{"".join(chips)}</div>',
        unsafe_allow_html=True,
    )


def callout(kind: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="pfm-callout {kind}">
          <div class="pfm-callout-title">{title}</div>
          <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, hint)"""
    cells = []
    for label, value, hint in items:
        cells.append(
            f'<div class="pfm-metric">'
            f'<div class="lbl">{label}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="hint">{hint}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="pfm-metric-strip">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


# Keep theme colours available through brand for pages that import one module.
INK = theme.INK
MUTED = theme.MUTED
ACCENT = theme.ACCENT
