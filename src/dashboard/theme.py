"""
Visual shell — Pocket FM × Apple-grade product UI.

Light, spacious, high-contrast chrome with Pocket FM magenta as the only
accent. Story text keeps a literary serif; everything else is a modern display
sans. The cream / terracotta look is gone on purpose: it read as a writing
app from 2014, and this product needs to feel like a studio tool.
"""

from __future__ import annotations

import streamlit as st

# Pocket FM brand red sampled from the official mark.
ACCENT = "#E61A4B"
ACCENT_DEEP = "#C4123D"
ACCENT_SOFT = "rgba(230, 26, 75, 0.08)"
ACCENT_RING = "rgba(230, 26, 75, 0.18)"

INK = "#1D1D1F"
MUTED = "#6E6E73"
FAINT = "#86868B"
PAPER = "#F5F5F7"
CARD = "#FFFFFF"
RULE = "rgba(0, 0, 0, 0.08)"
RULE_STRONG = "rgba(0, 0, 0, 0.12)"
# Opaque stand-in for Plotly and anywhere CSS rgba is awkward.
RULE_SOLID = "#E5E5EA"
SHADOW = "0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04)"
SHADOW_HOVER = "0 2px 8px rgba(0,0,0,0.06), 0 16px 40px rgba(0,0,0,0.08)"

RISK = "#D70015"
CAUTION = "#C93400"
CALM = "#248A3D"

# Expressive faces — not Inter / Roboto / Arial / system.
SANS = "'Sora', 'Plus Jakarta Sans', 'Avenir Next', sans-serif"
SERIF = "'Literata', 'Iowan Old Style', 'Palatino Linotype', Georgia, serif"
MONO = "'IBM Plex Mono', 'SF Mono', ui-monospace, monospace"

CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;1,7..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Sora:wght@400;500;600;700&display=swap');

  :root {{
    --pfm-accent: {ACCENT};
    --pfm-ink: {INK};
    --pfm-muted: {MUTED};
    --pfm-paper: {PAPER};
    --pfm-card: {CARD};
  }}

  html, body {{
    font-family: {SANS};
  }}
  /* Restore Material icon faces wherever Streamlit places them (sidebar nav,
     page titles, buttons). A global sans override had been printing the
     ligature names over the labels. */
  span[data-testid="stIconMaterial"],
  .material-symbols-rounded,
  .material-icons {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined",
      "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: "liga" !important;
    font-feature-settings: "liga" !important;
    -webkit-font-smoothing: antialiased;
  }}

  .stApp {{
    background:
      radial-gradient(1200px 600px at 85% -10%, rgba(230,26,75,0.07), transparent 55%),
      radial-gradient(900px 500px at -10% 20%, rgba(29,29,31,0.03), transparent 50%),
      {PAPER};
    color: {INK};
  }}

  .block-container {{
    padding-top: 1.6rem;
    padding-bottom: 5.5rem;
    max-width: 1120px;
  }}

  /* Room for the expand control when the panel is closed. */
  .stApp:has([data-testid="stExpandSidebarButton"]) .block-container {{
    padding-left: 3.75rem;
    padding-top: 2.4rem;
  }}

  @media (max-width: 900px) {{
    .block-container {{
      padding-top: 1.1rem;
      padding-left: 1rem;
      padding-right: 1rem;
    }}
    .stApp:has([data-testid="stExpandSidebarButton"]) .block-container {{
      padding-left: 3.5rem;
      padding-top: 2.2rem;
    }}
  }}

  /* Streamlit 1.60 puts the reopen control (stExpandSidebarButton) inside
     stToolbar. Hiding the toolbar left the sidebar with no way back. Keep the
     toolbar; only strip the bits of chrome we do not want. */
  #MainMenu, footer {{
    visibility: hidden;
  }}
  header[data-testid="stHeader"] {{
    background: transparent !important;
    box-shadow: none !important;
  }}
  header[data-testid="stHeader"] [data-testid="stToolbar"] {{
    display: flex !important;
    visibility: visible !important;
    background: transparent !important;
  }}
  /* Hide Deploy / status clutter; keep the expand-sidebar button. */
  header[data-testid="stHeader"] [data-testid="stToolbarActions"],
  header[data-testid="stHeader"] [data-testid="stStatusWidget"],
  header[data-testid="stHeader"] [data-testid="stAppDeployButton"],
  header[data-testid="stHeader"] .stAppDeployButton {{
    display: none !important;
  }}

  /* Native >> reopen control — icon only, no chrome. */
  [data-testid="stExpandSidebarButton"] {{
    visibility: visible !important;
    display: flex !important;
    z-index: 1000000 !important;
  }}
  [data-testid="stExpandSidebarButton"] button,
  button[data-testid="stExpandSidebarButton"] {{
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0 !important;
    color: {MUTED} !important;
    width: 2.4rem !important;
    height: 2.4rem !important;
  }}
  [data-testid="stExpandSidebarButton"] button:hover,
  button[data-testid="stExpandSidebarButton"]:hover {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: {ACCENT} !important;
  }}
  [data-testid="stExpandSidebarButton"] svg,
  [data-testid="stExpandSidebarButton"] span {{
    color: inherit !important;
    fill: currentColor !important;
  }}

  /* Legacy selectors, in case an older bundle is cached. */
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] {{
    visibility: visible !important;
    display: flex !important;
    z-index: 1000000 !important;
  }}
  [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="collapsedControl"] button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }}

  /* ---- sidebar ----------------------------------------------------- */
  section[data-testid="stSidebar"] {{
    background: rgba(255,255,255,0.82);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-right: 1px solid {RULE};
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.1rem;
  }}
  /* Style sidebar chrome, but never touch Material icon spans — those need
     the Material Symbols face. Forcing Sora onto them prints the ligature
     names (library_books, edit_note) over the nav labels. */
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded):not(.material-icons),
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] button,
  section[data-testid="stSidebar"] .pfm-product,
  section[data-testid="stSidebar"] .pfm-product-sub,
  section[data-testid="stSidebar"] .anu-meta {{
    font-family: {SANS};
  }}
  section[data-testid="stSidebar"] span[data-testid="stIconMaterial"],
  section[data-testid="stSidebar"] .material-symbols-rounded,
  section[data-testid="stSidebar"] .material-icons {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined",
      "Material Icons" !important;
  }}

  .pfm-sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid {RULE};
  }}
  .pfm-sidebar-brand-copy {{
    min-width: 0;
  }}
  .pfm-product {{
    font-family: {SANS};
    font-size: 1.05rem;
    font-weight: 650;
    letter-spacing: -0.03em;
    color: {INK};
    line-height: 1.15;
  }}
  .pfm-product-sub {{
    font-size: 0.7rem;
    color: {FAINT};
    letter-spacing: 0.02em;
    margin-top: 0.12rem;
  }}

  /* ---- type -------------------------------------------------------- */
  h1, h2, h3, h4, .pfm-hero-title, .pfm-project-title {{
    font-family: {SANS};
    color: {INK};
    letter-spacing: -0.035em;
    font-weight: 650;
  }}
  h1 {{ font-size: 2.1rem; line-height: 1.1; margin-bottom: 0.35rem; }}
  h2 {{ font-size: 1.35rem; }}
  h3 {{ font-size: 1.08rem; }}

  p, .stMarkdown, .stCaption {{
    color: {MUTED};
  }}

  /* ---- hero -------------------------------------------------------- */
  .pfm-hero {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1.5rem;
    margin: 0 0 1.6rem 0;
    padding: 1.5rem 1.6rem 1.55rem;
    border-radius: 22px;
    background:
      linear-gradient(135deg, rgba(230,26,75,0.09) 0%, rgba(255,255,255,0.95) 42%, #fff 100%);
    border: 1px solid {RULE};
    box-shadow: {SHADOW};
    position: relative;
    overflow: hidden;
  }}
  .pfm-hero::after {{
    content: "";
    position: absolute;
    right: -40px;
    top: -60px;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, rgba(230,26,75,0.16), transparent 68%);
    pointer-events: none;
  }}
  .pfm-hero-copy {{ position: relative; z-index: 1; max-width: 720px; }}
  .pfm-eyebrow {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.55rem;
  }}
  .pfm-hero-title {{
    font-size: clamp(1.7rem, 3.2vw, 2.35rem);
    margin: 0 0 0.4rem 0;
    color: {INK};
  }}
  .pfm-hero-sub {{
    margin: 0;
    font-size: 1.02rem;
    line-height: 1.55;
    color: {MUTED};
    max-width: 38rem;
  }}
  .pfm-hero-logo {{
    position: relative;
    z-index: 1;
    flex: 0 0 auto;
    box-shadow: {SHADOW};
    border-radius: 14px;
    overflow: hidden;
  }}

  @media (max-width: 720px) {{
    .pfm-hero {{
      flex-direction: column;
      align-items: flex-start;
      padding: 1.2rem 1.15rem;
    }}
    .pfm-hero-logo {{ align-self: flex-end; }}
  }}

  .pfm-section-label {{
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: {FAINT};
    margin: 1.4rem 0 0.65rem;
  }}

  /* ---- buttons ----------------------------------------------------- */
  .stButton > button {{
    font-family: {SANS};
    border-radius: 980px;
    border: 1px solid {RULE_STRONG};
    background: {CARD};
    color: {INK};
    font-weight: 550;
    font-size: 0.9rem;
    padding: 0.5rem 1.1rem;
    box-shadow: 0 1px 1px rgba(0,0,0,0.02);
    transition: transform .14s ease, box-shadow .14s ease, border-color .14s ease, background .14s ease, color .14s ease;
  }}
  .stButton > button:hover {{
    border-color: rgba(230,26,75,0.35);
    color: {ACCENT};
    background: #fff;
    box-shadow: {SHADOW};
    transform: translateY(-1px);
  }}
  .stButton > button[kind="primary"] {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #fff;
    box-shadow: 0 6px 18px rgba(230,26,75,0.28);
  }}
  .stButton > button[kind="primary"]:hover {{
    background: {ACCENT_DEEP};
    border-color: {ACCENT_DEEP};
    color: #fff;
    box-shadow: 0 8px 22px rgba(230,26,75,0.34);
  }}
  .stButton > button:disabled {{
    opacity: 0.45;
    transform: none !important;
    box-shadow: none !important;
  }}

  /* ---- inputs / composer ------------------------------------------- */
  .stTextArea textarea {{
    font-family: {SERIF} !important;
    font-size: 1.05rem !important;
    line-height: 1.78 !important;
    color: {INK} !important;
    background: {CARD} !important;
    border: 1px solid {RULE_STRONG} !important;
    border-radius: 18px !important;
    padding: 1.25rem 1.35rem !important;
    box-shadow: {SHADOW};
    transition: border-color .15s ease, box-shadow .15s ease;
  }}
  .stTextArea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 4px {ACCENT_RING}, {SHADOW} !important;
  }}

  .stTextInput input, .stSelectbox [data-baseweb="select"] > div,
  .stNumberInput input {{
    border-radius: 12px !important;
    border-color: {RULE_STRONG} !important;
    font-family: {SANS} !important;
  }}

  /* ---- cards / panels ---------------------------------------------- */
  .anu-card, .pfm-project, .pfm-empty, .pfm-callout {{
    background: {CARD};
    border: 1px solid {RULE};
    border-radius: 18px;
    box-shadow: {SHADOW};
  }}

  .anu-card {{
    padding: 1.2rem 1.35rem;
    margin-bottom: 0.9rem;
  }}
  .anu-card h4 {{
    margin: 0 0 0.35rem 0;
    font-size: 1.08rem;
    letter-spacing: -0.03em;
  }}
  .anu-card p {{
    margin: 0;
    color: {MUTED};
    font-size: 0.92rem;
    line-height: 1.55;
  }}

  .anu-meta {{
    color: {FAINT};
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
  }}

  .pfm-project {{
    padding: 1.15rem 1.2rem 1.1rem;
    margin-bottom: 0.75rem;
    min-height: 168px;
    display: flex;
    flex-direction: column;
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
  }}
  .pfm-project:hover {{
    transform: translateY(-2px);
    box-shadow: {SHADOW_HOVER};
    border-color: rgba(230,26,75,0.22);
  }}
  .pfm-project-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.85rem;
  }}
  .pfm-project-title {{
    margin: 0 0 0.4rem 0;
    font-size: 1.15rem;
    letter-spacing: -0.03em;
    color: {INK};
  }}
  .pfm-project-logline {{
    margin: 0;
    color: {MUTED};
    font-size: 0.9rem;
    line-height: 1.5;
    flex: 1;
  }}
  .pfm-project-meta {{
    margin-top: 0.9rem;
    font-size: 0.75rem;
    color: {FAINT};
    letter-spacing: 0.02em;
  }}

  .pfm-empty {{
    text-align: center;
    padding: 3rem 1.6rem 2.6rem;
    margin: 1rem 0 1.4rem;
  }}
  .pfm-empty-mark {{
    width: 56px;
    height: 56px;
    margin: 0 auto 1rem;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: {SHADOW};
  }}
  .pfm-empty h3 {{
    margin: 0 0 0.4rem;
    color: {INK};
  }}
  .pfm-empty p {{
    margin: 0 auto;
    max-width: 28rem;
    color: {MUTED};
    line-height: 1.55;
  }}

  .pfm-callout {{
    padding: 1rem 1.15rem;
    margin: 0.85rem 0 1rem;
    border-left: 3px solid {RULE_STRONG};
  }}
  .pfm-callout.calm {{ border-left-color: {CALM}; }}
  .pfm-callout.warn {{ border-left-color: {CAUTION}; }}
  .pfm-callout.accent {{ border-left-color: {ACCENT}; }}
  .pfm-callout-title {{
    font-weight: 650;
    color: {INK};
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
  }}
  .pfm-callout p {{
    margin: 0;
    color: {MUTED};
    font-size: 0.92rem;
    line-height: 1.5;
  }}

  /* ---- metrics / status -------------------------------------------- */
  .pfm-metric-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin: 0.4rem 0 1.2rem;
  }}
  .pfm-metric {{
    background: {CARD};
    border: 1px solid {RULE};
    border-radius: 16px;
    padding: 0.95rem 1.05rem;
    box-shadow: {SHADOW};
  }}
  .pfm-metric .lbl {{
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {FAINT};
    font-weight: 600;
  }}
  .pfm-metric .val {{
    font-size: 1.55rem;
    font-weight: 650;
    letter-spacing: -0.04em;
    color: {INK};
    margin: 0.2rem 0;
  }}
  .pfm-metric .hint {{
    font-size: 0.78rem;
    color: {MUTED};
  }}

  .pfm-status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.2rem 0 0.8rem;
  }}
  .pfm-status {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.28rem 0.65rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 550;
    background: rgba(0,0,0,0.03);
    color: {MUTED};
    border: 1px solid {RULE};
  }}
  .pfm-status .dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: {FAINT};
  }}
  .pfm-status.ok .dot {{ background: {CALM}; box-shadow: 0 0 0 3px rgba(36,138,61,0.15); }}
  .pfm-status.warn .dot {{ background: {CAUTION}; box-shadow: 0 0 0 3px rgba(201,52,0,0.15); }}
  .pfm-status.off .dot {{ background: {FAINT}; }}
  .pfm-status .detail {{
    color: {FAINT};
    font-weight: 450;
  }}

  /* ---- story text -------------------------------------------------- */
  .anu-story {{
    font-family: {SERIF};
    font-size: 1.02rem;
    line-height: 1.82;
    color: {INK};
    white-space: pre-wrap;
  }}
  .anu-quote {{
    font-family: {SERIF};
    font-style: italic;
    color: {MUTED};
    border-left: 2px solid {ACCENT};
    padding-left: 0.9rem;
    margin: 0.45rem 0;
    line-height: 1.65;
  }}

  /* ---- findings ---------------------------------------------------- */
  .anu-finding {{
    background: {CARD};
    border: 1px solid {RULE};
    border-left: 3px solid {RULE_STRONG};
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    box-shadow: {SHADOW};
  }}
  .anu-finding.high {{ border-left-color: {RISK}; }}
  .anu-finding.medium {{ border-left-color: {CAUTION}; }}
  .anu-finding.low {{ border-left-color: {FAINT}; }}
  .anu-finding .lbl {{
    font-size: 0.7rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: {MUTED};
    font-weight: 650;
  }}
  .anu-finding .what {{ color: {INK}; margin: 0.35rem 0; line-height: 1.55; }}
  .anu-finding .fix {{ color: {CALM}; font-size: 0.88rem; margin-top: 0.4rem; }}

  /* ---- pills ------------------------------------------------------- */
  .anu-pill {{
    display: inline-block;
    padding: 0.18rem 0.62rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.02em;
    border: 1px solid {RULE};
    color: {MUTED};
    background: rgba(0,0,0,0.02);
    margin-right: 0.3rem;
  }}
  .anu-pill.accent {{
    border-color: rgba(230,26,75,0.28);
    color: {ACCENT};
    background: {ACCENT_SOFT};
  }}
  .anu-pill.calm {{
    border-color: rgba(36,138,61,0.28);
    color: {CALM};
    background: rgba(36,138,61,0.08);
  }}
  .anu-pill.risk {{
    border-color: rgba(215,0,21,0.28);
    color: {RISK};
    background: rgba(215,0,21,0.07);
  }}

  /* ---- tabs / expanders / misc ------------------------------------- */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 0.4rem;
    border-bottom: 1px solid {RULE};
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: {SANS};
    font-weight: 550;
    color: {MUTED};
    padding: 0.55rem 0.2rem;
  }}
  .stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
  }}

  div[data-testid="stExpander"] details {{
    border: 1px solid {RULE};
    border-radius: 14px;
    background: {CARD};
    box-shadow: {SHADOW};
  }}
  div[data-testid="stExpander"] summary {{
    font-size: 0.9rem;
    color: {MUTED};
    font-family: {SANS};
  }}

  hr {{ border-color: {RULE}; }}

  div[data-testid="stMetricValue"] {{
    font-size: 1.55rem;
    color: {INK};
    letter-spacing: -0.03em;
  }}
  div[data-testid="stMetricLabel"] {{
    color: {MUTED};
    font-size: 0.78rem;
  }}

  /* Navigation polish */
  [data-testid="stSidebarNav"] a {{
    border-radius: 10px !important;
    font-family: {SANS} !important;
    font-weight: 500 !important;
  }}
  [data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: {ACCENT_SOFT} !important;
    color: {ACCENT} !important;
  }}

  /* Plotly frame */
  .js-plotly-plot .plotly {{
    border-radius: 16px;
    overflow: hidden;
  }}

  /* Subtle scrollbar */
  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-thumb {{
    background: rgba(0,0,0,0.18);
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
  }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def card(title: str, body: str = "", meta: str = "") -> str:
    meta_html = f'<div class="anu-meta">{meta}</div>' if meta else ""
    body_html = f"<p>{body}</p>" if body else ""
    return f'<div class="anu-card">{meta_html}<h4>{title}</h4>{body_html}</div>'


def pill(text: str, tone: str = "") -> str:
    return f'<span class="anu-pill {tone}">{text}</span>'


def finding_card(label: str, what: str, severity: str, detail: str = "", fix: str = "") -> str:
    detail_html = f'<div class="anu-quote">{detail}</div>' if detail else ""
    fix_html = f'<div class="fix">{fix}</div>' if fix else ""
    return (
        f'<div class="anu-finding {severity}">'
        f'<div class="lbl">{label}</div>'
        f'<div class="what">{what}</div>'
        f"{detail_html}{fix_html}</div>"
    )
