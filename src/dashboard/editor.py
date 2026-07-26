"""
The writing surface.

A screenplay editor rather than a text box. Three things earn their place here
and nothing else does:

- **Structure you can add without remembering the format.** Scene headings and
  character cues have a shape, and getting the shape wrong changes how the
  script is read downstream.
- **A live reading of what you have written.** Words, runtime, scenes, and who
  speaks, updated as you type, because runtime is the number an audio writer
  actually works to.
- **A preview of how the script will be parsed.** This is the important one.
  Any all-caps line becomes a speaker, so a stray `ON THE RADIO` silently
  becomes a character with a voice. The preview shows exactly what will be
  performed, which turns a bug you find in the audio into one you see while
  writing.

Streamlit cannot insert at the caret, so the structure buttons append. That is
honest about the constraint rather than pretending otherwise, and it matches
how these lines are written anyway: at the point you have reached.
"""

from __future__ import annotations

import re

import streamlit as st

from audio_engine.synthesizer import parse_script
from dashboard import brand, theme
from retention_engine.scene_features import split_scenes

WORDS_PER_MINUTE = 150

CUE = re.compile(r"^\s*([A-Z][A-Z0-9 .'\-]{1,30})(\s*\([^)]*\))?\s*$")
NOT_A_CHARACTER = {
    "CUT TO", "FADE IN", "FADE OUT", "FADE TO BLACK", "SMASH TO BLACK",
    "CONTINUED", "THE END", "MONTAGE", "INTERCUT", "BEAT",
}

TIME_OF_DAY = ["NIGHT", "DAY", "CONTINUOUS", "LATER", "DAWN", "DUSK", "MOMENTS LATER"]


def known_characters(segments) -> list[str]:
    """
    Who has spoken in this story already.

    Taken from the parsed cues of earlier parts rather than from the fact
    ledger, because the ledger holds subjects like "the basement door" and this
    list is for people with voices.
    """
    seen: dict[str, int] = {}
    for segment in segments:
        for chunk in parse_script(segment.text):
            name = chunk.character.strip()
            if name:
                seen[name] = seen.get(name, 0) + 1

    return [name for name, _ in sorted(seen.items(), key=lambda kv: -kv[1])]


def _append(text: str) -> None:
    current = st.session_state.draft.rstrip()
    st.session_state.draft = (current + "\n\n" + text) if current else text
    # The text area is keyed on this, so bumping it makes the widget adopt the
    # new value instead of restoring what the browser last sent.
    st.session_state.draft_version += 1


# Words that appear in descriptions and essentially never in a character name.
# "ON THE RADIO" has two; "MATRON SULOCHANA" and "DR. KADAM" have none.
FUNCTION_WORDS = {
    "A", "AN", "AND", "AS", "AT", "BACK", "BY", "FOR", "FROM", "IN", "INTO",
    "OF", "ON", "ONTO", "OR", "OVER", "THE", "THEN", "TO", "UNDER", "WITH",
}


def _suspicious_cues(chunks) -> list[str]:
    """
    Speakers that are really stray description.

    Counting lines does not work: a real character is allowed exactly one
    short line, and flagging that is worse than saying nothing. What actually
    separates the two is grammar. A slug the writer meant as action reads as a
    phrase and carries function words; a name does not.
    """
    odd: set[str] = set()

    for name in {c.character for c in chunks}:
        base = re.sub(r"\([^)]*\)", "", name).strip().upper()
        tokens = [t for t in re.split(r"\W+", base) if t]
        if not tokens:
            continue
        if base in NOT_A_CHARACTER or any(t in FUNCTION_WORDS for t in tokens):
            odd.add(name)

    return sorted(odd)


def render_toolbar(characters: list[str]) -> None:
    """Structure you can add without remembering the format."""
    st.markdown(
        f"<div class='anu-meta' style='margin:.2rem 0 .55rem'>Writing tools</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns([1, 1, 1, 1, 3])

    with columns[0]:
        with st.popover("Scene", width="stretch", help="Add a scene heading"):
            where = st.radio(
                "Inside or out", ["INT.", "EXT."], horizontal=True,
                label_visibility="collapsed", key="ed_where",
            )
            place = st.text_input("Place", placeholder="WARD FOUR", key="ed_place")
            when = st.selectbox("When", TIME_OF_DAY, key="ed_when")
            if st.button("Add scene heading", type="primary", width="stretch"):
                name = (place or "SOMEWHERE").strip().upper()
                _append(f"{where} {name} - {when}")
                st.rerun()

    with columns[1]:
        with st.popover("Character", width="stretch", help="Add a line of dialogue"):
            if characters:
                st.caption("Already in this story")
                picked = st.radio(
                    "Who speaks",
                    characters[:8] + ["Someone new"],
                    label_visibility="collapsed",
                    key="ed_who",
                )
            else:
                picked = "Someone new"

            name = picked
            if picked == "Someone new":
                name = st.text_input("Name", placeholder="REVATI", key="ed_newwho")

            aside = st.text_input(
                "Direction", placeholder="quietly", key="ed_paren",
                help="Optional. Appears in brackets under the name.",
            )
            line = st.text_area("They say", height=90, key="ed_line")

            if st.button("Add dialogue", type="primary", width="stretch"):
                if name.strip():
                    block = name.strip().upper()
                    if aside.strip():
                        block += f"\n({aside.strip()})"
                    block += f"\n{line.strip() or '...'}"
                    _append(block)
                    st.rerun()

    with columns[2]:
        with st.popover("Narrator", width="stretch", help="Add narration"):
            st.caption(
                "Narration is the only way description reaches the audio. "
                "Action lines are not performed."
            )
            line = st.text_area("Narration", height=110, key="ed_narr")
            if st.button("Add narration", type="primary", width="stretch"):
                _append(f"NARRATOR (V.O.)\n{line.strip() or '...'}")
                st.rerun()

    with columns[3]:
        with st.popover("Help", width="stretch", help="How the format is read"):
            st.markdown(
                "**Scene heading** — a line starting `INT.` or `EXT.` starts a "
                "new scene.\n\n"
                "**Character cue** — a name in capitals on its own line. The "
                "lines under it are spoken.\n\n"
                "**Action** — ordinary sentences. Read by nobody: it becomes "
                "sound and atmosphere.\n\n"
                "**Narration** — `NARRATOR (V.O.)` when you want description "
                "actually heard.\n\n"
                "Careful with capitals. An all-caps line with a sentence "
                "beneath it becomes a speaker saying that sentence, so write "
                "`ON THE RADIO` as ordinary action instead."
            )


def render_stats(draft: str) -> tuple[list, list]:
    """The live reading. Returns parsed chunks and scenes for reuse."""
    words = len(draft.split())
    chunks = parse_script(draft) if draft.strip() else []
    scenes = split_scenes(draft) if draft.strip() else []
    spoken = sum(c.word_count for c in chunks)
    speakers = {c.character for c in chunks}

    minutes = spoken / WORDS_PER_MINUTE if spoken else 0
    runtime = f"{int(minutes)}:{int((minutes % 1) * 60):02d}"

    brand.metric_strip(
        [
            ("Words", f"{words:,}", "on the page"),
            ("Scenes", str(len(scenes)), "detected"),
            ("Voices", str(len(speakers)), "speaking"),
            ("Spoken", runtime, "at ~150 wpm"),
        ]
    )

    return chunks, scenes


def render_reading(draft: str, chunks, scenes) -> None:
    """How the script will actually be read, and what looks wrong."""
    if not draft.strip():
        return

    odd = _suspicious_cues(chunks)

    label = "How this will be read"
    if odd:
        label += f"  ·  {len(odd)} to check"

    with st.expander(label):
        if odd:
            st.warning(
                "These read as descriptions but will be performed as speech: "
                + ", ".join(f"**{name}**" for name in odd)
                + ". A capitalised line with a sentence under it becomes a "
                "character. Write them as ordinary action instead."
            )

        left, right = st.columns([1, 1.4], gap="medium")

        with left:
            st.markdown("**Scenes**")
            if not scenes:
                st.caption("No scene headings yet.")
            for scene in scenes:
                inferred = " (inferred)" if scene.inferred else ""
                st.markdown(
                    f"<div style='font-size:.82rem;margin:.2rem 0'>"
                    f"<span style='color:{theme.FAINT}'>{scene.index}</span> "
                    f"{scene.heading}{inferred}</div>",
                    unsafe_allow_html=True,
                )

        with right:
            st.markdown("**What gets performed**")
            if not chunks:
                st.caption(
                    "Nothing spoken yet. Add a character cue or narration, or "
                    "this part will be silent."
                )
            for chunk in chunks[:14]:
                text = chunk.text if len(chunk.text) < 110 else chunk.text[:110] + "..."
                tone = theme.ACCENT if chunk.kind == "narration" else theme.INK
                st.markdown(
                    f"<div style='font-size:.82rem;margin:.25rem 0'>"
                    f"<strong style='color:{tone}'>{chunk.character}</strong> "
                    f"<span style='color:{theme.MUTED}'>{text}</span></div>",
                    unsafe_allow_html=True,
                )
            if len(chunks) > 14:
                st.caption(f"...and {len(chunks) - 14} more.")


def render(characters: list[str], *, height: int = 460) -> str:
    """The whole writing surface. Returns the current draft."""
    render_toolbar(characters)

    draft = st.text_area(
        "Draft",
        value=st.session_state.draft,
        height=height,
        key=f"draft_{st.session_state.draft_version}",
        placeholder=(
            "INT. WARD FOUR - NIGHT\n\n"
            "Write here. Screenplay or prose both work.\n\n"
            "REVATI\n"
            "A name in capitals on its own line becomes a speaker."
        ),
        label_visibility="collapsed",
    )
    st.session_state.draft = draft

    chunks, scenes = render_stats(draft)
    render_reading(draft, chunks, scenes)

    return draft
