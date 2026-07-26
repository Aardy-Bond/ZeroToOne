"""
Audio synthesis and production export for Project Anubhuti.

Parses an approved script into spoken chunks, renders scratch audio with
OpenAI TTS, and compiles a production manifest that pins every foley trigger
to the audio chunk it fires over.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

if TYPE_CHECKING:
    from writers_room.schemas import FoleyTrigger

logger = logging.getLogger(__name__)

TTS_MODEL = "tts-1-hd"

# Steerable model used for the retention-directed preview. It accepts an
# `instructions` string describing delivery, which tts-1-hd ignores. Community
# reports say the rolling `gpt-4o-mini-tts` alias follows instructions less
# reliably than the 2025-03-20 snapshot, so pin DIRECTED_TTS_MODEL to that
# snapshot if delivery direction stops landing.
DIRECTED_TTS_MODEL = "gpt-4o-mini-tts"
INSTRUCTION_CAPABLE_PREFIXES = ("gpt-4o-mini-tts", "gpt-4o-audio")

NARRATOR_KEY = "NARRATOR"

# Must match the audience simulator's pacing assumption so manifest timestamps
# line up with the minute blocks the personas scored.
WORDS_PER_MINUTE = 150
WORDS_PER_SECOND = WORDS_PER_MINUTE / 60.0

# Beat of silence inserted between spoken chunks when laying out the timeline.
INTER_CHUNK_PAUSE_SECONDS = 0.6

# tts-1-hd voice pool. Narrator is pinned; the rest are assigned in order of
# first appearance so a given script always renders with the same casting.
NARRATOR_VOICE = "onyx"
VOICE_POOL: tuple[str, ...] = ("nova", "echo", "shimmer", "fable", "alloy")

SCENE_HEADING_RE = re.compile(r"^(INT\.|EXT\.|INT\./EXT\.|I/E\.)", re.IGNORECASE)
TRANSITION_RE = re.compile(
    r"^(CUT TO:|SMASH CUT.*|SMASH TO.*|FADE (IN|OUT).*|DISSOLVE TO:|MATCH CUT.*)$",
    re.IGNORECASE,
)
CHARACTER_CUE_RE = re.compile(r"^([A-Z][A-Z0-9 .'\-]{0,30})(\s*\([^)]*\))*\s*$")
PARENTHETICAL_RE = re.compile(r"^\(.*\)$")
TIMESTAMP_RE = re.compile(r"(\d{1,2}):([0-5]?\d)")


class SynthesisError(Exception):
    """Raised when audio synthesis or manifest export fails."""


@dataclass
class ScriptChunk:
    """One spoken unit of the script, with its place on the timeline."""

    index: int
    kind: str  # "dialogue" or "narration"
    character: str
    text: str
    voice: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    audio_file: str = ""
    foley: list[dict] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict:
        data = asdict(self)
        data["start"] = format_timestamp(self.start_seconds)
        data["end"] = format_timestamp(self.end_seconds)
        data["duration_seconds"] = round(self.duration_seconds, 2)
        data["start_seconds"] = round(self.start_seconds, 2)
        data["end_seconds"] = round(self.end_seconds, 2)
        data["word_count"] = self.word_count
        return data


def format_timestamp(seconds: float) -> str:
    """Render seconds as MM:SS."""
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_timestamp(value: str) -> float | None:
    """
    Read MM:SS out of a foley trigger timestamp.

    Returns None for beat references like 'on the bulb dying', which the panel
    is allowed to produce instead of a clock time.
    """
    match = TIMESTAMP_RE.search(value or "")
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _clean_character(raw: str) -> str:
    """Strip (V.O.), (O.S.), (CONT'D) so one character maps to one voice."""
    name = re.sub(r"\([^)]*\)", "", raw).strip()
    return re.sub(r"\s+", " ", name).upper()


def parse_script(script_text: str) -> list[ScriptChunk]:
    """
    Split a screenplay into spoken chunks.

    Action lines and scene headings are deliberately excluded: in an audio
    drama they are not performed, they become foley and ambience. Narration
    reaches the mix through an explicit NARRATOR (V.O.) cue.
    """
    chunks: list[ScriptChunk] = []
    lines = script_text.splitlines()

    current_character: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current_character, buffer
        if current_character and buffer:
            text = " ".join(line.strip() for line in buffer).strip()
            if text:
                kind = "narration" if NARRATOR_KEY in current_character else "dialogue"
                chunks.append(
                    ScriptChunk(
                        index=len(chunks) + 1,
                        kind=kind,
                        character=current_character,
                        text=text,
                    )
                )
        current_character = None
        buffer = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush()
            continue

        if SCENE_HEADING_RE.match(line) or TRANSITION_RE.match(line):
            flush()
            continue

        if current_character is None:
            if CHARACTER_CUE_RE.match(line) and not line.endswith("."):
                current_character = _clean_character(line)
            continue

        if PARENTHETICAL_RE.match(line):
            continue

        buffer.append(line)

    flush()
    return chunks


def assign_voices(
    chunks: Sequence[ScriptChunk],
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Cast each character to a TTS voice, stable across runs."""
    overrides = {k.upper(): v for k, v in (overrides or {}).items()}
    casting: dict[str, str] = {}
    pool_index = 0

    for chunk in chunks:
        name = chunk.character
        if name in casting:
            continue
        if name in overrides:
            casting[name] = overrides[name]
        elif NARRATOR_KEY in name:
            casting[name] = NARRATOR_VOICE
        else:
            casting[name] = VOICE_POOL[pool_index % len(VOICE_POOL)]
            pool_index += 1

    return casting


def lay_out_timeline(
    chunks: Sequence[ScriptChunk],
    *,
    words_per_second: float = WORDS_PER_SECOND,
    pause_seconds: float = INTER_CHUNK_PAUSE_SECONDS,
) -> float:
    """
    Assign start and end times to every chunk, in place.

    Durations are estimated from word count rather than measured from the
    rendered MP3, so the manifest stays consistent whether or not audio was
    actually generated. Re-time against real durations before a final mix.
    """
    cursor = 0.0
    for chunk in chunks:
        duration = max(chunk.word_count / words_per_second, 0.8)
        chunk.start_seconds = cursor
        chunk.end_seconds = cursor + duration
        cursor = chunk.end_seconds + pause_seconds
    return max(cursor - pause_seconds, 0.0)


def map_foley_to_chunks(
    chunks: Sequence[ScriptChunk],
    foley_triggers: Iterable[FoleyTrigger],
) -> list[dict]:
    """
    Pin each foley cue to the chunk playing when it fires.

    Cues with beat references instead of clock times, or that land past the
    final chunk, are returned as unassigned rather than silently dropped.
    """
    unassigned: list[dict] = []

    for trigger in foley_triggers:
        cue = {
            "timestamp": trigger.timestamp,
            "sound_effect": trigger.sound_effect,
        }
        at = parse_timestamp(trigger.timestamp)

        if at is None:
            unassigned.append({**cue, "reason": "no MM:SS timestamp (beat reference)"})
            continue

        placed = False
        for chunk in chunks:
            if chunk.start_seconds <= at < chunk.end_seconds + INTER_CHUNK_PAUSE_SECONDS:
                chunk.foley.append({**cue, "at_seconds": at})
                placed = True
                break

        if not placed:
            unassigned.append({**cue, "reason": "timestamp falls outside spoken audio"})

    return unassigned


class AudioSynthesizer:
    """Renders scratch audio and compiles the production manifest."""

    def __init__(
        self,
        *,
        output_dir: str | Path = "output",
        model: str = TTS_MODEL,
        client: OpenAI | None = None,
        voice_overrides: dict[str, str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "scratch_audio"
        self.model = model
        self.voice_overrides = voice_overrides or {}
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    @staticmethod
    def _build_client() -> OpenAI:
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise SynthesisError("Missing required environment variable: OPENAI_API_KEY")
        try:
            return OpenAI(api_key=key)
        except OpenAIError as exc:
            raise SynthesisError(f"Failed to initialize OpenAI client: {exc}") from exc

    def _render_chunk(
        self,
        chunk: ScriptChunk,
        *,
        instructions: str = "",
        speed: float | None = None,
        model: str | None = None,
        prefix: str = "",
    ) -> str:
        """
        Render one chunk to MP3.

        `instructions` and `speed` steer delivery on models that support them.
        If a directed call is rejected — an unsupported parameter, an
        unavailable model — the call is retried in its plainest form so a
        preview still renders instead of failing the whole export.
        """
        slug = re.sub(r"[^a-z0-9]+", "_", chunk.character.lower()).strip("_")
        filename = f"{prefix}{chunk.index:03d}_{slug or 'voice'}.mp3"
        path = self.audio_dir / filename
        tts_model = model or self.model

        payload: dict = {
            "model": tts_model,
            "voice": chunk.voice,
            "input": chunk.text,
        }
        if instructions and tts_model.startswith(INSTRUCTION_CAPABLE_PREFIXES):
            payload["instructions"] = instructions
        if speed is not None:
            payload["speed"] = round(max(0.25, min(4.0, speed)), 3)

        try:
            response = self.client.audio.speech.create(**payload)
            response.stream_to_file(path)
        except OpenAIError as exc:
            if len(payload) == 3:
                raise SynthesisError(
                    f"TTS failed for chunk {chunk.index} ({chunk.character}): {exc}"
                ) from exc

            logger.warning(
                "Directed TTS failed for chunk %d (%s); retrying undirected on %s.",
                chunk.index, exc, TTS_MODEL,
            )
            try:
                response = self.client.audio.speech.create(
                    model=TTS_MODEL, voice=chunk.voice, input=chunk.text
                )
                response.stream_to_file(path)
            except OpenAIError as fallback_exc:
                raise SynthesisError(
                    f"TTS failed for chunk {chunk.index} ({chunk.character}): {fallback_exc}"
                ) from fallback_exc

        logger.info("Rendered %s (%s, %s)", filename, chunk.character, chunk.voice)
        return filename

    def synthesize(
        self,
        script_text: str,
        foley_triggers: Iterable[FoleyTrigger],
        *,
        generate_audio: bool = True,
        manifest_name: str = "production_manifest.json",
    ) -> dict:
        """
        Run the full export: parse, cast, time, render, and write the manifest.

        Set `generate_audio=False` to produce the manifest and cue sheet
        without spending TTS credits.
        """
        script_text = script_text.strip()
        if not script_text:
            raise ValueError("script_text must not be empty.")

        chunks = parse_script(script_text)
        if not chunks:
            raise SynthesisError(
                "No spoken lines found. Expect screenplay format with ALL-CAPS "
                "character cues above dialogue."
            )

        casting = assign_voices(chunks, self.voice_overrides)
        for chunk in chunks:
            chunk.voice = casting[chunk.character]

        total_runtime = lay_out_timeline(chunks)
        triggers = list(foley_triggers)
        unassigned = map_foley_to_chunks(chunks, triggers)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if generate_audio:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            for chunk in chunks:
                chunk.audio_file = self._render_chunk(chunk)
        else:
            logger.info("generate_audio=False; skipping %d TTS call(s).", len(chunks))

        manifest = {
            "project": "Project Anubhuti",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tts_model": self.model if generate_audio else None,
            "audio_generated": generate_audio,
            "audio_dir": str(self.audio_dir),
            "total_runtime_seconds": round(total_runtime, 2),
            "total_runtime": format_timestamp(total_runtime),
            "casting": casting,
            "chunk_count": len(chunks),
            "foley_cue_count": len(triggers),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "unassigned_foley": unassigned,
        }

        manifest_path = self.output_dir / manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Wrote manifest to %s", manifest_path)

        cue_sheet_path = self.output_dir / "cue_sheet.txt"
        cue_sheet_path.write_text(build_cue_sheet(chunks, unassigned), encoding="utf-8")
        logger.info("Wrote cue sheet to %s", cue_sheet_path)

        manifest["manifest_path"] = str(manifest_path)
        manifest["cue_sheet_path"] = str(cue_sheet_path)
        return manifest

    def synthesize_directed_preview(
        self,
        preview_text: str,
        foley_triggers: Iterable[FoleyTrigger],
        directing_sheet: Sequence,
        *,
        generate_audio: bool = True,
        model: str = DIRECTED_TTS_MODEL,
        manifest_name: str = "preview_manifest.json",
        forecast_summary: dict | None = None,
    ) -> dict:
        """
        Render a retention-directed preview of the approved ending.

        Behaves like `synthesize` but renders through a steerable TTS model,
        passing each chunk the delivery instruction and speech rate from the
        directing sheet, and embeds that sheet in the manifest. Output goes to
        a separate `preview_audio/` directory and its own manifest file so the
        main production export is never overwritten.
        """
        return self.synthesize_directed_script(
            preview_text,
            foley_triggers,
            directing_sheet,
            generate_audio=generate_audio,
            model=model,
            manifest_name=manifest_name,
            forecast_summary=forecast_summary,
            export_type="retention_directed_preview",
            audio_subdir="preview_audio",
            cue_sheet_name="preview_cue_sheet.txt",
            file_prefix="preview_",
            chunks=None,
        )

    def synthesize_directed_script(
        self,
        script_text: str,
        foley_triggers: Iterable[FoleyTrigger],
        directing_sheet: Sequence,
        *,
        generate_audio: bool = True,
        model: str = DIRECTED_TTS_MODEL,
        manifest_name: str = "producer_manifest.json",
        forecast_summary: dict | None = None,
        export_type: str = "ai_producer",
        audio_subdir: str = "producer_audio",
        cue_sheet_name: str = "producer_cue_sheet.txt",
        file_prefix: str = "producer_",
        chunks: Sequence[ScriptChunk] | None = None,
        voice_overrides: dict[str, str] | None = None,
    ) -> dict:
        """
        Render a full directed script (AI Producer or ending preview).

        When `chunks` is provided (e.g. prose-as-narration lines from the
        producer), those are used instead of re-parsing the screenplay.
        """
        script_text = (script_text or "").strip()
        if not script_text and not chunks:
            raise ValueError("script_text must not be empty.")

        if chunks is None:
            parsed = parse_script(script_text)
            if not parsed:
                raise SynthesisError(
                    "No spoken lines found. Expect screenplay format with "
                    "ALL-CAPS character cues above dialogue, or pass producer chunks."
                )
            chunks = parsed
        else:
            chunks = list(chunks)

        overrides = {**(self.voice_overrides or {}), **(voice_overrides or {})}
        casting = assign_voices(chunks, overrides)
        for chunk in chunks:
            chunk.voice = casting[chunk.character]

        total_runtime = lay_out_timeline(chunks)
        triggers = list(foley_triggers)
        unassigned = map_foley_to_chunks(chunks, triggers)

        directions = {entry.chunk_index: entry for entry in directing_sheet}

        audio_dir = self.output_dir / audio_subdir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if generate_audio:
            audio_dir.mkdir(parents=True, exist_ok=True)
            original_audio_dir = self.audio_dir
            self.audio_dir = audio_dir
            try:
                for chunk in chunks:
                    entry = directions.get(chunk.index)
                    chunk.audio_file = self._render_chunk(
                        chunk,
                        instructions=getattr(entry, "instruction", "") if entry else "",
                        speed=getattr(entry, "target_speed", None) if entry else None,
                        model=model,
                        prefix=file_prefix,
                    )
            finally:
                self.audio_dir = original_audio_dir
        else:
            logger.info(
                "generate_audio=False; skipping %d directed TTS call(s).", len(chunks)
            )

        manifest = {
            "project": "Project Anubhuti",
            "export_type": export_type,
            "source": "ai_producer" if export_type == "ai_producer" else export_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tts_model": model if generate_audio else None,
            "audio_generated": generate_audio,
            "audio_dir": str(audio_dir),
            "total_runtime_seconds": round(total_runtime, 2),
            "total_runtime": format_timestamp(total_runtime),
            "casting": casting,
            "chunk_count": len(chunks),
            "foley_cue_count": len(triggers),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "unassigned_foley": unassigned,
            "retention_directing_sheet": [
                entry.model_dump() if hasattr(entry, "model_dump") else dict(entry)
                for entry in directing_sheet
            ],
            "forecast_summary": forecast_summary or {},
        }

        manifest_path = self.output_dir / manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        cue_sheet_path = self.output_dir / cue_sheet_name
        cue_sheet_path.write_text(
            build_directed_cue_sheet(chunks, directing_sheet, unassigned),
            encoding="utf-8",
        )

        logger.info("Wrote directed manifest to %s", manifest_path)
        manifest["manifest_path"] = str(manifest_path)
        manifest["cue_sheet_path"] = str(cue_sheet_path)
        return manifest


def build_cue_sheet(
    chunks: Sequence[ScriptChunk],
    unassigned: Sequence[dict] = (),
) -> str:
    """Render the human-readable cue sheet the sound editor works from."""
    lines = ["PROJECT ANUBHUTI — PRODUCTION CUE SHEET", "=" * 60, ""]

    for chunk in chunks:
        stamp = format_timestamp(chunk.start_seconds)
        preview = chunk.text if len(chunk.text) <= 70 else chunk.text[:67] + "..."
        lines.append(f"[{stamp}] {chunk.character} ({chunk.voice}): {preview}")
        for cue in chunk.foley:
            lines.append(
                f"    [{format_timestamp(cue['at_seconds'])} - FX: {cue['sound_effect']}]"
            )

    if unassigned:
        lines += ["", "UNPLACED CUES (assign manually)", "-" * 60]
        for cue in unassigned:
            lines.append(
                f"    [{cue['timestamp']} - FX: {cue['sound_effect']}]  ({cue['reason']})"
            )

    return "\n".join(lines) + "\n"


def build_directed_cue_sheet(
    chunks: Sequence[ScriptChunk],
    directing_sheet: Sequence,
    unassigned: Sequence[dict] = (),
) -> str:
    """Cue sheet for the directed preview, with performance notes inline."""
    directions = {getattr(e, "chunk_index", -1): e for e in directing_sheet}

    lines = [
        "PROJECT ANUBHUTI — RETENTION-DIRECTED PREVIEW CUE SHEET",
        "=" * 64,
        "Delivery direction derived from the Engagement Survival Forecast.",
        "Simulated pre-release forecast — not calibrated with listener data.",
        "",
    ]

    for chunk in chunks:
        stamp = format_timestamp(chunk.start_seconds)
        preview = chunk.text if len(chunk.text) <= 70 else chunk.text[:67] + "..."
        lines.append(f"[{stamp}] {chunk.character} ({chunk.voice}): {preview}")

        entry = directions.get(chunk.index)
        if entry is not None:
            lines.append(
                f"    ROLE: {entry.narrative_role} | risk: {entry.engagement_risk} "
                f"| emotion: {entry.dominant_emotion}"
            )
            lines.append(
                f"    DELIVERY: {entry.delivery_tempo} @ {entry.target_speed:.2f}x "
                f"| pause before reveal {entry.pause_before_reveal_ms}ms"
            )
            lines.append(f"    NOTE: {entry.instruction}")
            if entry.foley:
                lines.append(f"    FOLEY: {'; '.join(entry.foley)}")
            if entry.hook_role:
                lines.append(
                    f"    HOOK: {entry.hook_role} | Unlock Pull Index "
                    f"{entry.unlock_pull_index:.0f}/100"
                )

        for cue in chunk.foley:
            lines.append(
                f"    [{format_timestamp(cue['at_seconds'])} - FX: {cue['sound_effect']}]"
            )
        lines.append("")

    if unassigned:
        lines += ["UNPLACED CUES (assign manually)", "-" * 64]
        for cue in unassigned:
            lines.append(
                f"    [{cue['timestamp']} - FX: {cue['sound_effect']}]  ({cue['reason']})"
            )

    return "\n".join(lines) + "\n"
