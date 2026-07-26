"""
Scene segmentation and the Scene DNA sensor.

Segmentation prefers explicit slug lines. When a draft has none — plenty of
audio-drama writers work in prose — boundaries are inferred from paragraph
structure and flagged as inferred so the UI can show the writer exactly what
it decided to measure.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from .prompts import SCENE_DNA_SYSTEM, SCENE_DNA_USER
from .schemas import Scene, SceneDNA

logger = logging.getLogger(__name__)

# A small model is deliberate. This runs once per scene and only has to
# measure, not reason, so paying for a frontier model here buys nothing.
SENSOR_MODEL = "gpt-4o-mini"
SENSOR_TEMPERATURE = 0.0

SCENE_HEADING_RE = re.compile(
    r"^\s*(INT\.|EXT\.|INT\./EXT\.|I/E\.|SCENE\s+\d+)", re.IGNORECASE
)

# Prose fallback targets roughly a minute of narration per inferred scene,
# matching the 150 wpm the rest of the pipeline assumes.
PROSE_TARGET_WORDS = 220
PROSE_MIN_WORDS = 90


class SceneFeatureError(Exception):
    """Raised when scene measurement cannot complete."""


def split_scenes(script_text: str) -> list[Scene]:
    """
    Break a continuation into scenes.

    Returns scenes with `inferred=True` when the split came from paragraph
    heuristics rather than slug lines.
    """
    text = (script_text or "").strip()
    if not text:
        raise ValueError("script_text must not be empty.")

    headed = _split_on_headings(text)
    if len(headed) >= 2:
        return headed
    if len(headed) == 1 and headed[0].word_count >= PROSE_MIN_WORDS * 2:
        # A single slug line over a long body still needs subdividing, or the
        # whole episode collapses into one data point.
        inner = _split_prose(headed[0].text)
        if len(inner) >= 2:
            return _renumber(inner, heading_prefix=headed[0].heading)
    if headed:
        return headed
    return _split_prose(text)


def _split_on_headings(text: str) -> list[Scene]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if SCENE_HEADING_RE.match(line)]
    if not starts:
        return []

    # Text before the first slug line must never be dropped. Long enough to
    # stand alone, it becomes a cold open; otherwise it is folded into the
    # first slug scene so no words leave the analysis.
    blocks: list[tuple[str, list[str]]] = []
    preamble_lines = lines[: starts[0]]
    preamble_words = len(" ".join(preamble_lines).split())
    carry: list[str] = []

    if preamble_words >= PROSE_MIN_WORDS:
        blocks.append(("COLD OPEN", preamble_lines))
    elif preamble_words:
        carry = preamble_lines

    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = carry + lines[start:end] if n == 0 else lines[start:end]
        blocks.append((lines[start].strip(), body))

    scenes: list[Scene] = []
    for heading, body in blocks:
        body_text = "\n".join(body).strip()
        if not body_text.split():
            continue
        scenes.append(
            Scene(index=len(scenes) + 1, heading=heading, text=body_text, inferred=False)
        )
    return scenes


def _split_prose(text: str) -> list[Scene]:
    """Group paragraphs into scene-sized blocks without splitting mid-paragraph."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    scenes: list[Scene] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        body = "\n\n".join(buffer)
        scenes.append(
            Scene(
                index=len(scenes) + 1,
                heading=f"INFERRED SEGMENT {len(scenes) + 1}",
                text=body,
                inferred=True,
            )
        )
        buffer.clear()

    for paragraph in paragraphs:
        buffer.append(paragraph)
        if len(" ".join(buffer).split()) >= PROSE_TARGET_WORDS:
            flush()

    # Fold a stub tail into the previous scene rather than reporting a
    # 30-word "scene" that would skew every per-scene average.
    if buffer:
        tail_words = len(" ".join(buffer).split())
        if scenes and tail_words < PROSE_MIN_WORDS:
            merged = scenes[-1].text + "\n\n" + "\n\n".join(buffer)
            scenes[-1] = Scene(
                index=scenes[-1].index,
                heading=scenes[-1].heading,
                text=merged,
                inferred=True,
            )
            buffer.clear()
        else:
            flush()

    return scenes


def _renumber(scenes: list[Scene], *, heading_prefix: str = "") -> list[Scene]:
    out: list[Scene] = []
    for n, scene in enumerate(scenes, 1):
        heading = scene.heading
        if heading_prefix and n == 1:
            heading = heading_prefix
        out.append(Scene(index=n, heading=heading, text=scene.text, inferred=scene.inferred))
    return out


def build_client() -> OpenAI:
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise SceneFeatureError("Missing required environment variable: OPENAI_API_KEY")
    try:
        return OpenAI(api_key=key)
    except OpenAIError as exc:
        raise SceneFeatureError(f"Failed to initialize OpenAI client: {exc}") from exc


def measure_scene(
    scene: Scene,
    total: int,
    *,
    client: OpenAI,
    model: str = SENSOR_MODEL,
    temperature: float = SENSOR_TEMPERATURE,
) -> SceneDNA:
    """Take one Scene DNA reading. Raises SceneFeatureError on failure."""
    heading_note = f" — {scene.heading}" if scene.heading else ""
    user = SCENE_DNA_USER.format(
        index=scene.index,
        total=total,
        heading_note=heading_note,
        scene_text=scene.text,
    )

    try:
        completion = client.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SCENE_DNA_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format=SceneDNA,
        )
    except OpenAIError as exc:
        raise SceneFeatureError(f"Scene DNA request failed for scene {scene.index}: {exc}") from exc

    message = completion.choices[0].message
    if message.refusal:
        raise SceneFeatureError(f"Sensor refused scene {scene.index}: {message.refusal}")

    dna = message.parsed
    if dna is None:
        raise SceneFeatureError(f"Sensor returned no parsable reading for scene {scene.index}.")
    return dna


def measure_scenes(
    scenes: list[Scene],
    *,
    client: OpenAI | None = None,
    model: str = SENSOR_MODEL,
    max_workers: int = 6,
) -> list[SceneDNA]:
    """
    Measure every scene, in parallel.

    Scenes are independent readings, so they fan out. Order is restored by
    index before returning; a single failed scene fails the batch, because a
    hole in the middle of a survival curve is worse than no curve.
    """
    if not scenes:
        raise ValueError("scenes must not be empty.")

    openai_client = client or build_client()
    total = len(scenes)
    results: dict[int, SceneDNA] = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as pool:
        futures = {
            pool.submit(measure_scene, scene, total, client=openai_client, model=model): scene
            for scene in scenes
        }
        for future, scene in futures.items():
            try:
                results[scene.index] = future.result()
            except Exception as exc:
                logger.error("Scene %d measurement failed: %s", scene.index, exc)
                errors.append(f"scene {scene.index}: {exc}")

    if errors:
        raise SceneFeatureError("Scene DNA measurement failed — " + "; ".join(errors))

    logger.info("Measured Scene DNA for %d scene(s).", total)
    return [results[scene.index] for scene in scenes]
