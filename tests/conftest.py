"""
Shared fixtures and fakes.

Nothing in this suite touches the network. The OpenAI and TTS clients are
replaced with objects that mimic only the surface the code actually calls, so
a change to that surface fails a test rather than silently costing money.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retention_engine.schemas import Scene, SceneDNA  # noqa: E402

DNA_DEFAULTS: dict = {
    "action_density": 0.4,
    "dialogue_ratio": 0.5,
    "exposition_ratio": 0.2,
    "internal_monologue_ratio": 0.1,
    "scene_tempo": 0.55,
    "tension": 0.5,
    "dread": 0.4,
    "romance": 0.0,
    "humor": 0.0,
    "warmth": 0.1,
    "anger": 0.1,
    "hope": 0.1,
    "sadness": 0.1,
    "wonder": 0.1,
    "stakes_level": 0.5,
    "stakes_type": "physical",
    "conflict_present": True,
    "conflict_type": "external_threat",
    "surprise_factor": 0.4,
    "complexity": 0.4,
    "character_development_present": False,
    "new_information_revealed": True,
    "cliffhanger_strength": 0.5,
    "violence_intensity": 0.2,
    "sexual_content": 0.0,
    "profanity_level": 0.0,
    "dark_themes": 0.4,
    "gore_level": 0.0,
    "worldbuilding_density": 0.2,
    "mystery_questions_opened": 1,
    "mystery_questions_answered": 0,
    "active_character_count": 2,
    "setting_change": False,
    "time_skip": False,
    "flashback": False,
    "tropes_present": ["empty_house"],
    "genre_alignment": "horror",
    "dominant_emotion": "dread",
}


def make_dna(**overrides) -> SceneDNA:
    """A valid SceneDNA with sensible defaults, overridable per test."""
    return SceneDNA(**{**DNA_DEFAULTS, **overrides})


def make_scene(index: int = 1, text: str = "", heading: str = "INT. ROOM - NIGHT") -> Scene:
    return Scene(index=index, heading=heading, text=text or "A room. Someone waits.", inferred=False)


class FakeParseResponse:
    """Mimics the object returned by client.chat.completions.parse."""

    def __init__(self, parsed, refusal: str | None = None) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=refusal))]


class FakeCompletions:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeCompletions received more calls than responses.")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeParseResponse(item)


class FakeOpenAI:
    """Stands in for openai.OpenAI across the structured-output call sites."""

    def __init__(self, responses: list) -> None:
        self._completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self._completions)
        self.audio = SimpleNamespace(speech=FakeSpeech())

    @property
    def calls(self) -> list[dict]:
        return self._completions.calls


class FakeSpeechResponse:
    def __init__(self, recorder: list, payload: dict) -> None:
        self._recorder = recorder
        self._payload = payload

    def stream_to_file(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"ID3fake-mp3-bytes")
        self._recorder.append({**self._payload, "path": str(path)})


class FakeSpeech:
    """Records TTS calls instead of making them. Can be told to reject params."""

    def __init__(self, reject_kwargs: tuple[str, ...] = ()) -> None:
        self.calls: list[dict] = []
        self.reject_kwargs = reject_kwargs

    def create(self, **kwargs):
        if any(key in kwargs for key in self.reject_kwargs):
            from openai import OpenAIError

            raise OpenAIError(f"unsupported parameter: {self.reject_kwargs}")
        return FakeSpeechResponse(self.calls, kwargs)


@pytest.fixture
def screenplay() -> str:
    return (
        "INT. BASEMENT - NIGHT\n\n"
        "A bulb swings. PRIYA folds a shirt that is still warm.\n\n"
        "PRIYA\nThat's just the pipes.\n\n"
        "Something drags across concrete behind her.\n\n"
        "INT. KITCHEN - CONTINUOUS\n\n"
        "DEV stands at the counter. The basement door is open.\n\n"
        "DEV\nPriya? You want tea?\n\n"
        "No answer. He takes the first step down.\n\n"
        "INT. HALLWAY - LATER\n\n"
        "The bed has not been slept in.\n\n"
        "NARRATOR (V.O.)\nHe had put the boy there ninety minutes earlier.\n"
    )


@pytest.fixture
def prose() -> str:
    body = (
        "The house on Wexler Street had been quiet for four days, which was "
        "longer than it had ever been quiet before, and Dev had started to "
        "notice the shape of the silence rather than the silence itself. "
    )
    return "\n\n".join([body * 3, body * 3, body * 3])
