"""Offline tests for the genre / style rewrite engine."""

from __future__ import annotations

import pytest
from conftest import FakeOpenAI
from openai import OpenAIError

from writers_room.genre_rewrite import (
    GENRE_REWRITE_TARGETS,
    GenreChange,
    GenreRewriteError,
    GenreRewriteResult,
    rewrite_as_genre,
)

SCRIPT = (
    "INT. BASEMENT - NIGHT\n\n"
    "PRIYA folds a warm shirt.\n\n"
    "PRIYA\nThat's just the pipes.\n\n"
    "Something drags across the concrete behind her.\n"
)


def _sample_result(target: str = "horror") -> GenreRewriteResult:
    return GenreRewriteResult(
        rewritten_script=(
            "INT. BASEMENT - NIGHT\n\n"
            "PRIYA folds a shirt that still holds body heat.\n\n"
            "PRIYA\nThat's just the pipes.\n\n"
            "Something wet drags across the concrete behind her.\n"
        ),
        change_log=[
            GenreChange(
                aspect="imagery",
                change_made="Added tactile dread to the shirt and the drag sound.",
            )
        ],
        plot_preservation_note=(
            "Priya still dismisses the noise; something still approaches behind her."
        ),
        target_genre=target,
    )


def test_rewrite_as_genre_happy_path():
    client = FakeOpenAI([_sample_result("romance")])
    result = rewrite_as_genre(SCRIPT, "romance", client=client)

    assert "PRIYA" in result.rewritten_script
    assert result.target_genre == "romance"
    assert len(result.change_log) == 1
    assert result.change_log[0].aspect == "imagery"
    assert "plot" in result.plot_preservation_note.lower() or "Priya" in result.plot_preservation_note


def test_empty_script_raises():
    client = FakeOpenAI([_sample_result()])
    with pytest.raises(ValueError, match="must not be empty"):
        rewrite_as_genre("   ", "horror", client=client)
    assert client.calls == []


def test_unsupported_target_raises():
    client = FakeOpenAI([_sample_result()])
    with pytest.raises(ValueError, match="Unsupported genre rewrite target"):
        rewrite_as_genre(SCRIPT, "western", client=client)
    assert client.calls == []


def test_context_pack_appears_in_user_message():
    client = FakeOpenAI([_sample_result("thriller")])
    canon = "ESTABLISHED AND STILL TRUE\n- (world, part 0) Priya: lives on Wexler Street"

    rewrite_as_genre(
        SCRIPT,
        "thriller",
        context_pack=canon,
        client=client,
    )

    assert len(client.calls) == 1
    messages = client.calls[0]["messages"]
    user = next(m["content"] for m in messages if m["role"] == "user")
    assert "Wexler Street" in user
    assert "thriller" in user.lower()
    assert SCRIPT.strip() in user
    assert client.calls[0]["response_format"] is GenreRewriteResult


def test_empty_context_uses_fallback_notice():
    client = FakeOpenAI([_sample_result("comedy")])
    rewrite_as_genre(SCRIPT, "comedy", context_pack="", client=client)

    user = client.calls[0]["messages"][1]["content"]
    assert "No project canon was available" in user


def test_model_refusal_raises():
    client = FakeOpenAI([])
    # Inject a refusal response via FakeParseResponse path.
    from types import SimpleNamespace

    class RefusalClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []
            self.chat = SimpleNamespace(completions=self)

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(parsed=None, refusal="Cannot rewrite.")
                    )
                ]
            )

    refusal_client = RefusalClient()
    with pytest.raises(GenreRewriteError, match="refused"):
        rewrite_as_genre(SCRIPT, "anime", client=refusal_client)  # type: ignore[arg-type]


def test_empty_rewritten_script_raises():
    empty = GenreRewriteResult(
        rewritten_script="   ",
        change_log=[],
        plot_preservation_note="nothing",
        target_genre="horror",
    )
    client = FakeOpenAI([empty])
    with pytest.raises(GenreRewriteError, match="empty rewritten script"):
        rewrite_as_genre(SCRIPT, "horror", client=client)


def test_openai_error_surfaces_as_genre_rewrite_error():
    client = FakeOpenAI([OpenAIError("boom")])
    with pytest.raises(GenreRewriteError, match="OpenAI genre rewrite request failed"):
        rewrite_as_genre(SCRIPT, "horror", client=client)


def test_all_targets_accepted():
    for target in GENRE_REWRITE_TARGETS:
        client = FakeOpenAI([_sample_result(target)])
        result = rewrite_as_genre(SCRIPT, target.upper(), client=client)
        assert result.target_genre == target
