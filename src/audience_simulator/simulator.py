"""
Synthetic Audience Simulator for Project Anubhuti.

Runs a draft script past three synthetic listener personas in parallel and
aggregates their minute-by-minute engagement into a drop-off heatmap.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.8
DROP_OFF_THRESHOLD = 5
SCORE_MIN = 1
SCORE_MAX = 10

# Narration pace used to map minute blocks onto script text. Downstream
# segmentation must use the same value or heatmap minutes will not line up
# with the passages they scored.
WORDS_PER_MINUTE = 150


class AudienceSimulatorError(Exception):
    """Raised when the simulator cannot produce a verdict."""


class MinuteRating(BaseModel):
    """One persona's engagement for a single minute block."""

    model_config = ConfigDict(extra="forbid")

    minute: int = Field(
        description="Minute block index, starting at 1 for the first minute."
    )
    engagement_score: int = Field(
        description="Engagement from 1 (checked out) to 10 (gripped)."
    )
    drop_off_reason: str = Field(
        description=(
            "Why this listener disengaged, required whenever engagement_score "
            "is below 5. Use an empty string when the score is 5 or higher."
        )
    )


class PersonaVerdict(BaseModel):
    """A persona's full reaction to the script."""

    model_config = ConfigDict(extra="forbid")

    overall_summary: str = Field(
        description="This listener's blunt verdict on the script, in their voice."
    )
    would_finish: bool = Field(
        description="Whether this listener would reach the end of the episode."
    )
    minute_ratings: list[MinuteRating] = Field(
        description="One entry per minute block, in ascending order."
    )


@dataclass(frozen=True)
class Persona:
    """A synthetic listener profile."""

    key: str
    name: str
    system_prompt: str


IMPATIENT_COMMUTER = Persona(
    key="impatient_commuter",
    name="The Impatient Commuter",
    system_prompt="""\
You are a synthetic test listener: THE IMPATIENT COMMUTER.

You listen on a crowded train with one earbud in, half your attention on your
stop. Your attention span is short and your thumb is always near the skip
button. You have no loyalty to this show and no patience for being warmed up.

What you punish:
- Exposition that explains the world instead of showing something happening.
- Scene-setting that runs longer than a few lines before a hook lands.
- Slow, atmospheric build with no concrete event attached to it.
- Any two-minute stretch where nothing changes. You are gone.

What holds you:
- An immediate question you want answered.
- Sharp, fast dialogue that carries conflict.
- Concrete events, not moods.

Score each minute honestly from your own impatient perspective. You are not
here to be fair to the writer. If you would have skipped, score it low and say
exactly what made you reach for the button.\
""",
)

HORROR_FAN = Persona(
    key="horror_fan",
    name="The Die-Hard Horror Fan",
    system_prompt="""\
You are a synthetic test listener: THE DIE-HARD HORROR FAN.

You have heard every audio horror anthology worth hearing. You listen alone, in
the dark, with good headphones, and you want to be genuinely unsettled. You are
patient with slow burns because you know dread has to be earned, but you are
merciless about payoff.

What you reward:
- Mounting dread and atmosphere that tightens rather than repeats.
- Restraint. Silence and implication over explanation.
- Sound design and physical detail that put you inside the space.
- A slow burn that pays off. You will sit through quiet if the quiet is loaded.

What you punish:
- Cheap jump scares substituting for tension.
- Over-explaining the threat, which kills it instantly.
- A build with no payoff, or a payoff that undercuts the build.
- Genre clichés you have heard a hundred times.

Score each minute from your own perspective. A quiet minute can score high if
the dread is compounding. A loud minute can score low if it is unearned.\
""",
)

CASUAL_LISTENER = Persona(
    key="casual_listener",
    name="The Casual Listener",
    system_prompt="""\
You are a synthetic test listener: THE CASUAL LISTENER.

You put this on while cooking or winding down. You are not tracking a mythology,
you did not take notes, and you will not rewind to catch something you missed.
You want to feel something and to always know who is talking and why.

What confuses and loses you:
- Names, places, factions, or history dropped without explanation.
- Timeline jumps or ambiguity about where and when you are.
- More than a couple of characters introduced at once.
- Emotional stakes you have to infer rather than feel.

What holds you:
- One clear character wanting one clear thing.
- Emotional clarity you can feel immediately.
- Plain, human dialogue.

Score each minute from your own perspective. When you lose the thread, say
precisely what confused you. Being lost is your main reason for disengaging.\
""",
)

PERSONAS: tuple[Persona, ...] = (
    IMPATIENT_COMMUTER,
    HORROR_FAN,
    CASUAL_LISTENER,
)

USER_TEMPLATE = """\
Listen to the following script as your persona and rate it minute by minute.

Treat roughly {words_per_minute} spoken words as one minute of runtime. Produce
one rating per minute block for the entire script, numbered from 1, with no
gaps. Give every minute an engagement score from {score_min} to {score_max}.
Whenever a score falls below {threshold}, fill in drop_off_reason with the
specific moment that lost you; otherwise leave drop_off_reason as an empty
string.

[SCRIPT]
{script_text}\
"""


@dataclass
class MinuteHeatmapEntry:
    """Aggregated engagement across all personas for one minute block."""

    minute: int
    average_score: float
    persona_scores: dict[str, int] = field(default_factory=dict)
    drop_off_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def is_drop_off(self) -> bool:
        return self.average_score < DROP_OFF_THRESHOLD

    def to_dict(self) -> dict:
        return {
            "minute": self.minute,
            "average_score": round(self.average_score, 2),
            "persona_scores": dict(self.persona_scores),
            "drop_off_reasons": dict(self.drop_off_reasons),
            "is_drop_off": self.is_drop_off,
        }


@dataclass
class AudienceReport:
    """Full simulator output: per-persona verdicts plus the merged heatmap."""

    verdicts: dict[str, PersonaVerdict]
    heatmap: list[MinuteHeatmapEntry]
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def overall_average(self) -> float:
        if not self.heatmap:
            return 0.0
        return sum(e.average_score for e in self.heatmap) / len(self.heatmap)

    @property
    def weakest_minutes(self) -> list[MinuteHeatmapEntry]:
        return sorted(self.heatmap, key=lambda e: e.average_score)

    def to_dict(self) -> dict:
        return {
            "overall_average": round(self.overall_average, 2),
            "personas": {
                key: {
                    "name": _persona_name(key),
                    "would_finish": verdict.would_finish,
                    "overall_summary": verdict.overall_summary,
                    "minute_ratings": [
                        r.model_dump() for r in verdict.minute_ratings
                    ],
                }
                for key, verdict in self.verdicts.items()
            },
            "heatmap": [entry.to_dict() for entry in self.heatmap],
            "failures": dict(self.failures),
        }


def _persona_name(key: str) -> str:
    for persona in PERSONAS:
        if persona.key == key:
            return persona.name
    return key


def _clamp_score(score: int) -> int:
    return max(SCORE_MIN, min(SCORE_MAX, score))


class AudienceSimulator:
    """Runs a script past synthetic listener personas and aggregates reactions."""

    def __init__(
        self,
        *,
        personas: tuple[Persona, ...] = PERSONAS,
        model: str = MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        client: OpenAI | None = None,
    ) -> None:
        self.personas = personas
        self.model = model
        self.temperature = temperature
        self.client = client or self._build_client()

    @staticmethod
    def _build_client() -> OpenAI:
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise AudienceSimulatorError(
                "Missing required environment variable: OPENAI_API_KEY"
            )
        try:
            return OpenAI(api_key=key)
        except OpenAIError as exc:
            raise AudienceSimulatorError(
                f"Failed to initialize OpenAI client: {exc}"
            ) from exc

    def _evaluate_persona(self, persona: Persona, script_text: str) -> PersonaVerdict:
        user_prompt = USER_TEMPLATE.format(
            score_min=SCORE_MIN,
            score_max=SCORE_MAX,
            threshold=DROP_OFF_THRESHOLD,
            words_per_minute=WORDS_PER_MINUTE,
            script_text=script_text,
        )

        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": persona.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=PersonaVerdict,
            )
        except OpenAIError as exc:
            raise AudienceSimulatorError(
                f"OpenAI request failed for persona '{persona.key}': {exc}"
            ) from exc

        message = completion.choices[0].message
        if message.refusal:
            raise AudienceSimulatorError(
                f"Model refused persona '{persona.key}': {message.refusal}"
            )

        verdict = message.parsed
        if verdict is None:
            raise AudienceSimulatorError(
                f"Model returned no parsable verdict for persona '{persona.key}'."
            )

        for rating in verdict.minute_ratings:
            rating.engagement_score = _clamp_score(rating.engagement_score)
            if rating.engagement_score >= DROP_OFF_THRESHOLD:
                rating.drop_off_reason = ""

        logger.info(
            "%s rated %d minute block(s).", persona.name, len(verdict.minute_ratings)
        )
        return verdict

    def build_heatmap(
        self, verdicts: dict[str, PersonaVerdict]
    ) -> list[MinuteHeatmapEntry]:
        """Average persona scores per minute block into a single time series."""
        buckets: dict[int, MinuteHeatmapEntry] = {}

        for persona_key, verdict in verdicts.items():
            for rating in verdict.minute_ratings:
                entry = buckets.setdefault(
                    rating.minute,
                    MinuteHeatmapEntry(minute=rating.minute, average_score=0.0),
                )
                entry.persona_scores[persona_key] = rating.engagement_score
                if rating.drop_off_reason.strip():
                    entry.drop_off_reasons[persona_key] = rating.drop_off_reason.strip()

        for entry in buckets.values():
            scores = list(entry.persona_scores.values())
            entry.average_score = sum(scores) / len(scores) if scores else 0.0

        return [buckets[minute] for minute in sorted(buckets)]

    def simulate_audience(self, script_text: str) -> AudienceReport:
        """
        Run all personas in parallel and aggregate their engagement.

        Raises AudienceSimulatorError only when every persona fails.
        """
        script_text = script_text.strip()
        if not script_text:
            raise ValueError("script_text must not be empty.")

        verdicts: dict[str, PersonaVerdict] = {}
        failures: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=len(self.personas)) as pool:
            futures = {
                pool.submit(self._evaluate_persona, persona, script_text): persona
                for persona in self.personas
            }
            for future, persona in futures.items():
                try:
                    verdicts[persona.key] = future.result()
                except Exception as exc:
                    logger.error("Persona '%s' failed: %s", persona.key, exc)
                    failures[persona.key] = str(exc)

        if not verdicts:
            raise AudienceSimulatorError(
                f"All personas failed to return a verdict: {failures}"
            )

        return AudienceReport(
            verdicts=verdicts,
            heatmap=self.build_heatmap(verdicts),
            failures=failures,
        )
