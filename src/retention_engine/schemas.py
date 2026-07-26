"""
Schemas for the Engagement Survival Forecast.

Every model that crosses the OpenAI structured-output boundary forbids extra
keys and declares all fields required, which is what strict mode demands.
Numeric ranges are enforced by validators rather than JSON Schema `minimum`
and `maximum`, because strict mode rejects those keywords. Clamping in a
validator also means a sensor that returns 1.4 gets corrected instead of
failing the whole analysis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Shown wherever a forecast number is displayed. The whole product depends on
# this claim staying honest, so it lives in one place and is imported, never
# retyped.
FORECAST_DISCLAIMER = (
    "Simulated pre-release forecast — informed by narrative-quality and "
    "emotional signals. It is not calibrated with live listener, retention, "
    "unlock, or purchase data."
)

QUALITY_PRIOR_DISCLAIMER = (
    "The narrative quality prior is trained on human-and-LLM-rated short "
    "stories (lars1234/story_writing_benchmark). It is a story-quality proxy, "
    "not a retention label."
)

EMOTION_DISCLAIMER = (
    "GoEmotions is general emotional-language evidence from Reddit comments. "
    "It is not a retention dataset and not fiction-specific ground truth."
)

StakesType = Literal[
    "physical", "emotional", "social", "financial", "moral", "existential", "none"
]
ConflictType = Literal[
    "interpersonal", "internal", "external_threat", "institutional", "none"
]
GenreLabel = Literal[
    "horror", "thriller", "romance", "drama", "fantasy",
    "sci_fi", "mystery", "comedy", "action", "literary",
]
EmotionLabel = Literal[
    "tension", "dread", "romance", "humor", "warmth",
    "anger", "hope", "sadness", "wonder", "neutral",
]
CliffhangerType = Literal[
    "danger", "revelation", "betrayal", "decision", "countdown",
    "disappearance", "false_resolution", "weak_resolved",
]
PacePreference = Literal["slow", "balanced", "fast"]
ToleranceLevel = Literal["low", "medium", "high"]
ListeningMode = Literal["binge", "commute", "casual"]


def _unit(value: float) -> float:
    """Clamp a sensor reading to 0.0–1.0."""
    return max(0.0, min(1.0, float(value)))


class SceneDNA(BaseModel):
    """
    What one scene objectively contains.

    This is a sensor reading, not a verdict. Nothing here says whether the
    scene is any good; downstream layers decide that. Keeping measurement and
    judgement apart is what lets the forecast explain itself.
    """

    model_config = ConfigDict(extra="forbid")

    # --- Pacing ---
    action_density: float = Field(description="Share of the scene that is physical event or motion, 0-1.")
    dialogue_ratio: float = Field(description="Share of the scene that is spoken dialogue, 0-1.")
    exposition_ratio: float = Field(description="Share that explains backstory, world, or history, 0-1.")
    internal_monologue_ratio: float = Field(description="Share that is a character thinking rather than acting, 0-1.")
    scene_tempo: float = Field(description="How fast the scene moves, 0 glacial to 1 breakneck.")

    # --- Emotion ---
    tension: float = Field(description="Sustained unease about an unresolved situation, 0-1.")
    dread: float = Field(description="Anticipatory fear that something bad is coming, 0-1.")
    romance: float = Field(description="Romantic or intimate charge, 0-1.")
    humor: float = Field(description="Comedic content, 0-1.")
    warmth: float = Field(description="Affection, safety, or human connection, 0-1.")
    anger: float = Field(description="Hostility or rage present in the scene, 0-1.")
    hope: float = Field(description="Forward-looking optimism, 0-1.")
    sadness: float = Field(description="Grief, loss, or melancholy, 0-1.")
    wonder: float = Field(description="Awe or fascination, 0-1.")

    # --- Craft ---
    stakes_level: float = Field(description="How much is at risk in this scene, 0-1.")
    stakes_type: StakesType = Field(description="The dominant kind of thing at risk.")
    conflict_present: bool = Field(description="Whether opposing forces are actively pushing against each other.")
    conflict_type: ConflictType = Field(description="The kind of conflict, or 'none'.")
    surprise_factor: float = Field(description="How much the scene departs from what the setup implied, 0-1.")
    complexity: float = Field(description="Cognitive load: threads, names, and concepts to track, 0-1.")
    character_development_present: bool = Field(description="Whether a character visibly changes or reveals something.")
    new_information_revealed: bool = Field(description="Whether the listener learns a materially new fact.")
    cliffhanger_strength: float = Field(description="How unresolved and pulling the final beat is, 0-1.")

    # --- Content ---
    violence_intensity: float = Field(description="Depicted violence, 0-1.")
    sexual_content: float = Field(description="Sexual content, 0-1.")
    profanity_level: float = Field(description="Profanity, 0-1.")
    dark_themes: float = Field(description="Abuse, death, despair, and similar, 0-1.")
    gore_level: float = Field(description="Graphic bodily harm, 0-1.")

    # --- Narrative ---
    worldbuilding_density: float = Field(description="Rate of new world facts, names, and rules per line, 0-1.")
    mystery_questions_opened: int = Field(description="Count of new unanswered questions this scene raises.")
    mystery_questions_answered: int = Field(description="Count of previously open questions this scene resolves.")
    active_character_count: int = Field(description="Distinct characters who speak or act.")
    setting_change: bool = Field(description="Whether the location changes within the scene.")
    time_skip: bool = Field(description="Whether time jumps forward or backward.")
    flashback: bool = Field(description="Whether the scene depicts an earlier time.")
    tropes_present: list[str] = Field(description="Recognisable narrative devices, lowercase snake_case.")
    genre_alignment: GenreLabel = Field(description="The genre this scene most reads as.")
    dominant_emotion: EmotionLabel = Field(description="The single strongest emotional register.")

    @field_validator(
        "action_density", "dialogue_ratio", "exposition_ratio",
        "internal_monologue_ratio", "scene_tempo", "tension", "dread",
        "romance", "humor", "warmth", "anger", "hope", "sadness", "wonder",
        "stakes_level", "surprise_factor", "complexity", "cliffhanger_strength",
        "violence_intensity", "sexual_content", "profanity_level",
        "dark_themes", "gore_level", "worldbuilding_density",
        mode="after",
    )
    @classmethod
    def _clamp_unit(cls, value: float) -> float:
        return _unit(value)

    @field_validator(
        "mystery_questions_opened", "mystery_questions_answered",
        "active_character_count",
        mode="after",
    )
    @classmethod
    def _clamp_count(cls, value: int) -> int:
        return max(0, int(value))

    @property
    def emotional_intensity(self) -> float:
        """Peak arousal across the emotion axes."""
        return max(
            self.tension, self.dread, self.anger,
            self.romance, self.wonder, self.sadness,
        )

    @property
    def event_movement(self) -> float:
        """How much actually happens, as opposed to how much is described."""
        return _unit(0.5 * self.action_density + 0.3 * self.scene_tempo + 0.2 * self.stakes_level)


class Scene(BaseModel):
    """One segment of the continuation, before any analysis."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="1-based scene number.")
    heading: str = Field(description="Slug line, or an inferred label for prose.")
    text: str = Field(description="Raw scene text.")
    inferred: bool = Field(
        default=False,
        description="True when the boundary was guessed rather than read from a slug line.",
    )

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class StructuralSignals(BaseModel):
    """
    Deterministic measurements computed locally from the text.

    None of this involves a model, so it is reproducible, free, and available
    even when every API is down. It is also the evidence the UI quotes back to
    the writer, because "exposition was high for two consecutive scenes" is
    checkable in a way that a model's opinion is not.
    """

    model_config = ConfigDict(extra="forbid")

    word_count: int
    sentence_count: int
    mean_sentence_length: float
    sentence_length_variance: float
    urgency_punctuation_density: float
    dialogue_line_ratio: float
    named_character_density: float
    pov_dominance: float
    question_density: float
    time_pressure_hits: int
    hook_marker_present: bool
    speaking_characters: list[str]

    # Sequence-aware, filled in once the whole episode is known.
    consecutive_exposition_scenes: int = 0
    consecutive_low_conflict_scenes: int = 0
    open_questions_running_total: int = 0
    payoff_debt: int = 0
    pov_repeat_streak: int = 0


class QualityPrior(BaseModel):
    """Output of the trained narrative-quality regressor, or its absence."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    score: float = Field(default=0.5, description="Quality proxy 0-1. 0.5 when unavailable.")
    message: str = Field(default="")
    model_version: str = Field(default="")
    top_contributors: list[str] = Field(default_factory=list)


class EmotionSignals(BaseModel):
    """Arousal and valence evidence for one scene."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["goemotions", "scene_dna_fallback"]
    arousal: float
    warmth: float
    negative_load: float
    tension: float
    volatility_from_previous: float = 0.0
    top_emotions: list[str] = Field(default_factory=list)


class CohortProfile(BaseModel):
    """
    A writer-declared target audience.

    This is explicitly a stated intent, not a measured population. The writer
    says who they are writing for and the forecast reports fit against that
    stated target.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    genre_affinity: GenreLabel
    pace_preference: PacePreference
    complexity_tolerance: ToleranceLevel
    emotional_preference: Literal["romance", "warmth", "dread", "action", "mystery"]
    content_boundary: ToleranceLevel = Field(
        description="Tolerance for violence, gore, and dark themes."
    )
    listening_mode: ListeningMode
    age_band: str = Field(default="unspecified", description="Writer-selected context only.")
    is_counterfactual: bool = False


class HazardContribution(BaseModel):
    """One named term in a scene's hazard, kept separate so the UI can show it."""

    model_config = ConfigDict(extra="forbid")

    factor: str
    delta: float = Field(description="Positive raises drop-off risk, negative lowers it.")
    detail: str


class SceneForecast(BaseModel):
    """Hazard and survival for a single scene under one cohort."""

    model_config = ConfigDict(extra="forbid")

    scene_index: int
    hazard: float
    survival: float = Field(description="Relative survival proxy 0-100.")
    contributions: list[HazardContribution] = Field(default_factory=list)

    @property
    def top_risk_factors(self) -> list[HazardContribution]:
        raising = [c for c in self.contributions if c.delta > 0]
        return sorted(raising, key=lambda c: c.delta, reverse=True)


class CohortCurve(BaseModel):
    """A full survival curve for one cohort."""

    model_config = ConfigDict(extra="forbid")

    cohort_label: str
    is_counterfactual: bool
    scenes: list[SceneForecast]

    @property
    def final_survival(self) -> float:
        return self.scenes[-1].survival if self.scenes else 100.0

    @property
    def curve(self) -> list[float]:
        return [100.0] + [s.survival for s in self.scenes]


class CliffhangerReport(BaseModel):
    """Episode-ending analysis and the Unlock Pull Index."""

    model_config = ConfigDict(extra="forbid")

    types: list[CliffhangerType]
    hook_strength: float = Field(description="0-100.")
    stakes: float
    information_gap: float
    surprise: float
    emotional_investment: float
    novelty: float
    payoff_debt_penalty: float
    false_resolution_risk: float
    unlock_pull_index: float = Field(description="0-100.")
    recommendation: str


class RiskExplanation(BaseModel):
    """Writer-facing explanation for one risk scene."""

    model_config = ConfigDict(extra="forbid")

    scene_index: int
    why_risky: str
    cohort_expectation: str
    surgical_fix: str
    trade_off: str
    source: Literal["deterministic", "llm_deep_dive"] = "deterministic"


class SceneAnalysis(BaseModel):
    """Everything known about one scene, across all four evidence layers."""

    model_config = ConfigDict(extra="forbid")

    scene: Scene
    dna: SceneDNA
    structural: StructuralSignals
    emotion: EmotionSignals
    quality: QualityPrior


class ForecastResult(BaseModel):
    """A complete Engagement Survival Forecast for one draft."""

    model_config = ConfigDict(extra="forbid")

    scenes: list[SceneAnalysis]
    primary_curve: CohortCurve
    counterfactual_curves: list[CohortCurve]
    cliffhanger: CliffhangerReport
    risk_explanations: list[RiskExplanation]
    quality_prior: QualityPrior
    cohort: CohortProfile
    disclaimer: str = FORECAST_DISCLAIMER

    @property
    def overall_survival(self) -> float:
        """Mean final survival across the primary and counterfactual cohorts."""
        curves = [self.primary_curve, *self.counterfactual_curves]
        return sum(c.final_survival for c in curves) / len(curves)

    @property
    def risk_ranking(self) -> list[SceneForecast]:
        return sorted(self.primary_curve.scenes, key=lambda s: s.hazard, reverse=True)

    def scene_by_index(self, index: int) -> SceneAnalysis | None:
        for analysis in self.scenes:
            if analysis.scene.index == index:
                return analysis
        return None


class MetricDelta(BaseModel):
    """Before/after movement for one headline number."""

    model_config = ConfigDict(extra="forbid")

    label: str
    before: float
    after: float
    noise_band: float = 0.0
    """
    Smallest movement worth reporting for this metric.

    The Scene DNA sensor is not reproducible even at temperature 0: re-measuring
    an unchanged episode moved the Unlock Pull Index across an 8.4-point range
    (sd 3.25) over five runs, because the model flips between calibration
    anchors on genuinely ambiguous scenes. Passing a seed did not fix it. A
    delta inside this band is measurement noise, not an effect of the revision,
    and the comparison card must not sell it as an improvement.
    """

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def significant(self) -> bool:
        """False when the movement is indistinguishable from sensor noise."""
        return abs(self.delta) > self.noise_band

    @property
    def improved(self) -> bool:
        return self.delta > 0 and self.significant


class ComparisonResult(BaseModel):
    """Baseline versus revised forecast."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[MetricDelta]
    risk_scene_movement: list[str]
    primary_evidence: list[str]
    headline: str
    disclaimer: str = FORECAST_DISCLAIMER


class DirectingSheetEntry(BaseModel):
    """Performance direction for one audio chunk, derived from the forecast."""

    model_config = ConfigDict(extra="forbid")

    chunk_index: int
    scene: int
    character: str
    narrative_role: str
    engagement_risk: Literal["low", "elevated", "high"]
    dominant_emotion: str
    delivery_tempo: str
    target_speed: float
    pause_before_reveal_ms: int
    instruction: str
    foley: list[str] = Field(default_factory=list)
    hook_role: str = ""
    survival_proxy: float = 0.0
    unlock_pull_index: float = 0.0
