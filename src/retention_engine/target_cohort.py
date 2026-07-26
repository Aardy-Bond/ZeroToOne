"""
Target cohort modelling.

A cohort here is a writer-declared intent, not a measured population. The
writer states who they are writing for; the forecast reports fit against that
stated target. Nothing in this module claims to describe real listeners, and
the counterfactual scenarios are explicitly labelled as what-ifs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import CohortProfile, SceneDNA

# Where each preference sits on a 0-1 axis. These are design constants that
# define what "fast" means to the hazard model, not measurements of anybody.
PACE_TARGET = {"slow": 0.30, "balanced": 0.55, "fast": 0.80}
COMPLEXITY_CEILING = {"low": 0.35, "medium": 0.60, "high": 0.85}
CONTENT_CEILING = {"low": 0.30, "medium": 0.60, "high": 0.95}

# How much exposition a cohort will sit through before it starts costing them.
EXPOSITION_TOLERANCE = {"slow": 0.60, "balanced": 0.45, "fast": 0.28}

# Listening mode scales overall patience. A commuter with one earbud in is
# less forgiving of a slow scene than someone four episodes into a binge.
MODE_PATIENCE = {"binge": 1.15, "casual": 1.00, "commute": 0.82}

# Which Scene DNA axis each emotional preference reads from.
EMOTION_AXIS = {
    "romance": "romance",
    "warmth": "warmth",
    "dread": "dread",
    "action": "action_density",
    "mystery": "tension",
}

DEFAULT_COHORT = CohortProfile(
    label="Primary target cohort",
    genre_affinity="horror",
    pace_preference="balanced",
    complexity_tolerance="medium",
    emotional_preference="dread",
    content_boundary="medium",
    listening_mode="commute",
    age_band="unspecified",
    is_counterfactual=False,
)


@dataclass(frozen=True)
class CohortVector:
    """A cohort reduced to the numbers the hazard model consumes."""

    label: str
    is_counterfactual: bool
    pace_target: float
    complexity_ceiling: float
    exposition_tolerance: float
    content_ceiling: float
    emotion_axis: str
    patience: float
    genre: str

    def pace_mismatch(self, dna: SceneDNA) -> float:
        """Distance between the scene's tempo and what this cohort wants."""
        return abs(dna.scene_tempo - self.pace_target)

    def complexity_overshoot(self, dna: SceneDNA) -> float:
        return max(0.0, dna.complexity - self.complexity_ceiling)

    def exposition_overshoot(self, dna: SceneDNA) -> float:
        return max(0.0, dna.exposition_ratio - self.exposition_tolerance)

    def content_overshoot(self, dna: SceneDNA) -> float:
        worst = max(dna.violence_intensity, dna.gore_level, dna.dark_themes)
        return max(0.0, worst - self.content_ceiling)

    def emotional_alignment(self, dna: SceneDNA) -> float:
        """How strongly the scene delivers the register this cohort came for."""
        return float(getattr(dna, self.emotion_axis, 0.0))

    def genre_alignment(self, dna: SceneDNA) -> float:
        return 1.0 if dna.genre_alignment == self.genre else 0.0


def to_vector(cohort: CohortProfile) -> CohortVector:
    """Reduce a declared cohort to its numeric preference vector."""
    return CohortVector(
        label=cohort.label,
        is_counterfactual=cohort.is_counterfactual,
        pace_target=PACE_TARGET[cohort.pace_preference],
        complexity_ceiling=COMPLEXITY_CEILING[cohort.complexity_tolerance],
        exposition_tolerance=EXPOSITION_TOLERANCE[cohort.pace_preference],
        content_ceiling=CONTENT_CEILING[cohort.content_boundary],
        emotion_axis=EMOTION_AXIS[cohort.emotional_preference],
        patience=MODE_PATIENCE[cohort.listening_mode],
        genre=cohort.genre_affinity,
    )


def counterfactual_cohorts(primary: CohortProfile) -> list[CohortProfile]:
    """
    Three labelled what-if cohorts.

    Each varies one axis away from the writer's stated target so the UI can
    answer "what if I am wrong about who this is for?" without pretending
    these are real audience segments.
    """
    base = primary.model_dump()

    faster = {
        **base,
        "label": "What-if: faster pace, low exposition tolerance",
        "pace_preference": "fast",
        "listening_mode": "commute",
        "is_counterfactual": True,
    }
    slow_burn = {
        **base,
        "label": "What-if: slow-burn, high mystery tolerance",
        "pace_preference": "slow",
        "complexity_tolerance": "high",
        "listening_mode": "binge",
        "is_counterfactual": True,
    }
    clarity = {
        **base,
        "label": "What-if: clarity-first, low complexity tolerance",
        "complexity_tolerance": "low",
        "pace_preference": "balanced",
        "listening_mode": "casual",
        "is_counterfactual": True,
    }

    return [
        CohortProfile(**faster),
        CohortProfile(**slow_burn),
        CohortProfile(**clarity),
    ]
