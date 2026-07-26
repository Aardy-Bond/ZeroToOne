"""
Engagement Survival Forecast.

A transparent, evidence-backed pre-release forecast for serialized audio
fiction. Four separate evidence layers feed a hand-specified hazard model:

1. Scene DNA sensor       — one cheap structured LLM reading per scene
2. Structural signals     — deterministic, local, model-free measurements
3. Narrative quality prior— TF-IDF + Ridge trained on rated short stories
4. Target-cohort fit      — writer-declared audience, plus counterfactuals

Nothing here is calibrated against listener behaviour. See FORECAST_DISCLAIMER.
"""

from .schemas import (
    EMOTION_DISCLAIMER,
    FORECAST_DISCLAIMER,
    QUALITY_PRIOR_DISCLAIMER,
    CliffhangerReport,
    CohortProfile,
    ComparisonResult,
    DirectingSheetEntry,
    ForecastResult,
    QualityPrior,
    Scene,
    SceneAnalysis,
    SceneDNA,
    StructuralSignals,
)

__all__ = [
    "FORECAST_DISCLAIMER",
    "QUALITY_PRIOR_DISCLAIMER",
    "EMOTION_DISCLAIMER",
    "CliffhangerReport",
    "CohortProfile",
    "ComparisonResult",
    "DirectingSheetEntry",
    "ForecastResult",
    "QualityPrior",
    "Scene",
    "SceneAnalysis",
    "SceneDNA",
    "StructuralSignals",
]
