"""
Narrative quality prior.

Loads a locally trained TF-IDF + Ridge regressor and scores scene text on a
0-1 quality axis. The model is trained on lars1234/story_writing_benchmark,
which rates LLM-written short stories on craft dimensions.

This is a story-quality proxy. It is not a retention label, and it has never
seen a listener. When the artifact is missing the module reports
`available=False` and the hazard model skips the quality term entirely rather
than substituting a guess.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .schemas import QUALITY_PRIOR_DISCLAIMER, QualityPrior

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "models"
ARTIFACT_PATH = ARTIFACT_DIR / "quality_prior.joblib"
METADATA_PATH = ARTIFACT_DIR / "quality_prior.meta.json"

FEATURE_VERSION = "qp-1"

UNAVAILABLE_MESSAGE = (
    "Quality-prior artifact not trained. Run "
    "`python scripts/train_quality_proxy.py` to enable this evidence layer. "
    "The forecast omits the quality term until then."
)


class QualityProxy:
    """Lazy-loading wrapper around the trained regressor."""

    def __init__(self, artifact_path: Path | str = ARTIFACT_PATH) -> None:
        self.artifact_path = Path(artifact_path)
        self._pipeline = None
        self._metadata: dict = {}
        self._load_attempted = False
        self._load_error = ""

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._pipeline is not None

    @property
    def model_version(self) -> str:
        self._ensure_loaded()
        return str(self._metadata.get("model_version", ""))

    @property
    def metadata(self) -> dict:
        self._ensure_loaded()
        return dict(self._metadata)

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        if not self.artifact_path.exists():
            self._load_error = UNAVAILABLE_MESSAGE
            logger.info("Quality prior artifact absent at %s", self.artifact_path)
            return

        try:
            import joblib

            self._pipeline = joblib.load(self.artifact_path)
        except Exception as exc:
            self._load_error = f"Quality-prior artifact failed to load: {exc}"
            self._pipeline = None
            logger.warning(self._load_error)
            return

        meta_path = self.artifact_path.with_suffix("").with_suffix(".meta.json")
        if not meta_path.exists():
            meta_path = METADATA_PATH
        if meta_path.exists():
            try:
                self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._metadata = {}

        logger.info(
            "Loaded quality prior %s trained on %s samples.",
            self._metadata.get("model_version", "unknown"),
            self._metadata.get("sample_count", "?"),
        )

    def score(self, text: str) -> QualityPrior:
        """Score one passage. Never raises; an unusable model reports unavailable."""
        self._ensure_loaded()

        if self._pipeline is None:
            return QualityPrior(available=False, score=0.5, message=self._load_error)

        cleaned = (text or "").strip()
        if len(cleaned.split()) < 20:
            return QualityPrior(
                available=False,
                score=0.5,
                message="Passage too short for a reliable quality reading (under 20 words).",
                model_version=self.model_version,
            )

        try:
            raw = float(self._pipeline.predict([cleaned])[0])
        except Exception as exc:
            logger.warning("Quality prior prediction failed: %s", exc)
            return QualityPrior(
                available=False, score=0.5, message=f"Prediction failed: {exc}"
            )

        low = float(self._metadata.get("label_min", 0.0))
        high = float(self._metadata.get("label_max", 10.0))
        span = high - low if high > low else 1.0
        normalised = max(0.0, min(1.0, (raw - low) / span))

        return QualityPrior(
            available=True,
            score=round(normalised, 4),
            message=QUALITY_PRIOR_DISCLAIMER,
            model_version=self.model_version,
            top_contributors=list(self._metadata.get("top_contributors", []))[:5],
        )


_DEFAULT_PROXY: QualityProxy | None = None


def get_quality_proxy() -> QualityProxy:
    """Shared instance so the artifact is read from disk once per process."""
    global _DEFAULT_PROXY
    if _DEFAULT_PROXY is None:
        _DEFAULT_PROXY = QualityProxy()
    return _DEFAULT_PROXY


def score_text(text: str) -> QualityPrior:
    return get_quality_proxy().score(text)
