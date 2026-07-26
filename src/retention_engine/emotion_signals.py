"""
Emotion and arousal evidence.

Prefers a pre-trained GoEmotions classifier when transformers and torch are
installed and the weights are cached locally. Neither is a hard dependency —
they pull in gigabytes — so the default path derives the same signals from
Scene DNA and reports `source="scene_dna_fallback"` so the UI can say which
evidence it is showing.

GoEmotions is 58k labelled Reddit comments. It is general emotional-language
evidence, not a retention dataset and not fiction-specific ground truth.
"""

from __future__ import annotations

import logging
import re

from .schemas import EmotionSignals, SceneDNA

logger = logging.getLogger(__name__)

GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"

# GoEmotions labels grouped onto the axes the forecast actually uses.
HIGH_AROUSAL = {
    "anger", "annoyance", "excitement", "fear", "nervousness", "surprise",
    "disgust", "rage", "joy", "amusement",
}
WARM = {"love", "caring", "gratitude", "admiration", "joy", "relief", "approval"}
NEGATIVE = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}
TENSE = {"fear", "nervousness", "surprise", "confusion", "embarrassment"}

_CLASSIFIER = None
_CLASSIFIER_TRIED = False


def _load_classifier():
    """
    Try once to load a local GoEmotions pipeline.

    `local_files_only` is deliberate: a hackathon demo should never block on a
    multi-gigabyte download triggered by opening a dashboard.
    """
    global _CLASSIFIER, _CLASSIFIER_TRIED
    if _CLASSIFIER_TRIED:
        return _CLASSIFIER
    _CLASSIFIER_TRIED = True

    try:
        from transformers import pipeline

        _CLASSIFIER = pipeline(
            "text-classification",
            model=GOEMOTIONS_MODEL,
            top_k=None,
            truncation=True,
            local_files_only=True,
        )
        logger.info("GoEmotions classifier loaded from local cache.")
    except Exception as exc:
        logger.info("GoEmotions unavailable, using Scene DNA fallback (%s).", exc)
        _CLASSIFIER = None

    return _CLASSIFIER


def _from_scene_dna(dna: SceneDNA) -> EmotionSignals:
    """Derive the same axes from sensor output when no classifier is present."""
    arousal = max(
        dna.tension, dna.dread, dna.anger, dna.action_density * 0.9,
        dna.wonder * 0.8, dna.romance * 0.7,
    )
    negative = max(dna.dread, dna.anger, dna.sadness, dna.dark_themes * 0.8)

    ranked = sorted(
        [
            ("tension", dna.tension), ("dread", dna.dread), ("romance", dna.romance),
            ("humor", dna.humor), ("warmth", dna.warmth), ("anger", dna.anger),
            ("hope", dna.hope), ("sadness", dna.sadness), ("wonder", dna.wonder),
        ],
        key=lambda pair: pair[1],
        reverse=True,
    )

    return EmotionSignals(
        source="scene_dna_fallback",
        arousal=round(arousal, 3),
        warmth=round(max(dna.warmth, dna.hope * 0.7, dna.romance * 0.6), 3),
        negative_load=round(negative, 3),
        tension=round(dna.tension, 3),
        top_emotions=[name for name, score in ranked[:3] if score > 0.15],
    )


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]
    return parts or [text.strip()]


def _from_classifier(text: str, dna: SceneDNA) -> EmotionSignals | None:
    """
    Corroborating emotion evidence from GoEmotions.

    Scored per sentence and max-pooled. Classifying a whole scene at once
    averages an emotional peak away against the surrounding description and
    returns near-zero on everything.

    Measured on this repo's sample: the classifier labels every scene of a
    horror episode "neutral" at the passage level, and at sentence level
    returns confident but wrong labels (admiration 0.92 on a body-in-the-floor
    reveal). It was trained on first-person Reddit comments, where a writer
    states their own feelings; narrative prose describes events instead. So it
    is treated as evidence that can only *raise* an axis, never lower one. A
    "neutral" verdict on fiction is an absent reading, not evidence of calm,
    and zeroing the sensor on that basis would flatten the Narrative EKG.
    """
    classifier = _load_classifier()
    if classifier is None:
        return None

    try:
        raw = classifier(_sentences(text[:6000]))
    except Exception as exc:
        logger.warning("GoEmotions inference failed, falling back: %s", exc)
        return None

    per_sentence: list[dict[str, float]] = []
    for entry in raw:
        scores = entry if isinstance(entry, list) else [entry]
        per_sentence.append({s["label"].lower(): float(s["score"]) for s in scores})

    if not per_sentence:
        return None

    def group(labels: set[str]) -> float:
        return max(
            (max((sentence.get(label, 0.0) for label in labels), default=0.0)
             for sentence in per_sentence),
            default=0.0,
        )

    pooled: dict[str, float] = {}
    for sentence in per_sentence:
        for label, score in sentence.items():
            if label != "neutral":
                pooled[label] = max(pooled.get(label, 0.0), score)

    fallback = _from_scene_dna(dna)
    ranked = sorted(pooled.items(), key=lambda pair: pair[1], reverse=True)

    return EmotionSignals(
        source="goemotions",
        arousal=round(max(group(HIGH_AROUSAL), fallback.arousal), 3),
        warmth=round(max(group(WARM), fallback.warmth), 3),
        negative_load=round(max(group(NEGATIVE), fallback.negative_load), 3),
        tension=round(max(group(TENSE), dna.tension), 3),
        top_emotions=[label for label, score in ranked[:3] if score > 0.15],
    )


def analyse_emotions(
    texts: list[str],
    dna_list: list[SceneDNA],
    *,
    use_classifier: bool = True,
) -> list[EmotionSignals]:
    """
    Produce emotion signals for every scene, with volatility between them.

    Volatility is the scene-to-scene change in arousal. A flat episode and a
    wildly swinging one can share an average while feeling nothing alike.
    """
    if len(texts) != len(dna_list):
        raise ValueError("texts and dna_list must be the same length.")

    signals: list[EmotionSignals] = []
    for text, dna in zip(texts, dna_list):
        result = _from_classifier(text, dna) if use_classifier else None
        signals.append(result or _from_scene_dna(dna))

    for position in range(1, len(signals)):
        signals[position].volatility_from_previous = round(
            abs(signals[position].arousal - signals[position - 1].arousal), 3
        )

    return signals


def emotion_source_note(signals: list[EmotionSignals]) -> str:
    """One line describing which evidence the UI is actually showing."""
    if signals and all(s.source == "goemotions" for s in signals):
        return (
            "Emotion axes corroborated by a pre-trained GoEmotions classifier, "
            "pooled per sentence. GoEmotions is first-person Reddit comments, not "
            "fiction and not a retention dataset; on narrative prose it reads "
            "mostly neutral, so it can only raise an axis above the Scene DNA "
            "reading, never lower one."
        )
    return (
        "Emotion axes derived from Scene DNA sensor readings (GoEmotions model not "
        "installed locally). Install transformers and torch, then cache "
        f"{GOEMOTIONS_MODEL}, to switch to classifier-based evidence."
    )
