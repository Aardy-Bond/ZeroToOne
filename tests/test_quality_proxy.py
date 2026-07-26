"""Quality-prior artifact loading, and the unavailable fallback."""

from __future__ import annotations

import json

from retention_engine.quality_proxy import QualityProxy
from retention_engine.schemas import QualityPrior

PASSAGE = " ".join(["The bulb swung once and then held still above the folded shirts."] * 4)


class TestUnavailableArtifact:
    def test_missing_artifact_reports_unavailable(self, tmp_path):
        proxy = QualityProxy(tmp_path / "absent.joblib")
        assert proxy.available is False

        result = proxy.score(PASSAGE)
        assert result.available is False
        assert result.score == 0.5
        assert "train_quality_proxy" in result.message

    def test_corrupt_artifact_degrades_instead_of_raising(self, tmp_path):
        artifact = tmp_path / "broken.joblib"
        artifact.write_bytes(b"not a joblib file at all")

        result = QualityProxy(artifact).score(PASSAGE)
        assert result.available is False
        assert "failed to load" in result.message.lower()

    def test_short_passage_is_not_scored(self, tmp_path):
        result = QualityProxy(tmp_path / "absent.joblib").score("Too short.")
        assert result.available is False

    def test_neutral_score_carries_no_hazard_signal(self, tmp_path):
        """0.5 with available=False must never be read as a real reading."""
        result = QualityProxy(tmp_path / "absent.joblib").score(PASSAGE)
        assert result == QualityPrior(
            available=False, score=0.5, message=result.message
        )


class TestTrainedArtifact:
    def _write_artifact(self, tmp_path):
        joblib = __import__("joblib")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(
            [("tfidf", TfidfVectorizer(min_df=1)), ("ridge", Ridge(alpha=1.0))]
        )
        texts = ["a tense and frightening passage about a house"] * 5 + [
            "a dull recitation of county records and easements"
        ] * 5
        labels = [3.5] * 5 + [1.8] * 5
        pipeline.fit(texts, labels)

        artifact = tmp_path / "quality_prior.joblib"
        joblib.dump(pipeline, artifact)
        (tmp_path / "quality_prior.meta.json").write_text(
            json.dumps(
                {
                    "model_version": "qp-test",
                    "label_min": 1.67,
                    "label_max": 3.78,
                    "sample_count": 10,
                    "top_contributors": ["tense", "frightening"],
                }
            )
        )
        return artifact

    def test_trained_artifact_scores_in_unit_range(self, tmp_path):
        proxy = QualityProxy(self._write_artifact(tmp_path))
        result = proxy.score(PASSAGE)

        assert result.available is True
        assert 0.0 <= result.score <= 1.0
        assert result.model_version == "qp-test"

    def test_message_states_it_is_not_a_retention_label(self, tmp_path):
        result = QualityProxy(self._write_artifact(tmp_path)).score(PASSAGE)
        assert "not a retention label" in result.message

    def test_metadata_contributors_are_surfaced(self, tmp_path):
        result = QualityProxy(self._write_artifact(tmp_path)).score(PASSAGE)
        assert "tense" in result.top_contributors

    def test_artifact_is_read_from_disk_only_once(self, tmp_path):
        proxy = QualityProxy(self._write_artifact(tmp_path))
        proxy.score(PASSAGE)
        loaded = proxy._pipeline
        proxy.score(PASSAGE)
        assert proxy._pipeline is loaded
