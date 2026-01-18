"""
Tests for DPO Distillation Pipeline and Training Store.
"""

import pytest
from datetime import datetime

from qqr.verification.distillation import (
    DistillationPipeline,
    DatasetTriplet,
    TripletQuality,
    DatasetFormat,
    QualityFilter,
    RationaleExtractor,
    SlimeTrainerAdapter,
    DEEPSEEK_CODER_V2,
)

from qqr.verification.training_store import (
    TrainingStore,
    StoredTriplet,
    StoreStatus,
    DatasetVersion,
)


class TestDatasetTriplet:
    """Tests for DatasetTriplet."""

    def test_create_triplet(self):
        triplet = DatasetTriplet(
            id="test-001",
            input_query="Write a function",
            chosen=[{"role": "assistant", "content": "def foo(): pass"}],
            rejected=[{"role": "assistant", "content": "funciton foo() {}"}],
            confidence=0.9,
            quality=TripletQuality.GOLD,
            rationale="Chosen has correct syntax",
        )

        assert triplet.id == "test-001"
        assert triplet.confidence == 0.9
        assert triplet.quality == TripletQuality.GOLD

    def test_to_dict(self):
        triplet = DatasetTriplet(
            id="test-002",
            input_query="Test",
            chosen=[],
            rejected=[],
            confidence=0.8,
            quality=TripletQuality.SILVER,
        )

        result = triplet.to_dict()
        assert result["id"] == "test-002"
        assert result["quality"] == "silver"

    def test_to_slime_format(self):
        triplet = DatasetTriplet(
            id="test-003",
            input_query="Hello",
            chosen=[{"role": "assistant", "content": "Hi there!"}],
            rejected=[{"role": "assistant", "content": "..."}],
            confidence=0.85,
            quality=TripletQuality.BRONZE,
        )

        result = triplet.to_slime_format()
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert result["chosen_score"] == 0.85


class TestQualityFilter:
    """Tests for QualityFilter."""

    @pytest.fixture
    def filter(self):
        return QualityFilter(
            min_confidence=0.6,
            min_score_diff=0.1,
            max_trajectory_length=10,
        )

    def test_high_confidence_passes(self, filter):
        triplet = DatasetTriplet(
            id="test",
            input_query="Test",
            chosen=[{"role": "assistant", "content": "Good"}],
            rejected=[{"role": "assistant", "content": "Bad"}],
            confidence=0.9,
            quality=TripletQuality.GOLD,
        )

        passed, reason = filter.filter(triplet)
        assert passed
        assert reason is None

    def test_low_confidence_fails(self, filter):
        triplet = DatasetTriplet(
            id="test",
            input_query="Test",
            chosen=[],
            rejected=[],
            confidence=0.4,
            quality=TripletQuality.BRONZE,
        )

        passed, reason = filter.filter(triplet)
        assert not passed
        assert "confidence" in reason.lower()

    def test_long_trajectory_fails(self, filter):
        triplet = DatasetTriplet(
            id="test",
            input_query="Test",
            chosen=[{"role": "assistant", "content": f"msg{i}"} for i in range(20)],
            rejected=[],
            confidence=0.9,
            quality=TripletQuality.GOLD,
        )

        passed, reason = filter.filter(triplet)
        assert not passed
        assert "long" in reason.lower()


class TestDistillationPipeline:
    """Tests for DistillationPipeline."""

    @pytest.fixture
    def pipeline(self):
        return DistillationPipeline(
            quality_filter=QualityFilter(min_confidence=0.5),
            target_format=DatasetFormat.SLIME,
        )

    def test_process_tournament_result(self, pipeline):
        trajectories = [
            [{"role": "assistant", "content": "Winner response"}],
            [{"role": "assistant", "content": "Loser response"}],
        ]
        scores = [0.8, 0.2]

        triplets = pipeline.process_tournament_result(
            input_query="Test query",
            trajectories=trajectories,
            scores=scores,
        )

        assert len(triplets) == 1
        assert triplets[0].chosen == trajectories[0]
        assert triplets[0].rejected == trajectories[1]

    def test_process_correction_trajectory(self, pipeline):
        original = [{"role": "assistant", "content": "Broken code"}]
        corrected = [{"role": "assistant", "content": "Fixed code"}]

        triplet = pipeline.process_correction_trajectory(
            original_messages=[{"role": "user", "content": "Write code"}] + original,
            corrected_messages=[{"role": "user", "content": "Write code"}] + corrected,
            failure_reasons=["Syntax error"],
            num_attempts=2,
        )

        assert triplet is not None
        assert triplet.metadata["source"] == "self_correction"

    def test_export_jsonl(self, pipeline):
        trajectories = [
            [{"role": "assistant", "content": "A"}],
            [{"role": "assistant", "content": "B"}],
        ]

        pipeline.process_tournament_result("Query", trajectories, [0.9, 0.1])

        jsonl = pipeline.export(DatasetFormat.JSONL)
        assert isinstance(jsonl, str)
        assert len(jsonl.split("\n")) == 1

    def test_export_slime(self, pipeline):
        trajectories = [
            [{"role": "assistant", "content": "A"}],
            [{"role": "assistant", "content": "B"}],
        ]

        pipeline.process_tournament_result("Query", trajectories, [0.9, 0.1])

        slime_data = pipeline.export(DatasetFormat.SLIME)
        assert isinstance(slime_data, list)
        assert len(slime_data) == 1
        assert "prompt" in slime_data[0]

    def test_get_stats(self, pipeline):
        trajectories = [
            [{"role": "assistant", "content": "A"}],
            [{"role": "assistant", "content": "B"}],
        ]

        pipeline.process_tournament_result("Query", trajectories, [0.9, 0.1])

        stats = pipeline.get_stats()
        assert stats["total_triplets"] == 1
        assert stats["total_processed"] == 1


class TestTrainingStore:
    """Tests for TrainingStore."""

    @pytest.fixture
    def store(self):
        return TrainingStore()

    def test_add_triplet(self, store):
        triplet_id = store.add_triplet(
            input_query="Write hello world",
            chosen=[{"role": "assistant", "content": "print('Hello')"}],
            rejected=[{"role": "assistant", "content": "echo Hello"}],
            confidence=0.9,
            quality="gold",
        )

        assert triplet_id is not None

        retrieved = store.get_triplet(triplet_id)
        assert retrieved is not None
        assert retrieved.input_query == "Write hello world"
        assert retrieved.confidence == 0.9

    def test_add_triplets_batch(self, store):
        triplets = [
            {
                "input_query": f"Query {i}",
                "chosen": [],
                "rejected": [],
                "confidence": 0.8,
                "quality": "silver",
            }
            for i in range(5)
        ]

        ids = store.add_triplets_batch(triplets)
        assert len(ids) == 5

    def test_query_triplets_by_quality(self, store):
        store.add_triplet("Q1", [], [], 0.9, "gold")
        store.add_triplet("Q2", [], [], 0.7, "silver")
        store.add_triplet("Q3", [], [], 0.9, "gold")

        gold_triplets = store.query_triplets(quality="gold")
        assert len(gold_triplets) == 2

    def test_query_triplets_by_confidence(self, store):
        store.add_triplet("Q1", [], [], 0.9, "gold")
        store.add_triplet("Q2", [], [], 0.5, "silver")
        store.add_triplet("Q3", [], [], 0.8, "bronze")

        high_conf = store.query_triplets(min_confidence=0.7)
        assert len(high_conf) == 2

    def test_create_dataset_version(self, store):
        store.add_triplet("Q1", [], [], 0.9, "gold")
        store.add_triplet("Q2", [], [], 0.7, "silver")

        version = store.create_dataset_version(
            name="v1.0",
            description="First release",
            quality_filter="gold",
        )

        assert version.triplet_count == 1
        assert version.name == "v1.0"

    def test_export_version_jsonl(self, store):
        store.add_triplet("Q1", [{"role": "a", "content": "X"}], [], 0.9, "gold")

        version = store.create_dataset_version("v1.0")
        jsonl = store.export_version_jsonl(version.version_id)

        assert isinstance(jsonl, str)
        assert "Q1" in jsonl

    def test_get_stats(self, store):
        store.add_triplet("Q1", [], [], 0.9, "gold", source="tournament")
        store.add_triplet("Q2", [], [], 0.7, "silver", source="correction")

        stats = store.get_stats()
        assert stats["total_triplets"] == 2
        assert "gold" in stats["quality_distribution"]
        assert "tournament" in stats["source_distribution"]


class TestRationaleExtractor:
    """Tests for RationaleExtractor."""

    @pytest.fixture
    def extractor(self):
        return RationaleExtractor()

    def test_extract_rationale_field(self, extractor):
        output = {"rationale": "A is better because..."}
        result = extractor.extract(output)
        assert result == "A is better because..."

    def test_extract_reasoning_field(self, extractor):
        output = {"reasoning": "The first solution is more efficient"}
        result = extractor.extract(output)
        assert "efficient" in result

    def test_format_for_training(self, extractor):
        rationale = "This is the reasoning"
        formatted = extractor.format_for_training(rationale, include_chain_of_thought=True)
        assert "<thinking>" in formatted
        assert "</thinking>" in formatted


class TestSlimeTrainerAdapter:
    """Tests for SlimeTrainerAdapter."""

    @pytest.fixture
    def adapter(self):
        return SlimeTrainerAdapter(model_config=DEEPSEEK_CODER_V2)

    def test_prepare_dataset(self, adapter):
        triplets = [
            DatasetTriplet(
                id=f"t{i}",
                input_query=f"Query {i}",
                chosen=[],
                rejected=[],
                confidence=0.8,
                quality=TripletQuality.GOLD,
            )
            for i in range(10)
        ]

        dataset = adapter.prepare_dataset(triplets, split_ratio=0.8)

        assert "train" in dataset
        assert "eval" in dataset
        assert len(dataset["train"]) == 8
        assert len(dataset["eval"]) == 2

    def test_generate_config(self, adapter):
        config = adapter.generate_config()

        assert config["model"]["name"] == "deepseek-coder-v2"
        assert "training" in config
        assert "dpo" in config

    def test_estimate_training_time(self, adapter):
        estimate = adapter.estimate_training_time(1000, gpu_type="H100")

        assert estimate["estimated_hours"] > 0
        assert estimate["estimated_cost_usd"] > 0
        assert estimate["gpu_type"] == "H100"
