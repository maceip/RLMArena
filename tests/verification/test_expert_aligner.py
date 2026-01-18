"""
Tests for Expert Aligner Service.
"""

import pytest
from datetime import datetime

from qqr.verification.expert_aligner import (
    ExpertAlignerService,
    TrajectoryPoint,
    TechnicalRationale,
    FailureCategory,
    GoldComparison,
)


class TestTrajectoryPoint:
    """Tests for TrajectoryPoint dataclass."""

    def test_create_trajectory_point(self):
        point = TrajectoryPoint(
            message_index=3,
            tool_call_index=1,
            content_snippet="eval(user_input)",
        )

        assert point.message_index == 3
        assert point.tool_call_index == 1
        assert point.content_snippet == "eval(user_input)"

    def test_to_dict(self):
        point = TrajectoryPoint(message_index=5)
        result = point.to_dict()

        assert result["message_index"] == 5
        assert result["tool_call_index"] is None

    def test_from_dict(self):
        data = {
            "message_index": 2,
            "tool_call_index": 0,
            "token_start": 100,
            "token_end": 150,
            "content_snippet": "test",
        }
        point = TrajectoryPoint.from_dict(data)

        assert point.message_index == 2
        assert point.tool_call_index == 0
        assert point.token_start == 100


class TestTechnicalRationale:
    """Tests for TechnicalRationale dataclass."""

    def test_create_rationale(self):
        rationale = TechnicalRationale(
            category=FailureCategory.SECURITY_VIOLATION,
            description="SQL injection vulnerability detected",
            severity=9,
            rule_id="SEC001",
            suggested_fix="Use parameterized queries",
        )

        assert rationale.category == FailureCategory.SECURITY_VIOLATION
        assert rationale.severity == 9
        assert "injection" in rationale.description.lower()

    def test_to_dict(self):
        rationale = TechnicalRationale(
            category=FailureCategory.SYNTAX_ERROR,
            description="Invalid syntax",
            severity=5,
        )
        result = rationale.to_dict()

        assert result["category"] == "syntax_error"
        assert result["severity"] == 5

    def test_from_dict(self):
        data = {
            "category": "policy_breach",
            "description": "Violates network policy",
            "severity": 7,
        }
        rationale = TechnicalRationale.from_dict(data)

        assert rationale.category == FailureCategory.POLICY_BREACH


class TestExpertAlignerService:
    """Tests for ExpertAlignerService."""

    @pytest.fixture
    def service(self):
        return ExpertAlignerService()

    @pytest.fixture
    def sample_trajectories(self):
        return (
            [
                {"role": "user", "content": "Write a function to add two numbers"},
                {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
            ],
            [
                {"role": "user", "content": "Write a function to add two numbers"},
                {"role": "assistant", "content": "def add(a, b):\n    result = a + b\n    return result"},
            ],
        )

    def test_create_gold_comparison(self, service, sample_trajectories):
        traj_a, traj_b = sample_trajectories

        comparison = service.create_gold_comparison(
            input_query="Write a function to add two numbers",
            trajectory_a=traj_a,
            trajectory_b=traj_b,
        )

        assert comparison.id is not None
        assert comparison.input_query == "Write a function to add two numbers"
        assert comparison.trajectory_a == traj_a
        assert comparison.trajectory_b == traj_b

    def test_add_sme_label(self, service, sample_trajectories):
        traj_a, traj_b = sample_trajectories

        comparison = service.create_gold_comparison(
            input_query="Test query",
            trajectory_a=traj_a,
            trajectory_b=traj_b,
        )

        label = service.add_sme_label(
            comparison_id=comparison.id,
            sme_id="sme_001",
            winner="a",
            confidence=0.9,
            failure_points=[TrajectoryPoint(message_index=1)],
            rationales=[
                TechnicalRationale(
                    category=FailureCategory.LOGIC_ERROR,
                    description="Unnecessary variable",
                    severity=3,
                )
            ],
        )

        assert label.sme_id == "sme_001"
        assert label.winner == "a"
        assert label.confidence == 0.9

    def test_consensus_computation(self, service, sample_trajectories):
        traj_a, traj_b = sample_trajectories

        comparison = service.create_gold_comparison(
            input_query="Test query",
            trajectory_a=traj_a,
            trajectory_b=traj_b,
        )

        # Add multiple SME labels
        service.add_sme_label(comparison.id, "sme_001", "a", 0.9)
        service.add_sme_label(comparison.id, "sme_002", "a", 0.8)
        service.add_sme_label(comparison.id, "sme_003", "b", 0.6)

        # Retrieve and check consensus
        updated = service.get_comparison(comparison.id)
        assert updated.consensus_winner == "a"
        assert updated.consensus_confidence > 0.5

    def test_export_for_training(self, service, sample_trajectories):
        traj_a, traj_b = sample_trajectories

        comparison = service.create_gold_comparison(
            input_query="Test query",
            trajectory_a=traj_a,
            trajectory_b=traj_b,
        )

        service.add_sme_label(comparison.id, "sme_001", "a", 0.9)

        training_data = service.export_for_training(min_confidence=0.5)

        assert len(training_data) == 1
        assert training_data[0]["id"] == comparison.id
        assert "chosen" in training_data[0]
        assert "rejected" in training_data[0]

    def test_get_statistics(self, service, sample_trajectories):
        traj_a, traj_b = sample_trajectories

        # Create some comparisons
        for i in range(3):
            comp = service.create_gold_comparison(f"Query {i}", traj_a, traj_b)
            service.add_sme_label(comp.id, f"sme_{i}", "a", 0.8)

        stats = service.get_statistics()

        assert stats["total_comparisons"] == 3
        assert stats["total_labels"] == 3
        assert stats["unique_smes"] == 3

    def test_invalid_winner_raises_error(self, service, sample_trajectories):
        traj_a, traj_b = sample_trajectories

        comparison = service.create_gold_comparison("Test", traj_a, traj_b)

        with pytest.raises(ValueError):
            service.add_sme_label(comparison.id, "sme_001", "invalid", 0.8)

    def test_invalid_confidence_raises_error(self, service, sample_trajectories):
        traj_a, traj_b = sample_trajectories

        comparison = service.create_gold_comparison("Test", traj_a, traj_b)

        with pytest.raises(ValueError):
            service.add_sme_label(comparison.id, "sme_001", "a", 1.5)
