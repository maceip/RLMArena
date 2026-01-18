"""
DPO Distillation Pipeline for model training.

Converts tournament history and correction trajectories into training datasets
for student model fine-tuning using Direct Preference Optimization (DPO).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json
import hashlib
import uuid


class DatasetFormat(Enum):
    """Supported dataset output formats."""
    JSONL = "jsonl"
    PARQUET = "parquet"
    HF_DATASET = "hf_dataset"
    SLIME = "slime"


class TripletQuality(Enum):
    """Quality tier for training triplets."""
    GOLD = "gold"  # SME-verified
    SILVER = "silver"  # High-confidence tournament result
    BRONZE = "bronze"  # Lower confidence but still valid


@dataclass
class DatasetTriplet:
    """
    Training triplet for DPO: (Input, Verified_Winner, DoA_Failure).

    Following the STaR (Self-Taught Reasoner) paper format.
    """
    id: str
    input_query: str
    chosen: list[dict[str, Any]]  # Verified winner trajectory
    rejected: list[dict[str, Any]]  # DoA/failed trajectory
    confidence: float
    quality: TripletQuality
    rationale: Optional[str] = None
    failure_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input_query,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "confidence": self.confidence,
            "quality": self.quality.value,
            "rationale": self.rationale,
            "failure_reasons": self.failure_reasons,
            "metadata": self.metadata,
        }

    def to_slime_format(self) -> dict[str, Any]:
        """Convert to SLIME trainer format."""
        return {
            "prompt": self.input_query,
            "chosen": self._messages_to_text(self.chosen),
            "rejected": self._messages_to_text(self.rejected),
            "chosen_score": self.confidence,
            "rejected_score": 1.0 - self.confidence,
        }

    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        """Convert messages to text format."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)


@dataclass
class StudentModelConfig:
    """Configuration for student model deployment."""
    model_name: str
    model_path: Optional[str] = None
    quantization: Optional[str] = None  # "int8", "int4", "fp16"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)


# Preset student model configurations
DEEPSEEK_CODER_V2 = StudentModelConfig(
    model_name="deepseek-coder-v2",
    quantization="fp16",
    max_tokens=8192,
    temperature=0.3,
    metadata={"use_case": "OpenCoder"},
)

QWEN3_CODER = StudentModelConfig(
    model_name="qwen3-coder",
    quantization="int8",
    max_tokens=4096,
    temperature=0.5,
    metadata={"use_case": "OpenCloudInfra"},
)


class RationaleExtractor:
    """
    Extracts judge reasoning for training data enrichment.

    Following the STaR approach of fine-tuning on rationales
    that lead to correct outcomes.
    """

    def extract(
        self,
        judge_output: dict[str, Any],
    ) -> Optional[str]:
        """Extract rationale from judge output."""
        # Try different common formats
        rationale = judge_output.get("rationale")
        if rationale:
            return rationale

        rationale = judge_output.get("reasoning")
        if rationale:
            return rationale

        rationale = judge_output.get("explanation")
        if rationale:
            return rationale

        # Try to extract from structured response
        if "analysis" in judge_output:
            analysis = judge_output["analysis"]
            if isinstance(analysis, dict):
                parts = []
                for key, value in analysis.items():
                    if isinstance(value, str):
                        parts.append(f"{key}: {value}")
                if parts:
                    return "\n".join(parts)

        return None

    def format_for_training(
        self,
        rationale: str,
        include_chain_of_thought: bool = True,
    ) -> str:
        """Format rationale for training data."""
        if include_chain_of_thought:
            return f"<thinking>\n{rationale}\n</thinking>"
        return rationale


class QualityFilter:
    """
    Filters low-confidence or ambiguous comparisons.

    Ensures training data quality by excluding uncertain examples.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_score_diff: float = 0.1,
        require_rationale: bool = False,
        max_trajectory_length: int = 50,
    ):
        self.min_confidence = min_confidence
        self.min_score_diff = min_score_diff
        self.require_rationale = require_rationale
        self.max_trajectory_length = max_trajectory_length

    def filter(self, triplet: DatasetTriplet) -> tuple[bool, Optional[str]]:
        """
        Check if triplet passes quality filter.

        Returns (passed, reason_if_failed).
        """
        if triplet.confidence < self.min_confidence:
            return False, f"Confidence {triplet.confidence} below threshold {self.min_confidence}"

        if self.require_rationale and not triplet.rationale:
            return False, "Missing rationale"

        if len(triplet.chosen) > self.max_trajectory_length:
            return False, f"Chosen trajectory too long: {len(triplet.chosen)}"

        if len(triplet.rejected) > self.max_trajectory_length:
            return False, f"Rejected trajectory too long: {len(triplet.rejected)}"

        return True, None


class DistillationPipeline:
    """
    Converts tournament history into DPO training datasets.

    Pipeline stages:
    1. Extract comparisons from tournament history
    2. Filter by quality
    3. Enrich with rationales
    4. Format for target training framework
    """

    def __init__(
        self,
        quality_filter: Optional[QualityFilter] = None,
        rationale_extractor: Optional[RationaleExtractor] = None,
        target_format: DatasetFormat = DatasetFormat.SLIME,
    ):
        self.quality_filter = quality_filter or QualityFilter()
        self.rationale_extractor = rationale_extractor or RationaleExtractor()
        self.target_format = target_format
        self._triplets: list[DatasetTriplet] = []
        self._filtered_count = 0
        self._stats: dict[str, int] = {
            "total_processed": 0,
            "passed_filter": 0,
            "gold": 0,
            "silver": 0,
            "bronze": 0,
        }

    def process_tournament_result(
        self,
        input_query: str,
        trajectories: list[list[dict[str, Any]]],
        scores: list[float],
        judge_outputs: Optional[list[dict[str, Any]]] = None,
        verification_results: Optional[list[dict[str, Any]]] = None,
    ) -> list[DatasetTriplet]:
        """
        Process a tournament result into training triplets.

        Creates triplets from winner vs loser comparisons.
        """
        if len(trajectories) < 2:
            return []

        self._stats["total_processed"] += 1

        # Sort by score
        indexed = list(enumerate(zip(trajectories, scores)))
        indexed.sort(key=lambda x: x[1][1], reverse=True)

        triplets = []
        winner_idx, (winner_traj, winner_score) = indexed[0]

        # Create triplets: winner vs each loser
        for loser_idx, (loser_traj, loser_score) in indexed[1:]:
            score_diff = winner_score - loser_score

            # Determine quality based on score difference
            if score_diff > 0.5:
                quality = TripletQuality.GOLD
            elif score_diff > 0.2:
                quality = TripletQuality.SILVER
            else:
                quality = TripletQuality.BRONZE

            # Extract rationale if available
            rationale = None
            if judge_outputs:
                for output in judge_outputs:
                    extracted = self.rationale_extractor.extract(output)
                    if extracted:
                        rationale = extracted
                        break

            # Get failure reasons from verification
            failure_reasons = []
            if verification_results:
                for vr in verification_results:
                    if vr.get("index") == loser_idx and vr.get("status") == "failed":
                        failure_reasons.append(vr.get("message", "Unknown failure"))

            triplet = DatasetTriplet(
                id=str(uuid.uuid4()),
                input_query=input_query,
                chosen=winner_traj,
                rejected=loser_traj,
                confidence=min(score_diff + 0.5, 1.0),
                quality=quality,
                rationale=rationale,
                failure_reasons=failure_reasons,
                metadata={
                    "winner_score": winner_score,
                    "loser_score": loser_score,
                    "score_diff": score_diff,
                },
            )

            # Apply quality filter
            passed, reason = self.quality_filter.filter(triplet)
            if passed:
                self._triplets.append(triplet)
                triplets.append(triplet)
                self._stats["passed_filter"] += 1
                self._stats[quality.value] += 1
            else:
                self._filtered_count += 1

        return triplets

    def process_correction_trajectory(
        self,
        original_messages: list[dict[str, Any]],
        corrected_messages: list[dict[str, Any]],
        failure_reasons: list[str],
        num_attempts: int,
    ) -> Optional[DatasetTriplet]:
        """Process a self-correction trajectory into a training triplet."""
        if not original_messages or not corrected_messages:
            return None

        # Extract input query from first user message
        input_query = ""
        for msg in original_messages:
            if msg.get("role") == "user":
                input_query = msg.get("content", "")
                break

        # Corrections are high quality since they were verified
        triplet = DatasetTriplet(
            id=str(uuid.uuid4()),
            input_query=input_query,
            chosen=corrected_messages,
            rejected=original_messages,
            confidence=0.9,  # High confidence for verified corrections
            quality=TripletQuality.SILVER,
            failure_reasons=failure_reasons,
            metadata={
                "source": "self_correction",
                "correction_attempts": num_attempts,
            },
        )

        passed, _ = self.quality_filter.filter(triplet)
        if passed:
            self._triplets.append(triplet)
            self._stats["passed_filter"] += 1
            self._stats["silver"] += 1
            return triplet

        return None

    def export(self, format: Optional[DatasetFormat] = None) -> Any:
        """Export triplets in the specified format."""
        format = format or self.target_format

        if format == DatasetFormat.JSONL:
            return self._export_jsonl()
        elif format == DatasetFormat.SLIME:
            return self._export_slime()
        elif format == DatasetFormat.HF_DATASET:
            return self._export_hf()
        else:
            return [t.to_dict() for t in self._triplets]

    def _export_jsonl(self) -> str:
        """Export as JSONL format."""
        lines = [json.dumps(t.to_dict()) for t in self._triplets]
        return "\n".join(lines)

    def _export_slime(self) -> list[dict[str, Any]]:
        """Export in SLIME trainer format."""
        return [t.to_slime_format() for t in self._triplets]

    def _export_hf(self) -> dict[str, list]:
        """Export in HuggingFace datasets format."""
        return {
            "prompt": [t.input_query for t in self._triplets],
            "chosen": [t._messages_to_text(t.chosen) for t in self._triplets],
            "rejected": [t._messages_to_text(t.rejected) for t in self._triplets],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "total_triplets": len(self._triplets),
            "filtered_out": self._filtered_count,
        }

    def clear(self) -> None:
        """Clear all triplets."""
        self._triplets = []
        self._filtered_count = 0
        for key in self._stats:
            self._stats[key] = 0


class SlimeTrainerAdapter:
    """
    Adapter for integrating with the SLIME trainer.

    Provides interface to export training data in SLIME format
    and trigger training runs.
    """

    def __init__(
        self,
        output_dir: str = "./training_data",
        model_config: Optional[StudentModelConfig] = None,
    ):
        self.output_dir = output_dir
        self.model_config = model_config or DEEPSEEK_CODER_V2

    def prepare_dataset(
        self,
        triplets: list[DatasetTriplet],
        split_ratio: float = 0.9,
    ) -> dict[str, list[dict[str, Any]]]:
        """Prepare dataset with train/eval split."""
        import random

        shuffled = triplets.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * split_ratio)
        train = shuffled[:split_idx]
        eval_set = shuffled[split_idx:]

        return {
            "train": [t.to_slime_format() for t in train],
            "eval": [t.to_slime_format() for t in eval_set],
        }

    def generate_config(self) -> dict[str, Any]:
        """Generate SLIME training configuration."""
        return {
            "model": {
                "name": self.model_config.model_name,
                "path": self.model_config.model_path,
                "quantization": self.model_config.quantization,
            },
            "training": {
                "batch_size": self.model_config.batch_size,
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "warmup_steps": 100,
                "gradient_accumulation_steps": 4,
            },
            "dpo": {
                "beta": 0.1,
                "loss_type": "sigmoid",
            },
            "output_dir": self.output_dir,
        }

    def estimate_training_time(
        self,
        num_triplets: int,
        gpu_type: str = "H100",
    ) -> dict[str, float]:
        """Estimate training time and cost."""
        # Rough estimates based on typical training speeds
        tokens_per_triplet = 2000  # Average
        total_tokens = num_triplets * tokens_per_triplet

        # Tokens per second by GPU type
        tps = {
            "H100": 50000,
            "A100": 30000,
            "A10": 15000,
            "T4": 5000,
        }

        speed = tps.get(gpu_type, 10000)
        hours = total_tokens / speed / 3600

        # Cost per hour by GPU type (rough cloud prices)
        cost_per_hour = {
            "H100": 4.0,
            "A100": 2.0,
            "A10": 1.0,
            "T4": 0.35,
        }

        return {
            "estimated_hours": hours,
            "estimated_cost_usd": hours * cost_per_hour.get(gpu_type, 1.0),
            "total_tokens": total_tokens,
            "gpu_type": gpu_type,
        }
