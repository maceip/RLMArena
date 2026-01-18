"""
Flywheel Worker for Recursive Language Model Training

This module implements the "Recursive Distillation Factory" - a background worker
that converts ShadowArena results into DPO (Direct Preference Optimization) format
for continuous model improvement.

The flywheel:
1. Scrapes ShadowArena logs periodically
2. Converts verification results into preference pairs
3. Buffers training data in DPO format
4. Kicks off fine-tuning runs for local models (e.g., OpenCoder)

This creates the "Expert-in-the-Loop" data flywheel where:
- Tournament winners become the "chosen" response
- Failed/lower-scored responses become "rejected"
- Verification results provide hard labels
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingStatus(Enum):
    """Status of training samples."""
    PENDING = "pending"
    PROCESSED = "processed"
    EXPORTED = "exported"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class PreferenceSource(Enum):
    """Source of preference labels."""
    VERIFICATION = "verification"  # Hard check pass/fail
    TOURNAMENT = "tournament"  # LLM judge ranking
    HUMAN = "human"  # Expert annotation
    HYBRID = "hybrid"  # Combination


@dataclass
class PreferencePair:
    """A single preference pair for DPO training."""
    id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    preference_source: PreferenceSource
    verification_label: Optional[bool] = None  # Hard label from verification
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "preference_source": self.preference_source.value,
            "verification_label": self.verification_label,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def to_dpo_format(self) -> dict:
        """Convert to standard DPO training format."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }

    def to_openai_format(self) -> dict:
        """Convert to OpenAI fine-tuning format."""
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.chosen},
            ],
            "rejected_messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.rejected},
            ],
        }


@dataclass
class TrainingBatch:
    """A batch of preference pairs for training."""
    id: str
    pairs: list[PreferencePair]
    model_name: str
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pair_count": len(self.pairs),
            "model_name": self.model_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics,
        }


class TrainingBuffer:
    """
    Persistent buffer for training data.

    Uses SQLite for durability and efficient querying.
    """

    def __init__(self, db_path: str = "training_buffer.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_pairs (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                chosen TEXT NOT NULL,
                rejected TEXT NOT NULL,
                chosen_score REAL,
                rejected_score REAL,
                preference_source TEXT,
                verification_label INTEGER,
                metadata TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                processed_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_batches (
                id TEXT PRIMARY KEY,
                model_name TEXT,
                pair_ids TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                completed_at TEXT,
                metrics TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arena_logs (
                id TEXT PRIMARY KEY,
                query TEXT,
                result_json TEXT,
                processed INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pairs_status ON preference_pairs(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_logs_processed ON arena_logs(processed)
        """)

        conn.commit()
        conn.close()

    def add_arena_log(self, query: str, result: dict) -> str:
        """Add a ShadowArena result to the log."""
        log_id = hashlib.sha256(
            f"{query}{json.dumps(result)}{time.time()}".encode()
        ).hexdigest()[:16]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO arena_logs (id, query, result_json, created_at)
            VALUES (?, ?, ?, ?)
        """, (log_id, query, json.dumps(result), datetime.utcnow().isoformat()))

        conn.commit()
        conn.close()

        return log_id

    def get_unprocessed_logs(self, limit: int = 100) -> list[tuple[str, str, dict]]:
        """Get unprocessed arena logs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, query, result_json FROM arena_logs
            WHERE processed = 0
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append((row[0], row[1], json.loads(row[2])))

        conn.close()
        return results

    def mark_logs_processed(self, log_ids: list[str]) -> None:
        """Mark logs as processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executemany("""
            UPDATE arena_logs SET processed = 1 WHERE id = ?
        """, [(log_id,) for log_id in log_ids])

        conn.commit()
        conn.close()

    def add_preference_pair(self, pair: PreferencePair) -> None:
        """Add a preference pair to the buffer."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO preference_pairs
            (id, prompt, chosen, rejected, chosen_score, rejected_score,
             preference_source, verification_label, metadata, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair.id,
            pair.prompt,
            pair.chosen,
            pair.rejected,
            pair.chosen_score,
            pair.rejected_score,
            pair.preference_source.value,
            1 if pair.verification_label else 0 if pair.verification_label is not None else None,
            json.dumps(pair.metadata),
            TrainingStatus.PENDING.value,
            pair.created_at.isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_pending_pairs(self, limit: int = 1000) -> list[PreferencePair]:
        """Get pending preference pairs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, prompt, chosen, rejected, chosen_score, rejected_score,
                   preference_source, verification_label, metadata, created_at
            FROM preference_pairs
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,))

        pairs = []
        for row in cursor.fetchall():
            pairs.append(PreferencePair(
                id=row[0],
                prompt=row[1],
                chosen=row[2],
                rejected=row[3],
                chosen_score=row[4],
                rejected_score=row[5],
                preference_source=PreferenceSource(row[6]),
                verification_label=bool(row[7]) if row[7] is not None else None,
                metadata=json.loads(row[8]) if row[8] else {},
                created_at=datetime.fromisoformat(row[9]),
            ))

        conn.close()
        return pairs

    def update_pair_status(self, pair_ids: list[str], status: TrainingStatus) -> None:
        """Update status of preference pairs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executemany("""
            UPDATE preference_pairs
            SET status = ?, processed_at = ?
            WHERE id = ?
        """, [(status.value, datetime.utcnow().isoformat(), pid) for pid in pair_ids])

        conn.commit()
        conn.close()

    def get_buffer_stats(self) -> dict:
        """Get buffer statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*) FROM preference_pairs GROUP BY status
        """)
        stats["pairs_by_status"] = dict(cursor.fetchall())

        # Total logs
        cursor.execute("SELECT COUNT(*) FROM arena_logs")
        stats["total_logs"] = cursor.fetchone()[0]

        # Unprocessed logs
        cursor.execute("SELECT COUNT(*) FROM arena_logs WHERE processed = 0")
        stats["unprocessed_logs"] = cursor.fetchone()[0]

        # Average scores
        cursor.execute("""
            SELECT AVG(chosen_score), AVG(rejected_score)
            FROM preference_pairs
        """)
        row = cursor.fetchone()
        stats["avg_chosen_score"] = row[0]
        stats["avg_rejected_score"] = row[1]

        conn.close()
        return stats


class ArenaLogConverter:
    """
    Converts ShadowArena results to preference pairs.

    Uses verification results as hard labels and tournament scores
    as soft labels for preference learning.
    """

    def __init__(
        self,
        min_score_gap: float = 0.1,
        prefer_verification_labels: bool = True,
    ):
        self.min_score_gap = min_score_gap
        self.prefer_verification_labels = prefer_verification_labels

    def convert(self, query: str, result: dict) -> list[PreferencePair]:
        """
        Convert a ShadowArena result to preference pairs.

        Creates pairs from:
        1. Best vs all rejected (verification-based)
        2. Pairwise comparisons from tournament (score-based)
        """
        pairs = []

        all_responses = result.get("all_responses", [])
        verification_results = result.get("verification_results", [])
        tournament_scores = result.get("tournament_scores", [])
        best_response = result.get("best_response")

        if not all_responses or not best_response:
            return pairs

        # Extract response contents
        responses_content = []
        for trajectory in all_responses:
            content = ""
            for msg in reversed(trajectory):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    break
            responses_content.append(content)

        best_content = best_response.get("content", "")
        if not best_content:
            return pairs

        # Create verification-based pairs (hard labels)
        if self.prefer_verification_labels and verification_results:
            valid_indices = [
                i for i, v in enumerate(verification_results)
                if isinstance(v, dict) and v.get("all_passed", False)
            ]
            invalid_indices = [
                i for i, v in enumerate(verification_results)
                if isinstance(v, dict) and not v.get("all_passed", False)
            ]

            # Best (verified) vs each failed response
            for idx in invalid_indices:
                if idx < len(responses_content) and responses_content[idx]:
                    pair_id = hashlib.sha256(
                        f"{query}{best_content}{responses_content[idx]}".encode()
                    ).hexdigest()[:16]

                    pairs.append(PreferencePair(
                        id=pair_id,
                        prompt=query,
                        chosen=best_content,
                        rejected=responses_content[idx],
                        chosen_score=1.0,
                        rejected_score=-1.0,
                        preference_source=PreferenceSource.VERIFICATION,
                        verification_label=True,
                        metadata={
                            "chosen_verified": True,
                            "rejected_verified": False,
                            "verification_failures": verification_results[idx].get("failure_messages", []) if isinstance(verification_results[idx], dict) else [],
                        },
                    ))

        # Create tournament-based pairs (soft labels)
        if tournament_scores:
            scored_responses = list(zip(responses_content, tournament_scores))
            scored_responses.sort(key=lambda x: x[1], reverse=True)

            for i in range(len(scored_responses)):
                for j in range(i + 1, len(scored_responses)):
                    content_i, score_i = scored_responses[i]
                    content_j, score_j = scored_responses[j]

                    if score_i - score_j < self.min_score_gap:
                        continue

                    if not content_i or not content_j:
                        continue

                    pair_id = hashlib.sha256(
                        f"{query}{content_i}{content_j}".encode()
                    ).hexdigest()[:16]

                    pairs.append(PreferencePair(
                        id=pair_id,
                        prompt=query,
                        chosen=content_i,
                        rejected=content_j,
                        chosen_score=score_i,
                        rejected_score=score_j,
                        preference_source=PreferenceSource.TOURNAMENT,
                        metadata={
                            "score_gap": score_i - score_j,
                        },
                    ))

        return pairs


class FlywheelWorker:
    """
    Background worker for the recursive distillation flywheel.

    Periodically:
    1. Scrapes new ShadowArena logs
    2. Converts to DPO preference pairs
    3. Buffers training data
    4. Triggers fine-tuning when batch is ready
    """

    def __init__(
        self,
        buffer: TrainingBuffer,
        converter: Optional[ArenaLogConverter] = None,
        batch_size: int = 1000,
        processing_interval: float = 60.0,
        training_trigger_size: int = 5000,
        training_callback: Optional[Callable[[TrainingBatch], None]] = None,
    ):
        self.buffer = buffer
        self.converter = converter or ArenaLogConverter()
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.training_trigger_size = training_trigger_size
        self.training_callback = training_callback
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_count = 0
        self._pair_count = 0

    async def start(self) -> None:
        """Start the flywheel worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the flywheel worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                await self._process_logs()
                await self._check_training_trigger()
                await asyncio.sleep(self.processing_interval)
            except Exception as e:
                # Log error but continue
                print(f"Flywheel error: {e}")
                await asyncio.sleep(self.processing_interval)

    async def _process_logs(self) -> None:
        """Process unprocessed arena logs."""
        logs = self.buffer.get_unprocessed_logs(limit=self.batch_size)

        if not logs:
            return

        processed_ids = []
        new_pairs = []

        for log_id, query, result in logs:
            pairs = self.converter.convert(query, result)
            new_pairs.extend(pairs)
            processed_ids.append(log_id)

        # Add pairs to buffer
        for pair in new_pairs:
            self.buffer.add_preference_pair(pair)

        # Mark logs as processed
        self.buffer.mark_logs_processed(processed_ids)

        self._processed_count += len(processed_ids)
        self._pair_count += len(new_pairs)

    async def _check_training_trigger(self) -> None:
        """Check if we should trigger a training run."""
        pending_pairs = self.buffer.get_pending_pairs(limit=self.training_trigger_size)

        if len(pending_pairs) >= self.training_trigger_size:
            await self._trigger_training(pending_pairs)

    async def _trigger_training(self, pairs: list[PreferencePair]) -> None:
        """Trigger a training run with the accumulated pairs."""
        batch_id = hashlib.sha256(
            f"batch_{time.time()}".encode()
        ).hexdigest()[:16]

        batch = TrainingBatch(
            id=batch_id,
            pairs=pairs,
            model_name="opencoder",  # Default model
            status=TrainingStatus.PENDING,
        )

        # Mark pairs as being trained
        pair_ids = [p.id for p in pairs]
        self.buffer.update_pair_status(pair_ids, TrainingStatus.TRAINING)

        if self.training_callback:
            try:
                self.training_callback(batch)
                self.buffer.update_pair_status(pair_ids, TrainingStatus.COMPLETED)
            except Exception as e:
                self.buffer.update_pair_status(pair_ids, TrainingStatus.FAILED)
                print(f"Training failed: {e}")

    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            "running": self._running,
            "processed_logs": self._processed_count,
            "generated_pairs": self._pair_count,
            "buffer_stats": self.buffer.get_buffer_stats(),
        }


class DPOExporter:
    """
    Exports preference pairs to various training formats.
    """

    @staticmethod
    def to_jsonl(pairs: list[PreferencePair], output_path: str) -> int:
        """Export to JSONL format (standard DPO)."""
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dpo_format()) + "\n")
        return len(pairs)

    @staticmethod
    def to_openai_jsonl(pairs: list[PreferencePair], output_path: str) -> int:
        """Export to OpenAI fine-tuning format."""
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_openai_format()) + "\n")
        return len(pairs)

    @staticmethod
    def to_huggingface_dataset(
        pairs: list[PreferencePair],
        output_path: str,
    ) -> int:
        """Export to HuggingFace datasets format."""
        try:
            from datasets import Dataset

            data = {
                "prompt": [p.prompt for p in pairs],
                "chosen": [p.chosen for p in pairs],
                "rejected": [p.rejected for p in pairs],
                "chosen_score": [p.chosen_score for p in pairs],
                "rejected_score": [p.rejected_score for p in pairs],
            }

            dataset = Dataset.from_dict(data)
            dataset.save_to_disk(output_path)
            return len(pairs)
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")

    @staticmethod
    def to_torch_dataset(
        pairs: list[PreferencePair],
        tokenizer: Any,
        max_length: int = 2048,
    ) -> Any:
        """Create a PyTorch dataset for DPO training."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        class DPODataset(torch.utils.data.Dataset):
            def __init__(self, pairs, tokenizer, max_length):
                self.pairs = pairs
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                pair = self.pairs[idx]

                prompt_chosen = f"{pair.prompt}\n\n{pair.chosen}"
                prompt_rejected = f"{pair.prompt}\n\n{pair.rejected}"

                chosen_encoding = self.tokenizer(
                    prompt_chosen,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                rejected_encoding = self.tokenizer(
                    prompt_rejected,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                return {
                    "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
                    "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
                    "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
                    "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(),
                }

        return DPODataset(pairs, tokenizer, max_length)


def create_flywheel(
    db_path: str = "training_buffer.db",
    batch_size: int = 1000,
    training_trigger_size: int = 5000,
    training_callback: Optional[Callable] = None,
) -> FlywheelWorker:
    """
    Create a configured flywheel worker.

    Args:
        db_path: Path to SQLite database
        batch_size: Number of logs to process per iteration
        training_trigger_size: Number of pairs before triggering training
        training_callback: Callback function for training

    Returns:
        Configured FlywheelWorker instance
    """
    buffer = TrainingBuffer(db_path)
    converter = ArenaLogConverter()

    return FlywheelWorker(
        buffer=buffer,
        converter=converter,
        batch_size=batch_size,
        training_trigger_size=training_trigger_size,
        training_callback=training_callback,
    )


# Integration with ShadowArena
class ShadowArenaHook:
    """
    Hook to automatically log ShadowArena results to the training buffer.

    Usage:
        buffer = TrainingBuffer()
        hook = ShadowArenaHook(buffer)

        # In your ShadowArena processing:
        result = await arena.process_query(query, context)
        hook.log_result(query, result)
    """

    def __init__(self, buffer: TrainingBuffer):
        self.buffer = buffer

    def log_result(self, query: str, result: Any) -> str:
        """Log a ShadowArena result."""
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        else:
            result_dict = result

        return self.buffer.add_arena_log(query, result_dict)
