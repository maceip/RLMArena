"""
Expert Aligner Service for SME DNA Capture.

This module implements the formal SME rationalization protocol that captures
Subject Matter Expert reasoning into structured schemas for judge fine-tuning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path


class FailureCategory(Enum):
    """Categories of trajectory failures identified by SMEs."""
    SYNTAX_ERROR = "syntax_error"
    SECURITY_VIOLATION = "security_violation"
    POLICY_BREACH = "policy_breach"
    LOGIC_ERROR = "logic_error"
    RESOURCE_VIOLATION = "resource_violation"
    OUTPUT_MALFORMED = "output_malformed"
    TOOL_MISUSE = "tool_misuse"
    CREDENTIAL_LEAK = "credential_leak"
    PERFORMANCE_ISSUE = "performance_issue"
    COMPLIANCE_FAILURE = "compliance_failure"


@dataclass
class TrajectoryPoint:
    """
    Captures the exact location in a trajectory where a failure occurred.

    This enables fine-grained error attribution for training data generation.
    """
    message_index: int
    tool_call_index: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    content_snippet: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_index": self.message_index,
            "tool_call_index": self.tool_call_index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "content_snippet": self.content_snippet,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryPoint":
        return cls(
            message_index=data["message_index"],
            tool_call_index=data.get("tool_call_index"),
            token_start=data.get("token_start"),
            token_end=data.get("token_end"),
            content_snippet=data.get("content_snippet"),
        )


@dataclass
class TechnicalRationale:
    """
    Structured technical reasoning for why a trajectory failed or succeeded.

    Examples:
    - "Violates Python AST depth constraints"
    - "Implicitly shares production credentials"
    - "Terraform resource lacks encryption configuration"
    """
    category: FailureCategory
    description: str
    severity: int  # 1-10 scale
    rule_id: Optional[str] = None
    evidence: Optional[str] = None
    suggested_fix: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "evidence": self.evidence,
            "suggested_fix": self.suggested_fix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TechnicalRationale":
        return cls(
            category=FailureCategory(data["category"]),
            description=data["description"],
            severity=data["severity"],
            rule_id=data.get("rule_id"),
            evidence=data.get("evidence"),
            suggested_fix=data.get("suggested_fix"),
        )


@dataclass
class SMELabel:
    """
    Complete SME label for a trajectory comparison.

    Captures which trajectory won, where failures occurred, and the reasoning.
    """
    id: str
    sme_id: str
    timestamp: datetime
    trajectory_a_id: str
    trajectory_b_id: str
    winner: str  # "a", "b", or "tie"
    confidence: float  # 0.0 to 1.0
    failure_points: list[TrajectoryPoint] = field(default_factory=list)
    rationales: list[TechnicalRationale] = field(default_factory=list)
    notes: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sme_id": self.sme_id,
            "timestamp": self.timestamp.isoformat(),
            "trajectory_a_id": self.trajectory_a_id,
            "trajectory_b_id": self.trajectory_b_id,
            "winner": self.winner,
            "confidence": self.confidence,
            "failure_points": [fp.to_dict() for fp in self.failure_points],
            "rationales": [r.to_dict() for r in self.rationales],
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SMELabel":
        return cls(
            id=data["id"],
            sme_id=data["sme_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            trajectory_a_id=data["trajectory_a_id"],
            trajectory_b_id=data["trajectory_b_id"],
            winner=data["winner"],
            confidence=data["confidence"],
            failure_points=[TrajectoryPoint.from_dict(fp) for fp in data.get("failure_points", [])],
            rationales=[TechnicalRationale.from_dict(r) for r in data.get("rationales", [])],
            notes=data.get("notes"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GoldComparison:
    """
    A gold-standard trajectory comparison for judge training.

    Contains the full trajectory data along with SME labels.
    """
    id: str
    input_query: str
    trajectory_a: list[dict[str, Any]]
    trajectory_b: list[dict[str, Any]]
    sme_labels: list[SMELabel]
    consensus_winner: Optional[str] = None
    consensus_confidence: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def compute_consensus(self) -> tuple[Optional[str], Optional[float]]:
        """Compute consensus winner from multiple SME labels."""
        if not self.sme_labels:
            return None, None

        votes = {"a": 0.0, "b": 0.0, "tie": 0.0}
        total_confidence = 0.0

        for label in self.sme_labels:
            votes[label.winner] += label.confidence
            total_confidence += label.confidence

        if total_confidence == 0:
            return None, None

        # Normalize votes by confidence
        winner = max(votes, key=votes.get)
        confidence = votes[winner] / total_confidence

        return winner, confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_query": self.input_query,
            "trajectory_a": self.trajectory_a,
            "trajectory_b": self.trajectory_b,
            "sme_labels": [label.to_dict() for label in self.sme_labels],
            "consensus_winner": self.consensus_winner,
            "consensus_confidence": self.consensus_confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldComparison":
        return cls(
            id=data["id"],
            input_query=data["input_query"],
            trajectory_a=data["trajectory_a"],
            trajectory_b=data["trajectory_b"],
            sme_labels=[SMELabel.from_dict(label) for label in data.get("sme_labels", [])],
            consensus_winner=data.get("consensus_winner"),
            consensus_confidence=data.get("consensus_confidence"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
        )


class ExpertAlignerService:
    """
    Service for ingesting and managing SME labels for judge fine-tuning.

    This service:
    1. Ingests SME labels with trajectory points and rationales
    2. Manages the gold comparison training set
    3. Exports data for judge fine-tuning
    4. Computes consensus across multiple SME reviewers
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the Expert Aligner Service with optional SQLite storage."""
        self.db_path = db_path or ":memory:"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gold_comparisons (
                id TEXT PRIMARY KEY,
                input_query TEXT NOT NULL,
                trajectory_a TEXT NOT NULL,
                trajectory_b TEXT NOT NULL,
                consensus_winner TEXT,
                consensus_confidence REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sme_labels (
                id TEXT PRIMARY KEY,
                comparison_id TEXT NOT NULL,
                sme_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                trajectory_a_id TEXT NOT NULL,
                trajectory_b_id TEXT NOT NULL,
                winner TEXT NOT NULL,
                confidence REAL NOT NULL,
                failure_points TEXT NOT NULL,
                rationales TEXT NOT NULL,
                notes TEXT,
                metadata TEXT,
                FOREIGN KEY (comparison_id) REFERENCES gold_comparisons(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sme_labels_comparison
            ON sme_labels(comparison_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sme_labels_sme
            ON sme_labels(sme_id)
        """)

        conn.commit()
        # Don't close in-memory connections
        if self.db_path != ":memory:":
            self._close_conn(conn)

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection. For :memory: DBs, reuse connection."""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
            return self._conn
        return sqlite3.connect(self.db_path)

    def _close_conn(self, conn: sqlite3.Connection) -> None:
        """Close connection if not an in-memory shared connection."""
        if self.db_path != ":memory:":
            self._close_conn(conn)

    def create_gold_comparison(
        self,
        input_query: str,
        trajectory_a: list[dict[str, Any]],
        trajectory_b: list[dict[str, Any]],
    ) -> GoldComparison:
        """Create a new gold comparison for SME labeling."""
        comparison_id = str(uuid.uuid4())

        comparison = GoldComparison(
            id=comparison_id,
            input_query=input_query,
            trajectory_a=trajectory_a,
            trajectory_b=trajectory_b,
            sme_labels=[],
        )

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO gold_comparisons
            (id, input_query, trajectory_a, trajectory_b, consensus_winner,
             consensus_confidence, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comparison.id,
                comparison.input_query,
                json.dumps(comparison.trajectory_a),
                json.dumps(comparison.trajectory_b),
                comparison.consensus_winner,
                comparison.consensus_confidence,
                comparison.created_at.isoformat(),
                comparison.updated_at.isoformat(),
            )
        )

        conn.commit()
        self._close_conn(conn)

        return comparison

    def add_sme_label(
        self,
        comparison_id: str,
        sme_id: str,
        winner: str,
        confidence: float,
        failure_points: Optional[list[TrajectoryPoint]] = None,
        rationales: Optional[list[TechnicalRationale]] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SMELabel:
        """Add an SME label to an existing gold comparison."""
        if winner not in ("a", "b", "tie"):
            raise ValueError(f"Winner must be 'a', 'b', or 'tie', got {winner}")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        label = SMELabel(
            id=str(uuid.uuid4()),
            sme_id=sme_id,
            timestamp=datetime.utcnow(),
            trajectory_a_id=f"{comparison_id}_a",
            trajectory_b_id=f"{comparison_id}_b",
            winner=winner,
            confidence=confidence,
            failure_points=failure_points or [],
            rationales=rationales or [],
            notes=notes,
            metadata=metadata or {},
        )

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sme_labels
            (id, comparison_id, sme_id, timestamp, trajectory_a_id, trajectory_b_id,
             winner, confidence, failure_points, rationales, notes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                label.id,
                comparison_id,
                label.sme_id,
                label.timestamp.isoformat(),
                label.trajectory_a_id,
                label.trajectory_b_id,
                label.winner,
                label.confidence,
                json.dumps([fp.to_dict() for fp in label.failure_points]),
                json.dumps([r.to_dict() for r in label.rationales]),
                label.notes,
                json.dumps(label.metadata),
            )
        )

        # Update consensus
        self._update_consensus(comparison_id, cursor)

        conn.commit()
        self._close_conn(conn)

        return label

    def _update_consensus(self, comparison_id: str, cursor: sqlite3.Cursor) -> None:
        """Update consensus for a comparison after new label."""
        cursor.execute(
            "SELECT winner, confidence FROM sme_labels WHERE comparison_id = ?",
            (comparison_id,)
        )

        votes = {"a": 0.0, "b": 0.0, "tie": 0.0}
        total_confidence = 0.0

        for winner, confidence in cursor.fetchall():
            votes[winner] += confidence
            total_confidence += confidence

        if total_confidence > 0:
            consensus_winner = max(votes, key=votes.get)
            consensus_confidence = votes[consensus_winner] / total_confidence

            cursor.execute(
                """
                UPDATE gold_comparisons
                SET consensus_winner = ?, consensus_confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (consensus_winner, consensus_confidence, datetime.utcnow().isoformat(), comparison_id)
            )

    def get_comparison(self, comparison_id: str) -> Optional[GoldComparison]:
        """Retrieve a gold comparison by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM gold_comparisons WHERE id = ?",
            (comparison_id,)
        )
        row = cursor.fetchone()

        if not row:
            self._close_conn(conn)
            return None

        # Get labels
        cursor.execute(
            "SELECT * FROM sme_labels WHERE comparison_id = ?",
            (comparison_id,)
        )
        label_rows = cursor.fetchall()

        self._close_conn(conn)

        labels = []
        for lrow in label_rows:
            labels.append(SMELabel(
                id=lrow[0],
                sme_id=lrow[2],
                timestamp=datetime.fromisoformat(lrow[3]),
                trajectory_a_id=lrow[4],
                trajectory_b_id=lrow[5],
                winner=lrow[6],
                confidence=lrow[7],
                failure_points=[TrajectoryPoint.from_dict(fp) for fp in json.loads(lrow[8])],
                rationales=[TechnicalRationale.from_dict(r) for r in json.loads(lrow[9])],
                notes=lrow[10],
                metadata=json.loads(lrow[11]) if lrow[11] else {},
            ))

        return GoldComparison(
            id=row[0],
            input_query=row[1],
            trajectory_a=json.loads(row[2]),
            trajectory_b=json.loads(row[3]),
            sme_labels=labels,
            consensus_winner=row[4],
            consensus_confidence=row[5],
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
        )

    def list_comparisons(
        self,
        min_confidence: Optional[float] = None,
        winner: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GoldComparison]:
        """List gold comparisons with optional filtering."""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT id FROM gold_comparisons WHERE 1=1"
        params = []

        if min_confidence is not None:
            query += " AND consensus_confidence >= ?"
            params.append(min_confidence)

        if winner is not None:
            query += " AND consensus_winner = ?"
            params.append(winner)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        ids = [row[0] for row in cursor.fetchall()]
        self._close_conn(conn)

        return [self.get_comparison(id) for id in ids if self.get_comparison(id)]

    def export_for_training(
        self,
        min_confidence: float = 0.7,
        format: str = "jsonl",
    ) -> list[dict[str, Any]]:
        """
        Export gold comparisons for judge fine-tuning.

        Returns training data in format suitable for DPO/SFT training.
        """
        comparisons = self.list_comparisons(min_confidence=min_confidence, limit=10000)

        training_data = []
        for comp in comparisons:
            if not comp.consensus_winner or comp.consensus_winner == "tie":
                continue

            # Collect all rationales across labels
            all_rationales = []
            for label in comp.sme_labels:
                all_rationales.extend(label.rationales)

            # Format for training
            winner_traj = comp.trajectory_a if comp.consensus_winner == "a" else comp.trajectory_b
            loser_traj = comp.trajectory_b if comp.consensus_winner == "a" else comp.trajectory_a

            training_data.append({
                "id": comp.id,
                "input": comp.input_query,
                "chosen": winner_traj,
                "rejected": loser_traj,
                "confidence": comp.consensus_confidence,
                "rationales": [r.to_dict() for r in all_rationales],
                "metadata": {
                    "num_labels": len(comp.sme_labels),
                    "created_at": comp.created_at.isoformat(),
                },
            })

        return training_data

    def export_rationales(self) -> list[dict[str, Any]]:
        """Export all technical rationales for rule extraction."""
        comparisons = self.list_comparisons(limit=10000)

        rationales = []
        for comp in comparisons:
            for label in comp.sme_labels:
                for rationale in label.rationales:
                    rationales.append({
                        "comparison_id": comp.id,
                        "sme_id": label.sme_id,
                        "category": rationale.category.value,
                        "description": rationale.description,
                        "severity": rationale.severity,
                        "rule_id": rationale.rule_id,
                        "evidence": rationale.evidence,
                        "suggested_fix": rationale.suggested_fix,
                    })

        return rationales

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the gold comparison dataset."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM gold_comparisons")
        total_comparisons = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sme_labels")
        total_labels = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM gold_comparisons WHERE consensus_confidence >= 0.7"
        )
        high_confidence = cursor.fetchone()[0]

        cursor.execute(
            "SELECT consensus_winner, COUNT(*) FROM gold_comparisons GROUP BY consensus_winner"
        )
        winner_distribution = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(DISTINCT sme_id) FROM sme_labels")
        unique_smes = cursor.fetchone()[0]

        self._close_conn(conn)

        return {
            "total_comparisons": total_comparisons,
            "total_labels": total_labels,
            "high_confidence_comparisons": high_confidence,
            "winner_distribution": winner_distribution,
            "unique_smes": unique_smes,
            "avg_labels_per_comparison": total_labels / max(total_comparisons, 1),
        }

    def compute_trajectory_hash(self, trajectory: list[dict[str, Any]]) -> str:
        """Compute a hash for trajectory deduplication."""
        content = json.dumps(trajectory, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
