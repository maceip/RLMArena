"""
Training Data Store for persistent storage of distillation datasets.

Provides SQLite-based storage with export capabilities for JSONL and Parquet.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterator, Optional
import json
import sqlite3
import uuid


class StoreStatus(Enum):
    """Status of a stored item."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class StoredTriplet:
    """A triplet stored in the training store."""
    id: str
    input_query: str
    chosen: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    confidence: float
    quality: str
    rationale: Optional[str]
    failure_reasons: list[str]
    source: str
    status: StoreStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_query": self.input_query,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "confidence": self.confidence,
            "quality": self.quality,
            "rationale": self.rationale,
            "failure_reasons": self.failure_reasons,
            "source": self.source,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DatasetVersion:
    """A versioned snapshot of the training dataset."""
    version_id: str
    name: str
    description: Optional[str]
    triplet_count: int
    created_at: datetime
    created_by: Optional[str]
    filters_applied: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingStore:
    """
    Persistent storage for training triplets and dataset versions.

    Features:
    - SQLite-based storage
    - Dataset versioning
    - Quality-based filtering
    - Export to JSONL/Parquet
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triplets (
                id TEXT PRIMARY KEY,
                input_query TEXT NOT NULL,
                chosen TEXT NOT NULL,
                rejected TEXT NOT NULL,
                confidence REAL NOT NULL,
                quality TEXT NOT NULL,
                rationale TEXT,
                failure_reasons TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_versions (
                version_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                triplet_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT,
                filters_applied TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_triplets (
                version_id TEXT NOT NULL,
                triplet_id TEXT NOT NULL,
                PRIMARY KEY (version_id, triplet_id),
                FOREIGN KEY (version_id) REFERENCES dataset_versions(version_id),
                FOREIGN KEY (triplet_id) REFERENCES triplets(id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_triplets_quality ON triplets(quality)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_triplets_source ON triplets(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_triplets_status ON triplets(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_triplets_confidence ON triplets(confidence)")

        conn.commit()
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

    def add_triplet(
        self,
        input_query: str,
        chosen: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        confidence: float,
        quality: str = "bronze",
        rationale: Optional[str] = None,
        failure_reasons: Optional[list[str]] = None,
        source: str = "tournament",
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a triplet to the store."""
        triplet_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO triplets
            (id, input_query, chosen, rejected, confidence, quality, rationale,
             failure_reasons, source, status, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                triplet_id,
                input_query,
                json.dumps(chosen),
                json.dumps(rejected),
                confidence,
                quality,
                rationale,
                json.dumps(failure_reasons or []),
                source,
                StoreStatus.ACTIVE.value,
                now,
                now,
                json.dumps(metadata or {}),
            )
        )

        conn.commit()
        self._close_conn(conn)

        return triplet_id

    def add_triplets_batch(
        self,
        triplets: list[dict[str, Any]],
    ) -> list[str]:
        """Add multiple triplets in a batch."""
        conn = self._get_conn()
        cursor = conn.cursor()

        ids = []
        now = datetime.utcnow().isoformat()

        for triplet in triplets:
            triplet_id = str(uuid.uuid4())
            ids.append(triplet_id)

            cursor.execute(
                """
                INSERT INTO triplets
                (id, input_query, chosen, rejected, confidence, quality, rationale,
                 failure_reasons, source, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    triplet_id,
                    triplet.get("input_query", triplet.get("input", "")),
                    json.dumps(triplet.get("chosen", [])),
                    json.dumps(triplet.get("rejected", [])),
                    triplet.get("confidence", 0.5),
                    triplet.get("quality", "bronze"),
                    triplet.get("rationale"),
                    json.dumps(triplet.get("failure_reasons", [])),
                    triplet.get("source", "batch_import"),
                    StoreStatus.ACTIVE.value,
                    now,
                    now,
                    json.dumps(triplet.get("metadata", {})),
                )
            )

        conn.commit()
        self._close_conn(conn)

        return ids

    def get_triplet(self, triplet_id: str) -> Optional[StoredTriplet]:
        """Get a triplet by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM triplets WHERE id = ?", (triplet_id,))
        row = cursor.fetchone()
        self._close_conn(conn)

        if not row:
            return None

        return self._row_to_triplet(row)

    def _row_to_triplet(self, row: tuple) -> StoredTriplet:
        """Convert database row to StoredTriplet."""
        return StoredTriplet(
            id=row[0],
            input_query=row[1],
            chosen=json.loads(row[2]),
            rejected=json.loads(row[3]),
            confidence=row[4],
            quality=row[5],
            rationale=row[6],
            failure_reasons=json.loads(row[7]),
            source=row[8],
            status=StoreStatus(row[9]),
            created_at=datetime.fromisoformat(row[10]),
            updated_at=datetime.fromisoformat(row[11]),
            metadata=json.loads(row[12]) if row[12] else {},
        )

    def query_triplets(
        self,
        quality: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        status: StoreStatus = StoreStatus.ACTIVE,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[StoredTriplet]:
        """Query triplets with filters."""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT * FROM triplets WHERE status = ?"
        params: list[Any] = [status.value]

        if quality:
            query += " AND quality = ?"
            params.append(quality)

        if source:
            query += " AND source = ?"
            params.append(source)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        self._close_conn(conn)

        return [self._row_to_triplet(row) for row in rows]

    def update_triplet_status(
        self,
        triplet_id: str,
        status: StoreStatus,
    ) -> bool:
        """Update triplet status."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE triplets SET status = ?, updated_at = ? WHERE id = ?",
            (status.value, datetime.utcnow().isoformat(), triplet_id)
        )

        updated = cursor.rowcount > 0
        conn.commit()
        self._close_conn(conn)

        return updated

    def create_dataset_version(
        self,
        name: str,
        description: Optional[str] = None,
        quality_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
        sources: Optional[list[str]] = None,
        created_by: Optional[str] = None,
    ) -> DatasetVersion:
        """Create a versioned snapshot of the dataset."""
        version_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Build filter query
        filters: dict[str, Any] = {}
        query = "SELECT id FROM triplets WHERE status = 'active'"
        params: list[Any] = []

        if quality_filter:
            query += " AND quality = ?"
            params.append(quality_filter)
            filters["quality"] = quality_filter

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)
            filters["min_confidence"] = min_confidence

        if sources:
            placeholders = ",".join("?" * len(sources))
            query += f" AND source IN ({placeholders})"
            params.extend(sources)
            filters["sources"] = sources

        conn = self._get_conn()
        cursor = conn.cursor()

        # Get matching triplet IDs
        cursor.execute(query, params)
        triplet_ids = [row[0] for row in cursor.fetchall()]

        # Create version record
        cursor.execute(
            """
            INSERT INTO dataset_versions
            (version_id, name, description, triplet_count, created_at,
             created_by, filters_applied, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                name,
                description,
                len(triplet_ids),
                now.isoformat(),
                created_by,
                json.dumps(filters),
                json.dumps({}),
            )
        )

        # Link triplets to version
        for triplet_id in triplet_ids:
            cursor.execute(
                "INSERT INTO version_triplets (version_id, triplet_id) VALUES (?, ?)",
                (version_id, triplet_id)
            )

        conn.commit()
        self._close_conn(conn)

        return DatasetVersion(
            version_id=version_id,
            name=name,
            description=description,
            triplet_count=len(triplet_ids),
            created_at=now,
            created_by=created_by,
            filters_applied=filters,
        )

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get a dataset version by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM dataset_versions WHERE version_id = ?",
            (version_id,)
        )
        row = cursor.fetchone()
        self._close_conn(conn)

        if not row:
            return None

        return DatasetVersion(
            version_id=row[0],
            name=row[1],
            description=row[2],
            triplet_count=row[3],
            created_at=datetime.fromisoformat(row[4]),
            created_by=row[5],
            filters_applied=json.loads(row[6]) if row[6] else {},
            metadata=json.loads(row[7]) if row[7] else {},
        )

    def get_version_triplets(
        self,
        version_id: str,
    ) -> Iterator[StoredTriplet]:
        """Iterate over triplets in a dataset version."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT t.* FROM triplets t
            JOIN version_triplets vt ON t.id = vt.triplet_id
            WHERE vt.version_id = ?
            """,
            (version_id,)
        )

        for row in cursor:
            yield self._row_to_triplet(row)

        self._close_conn(conn)

    def export_version_jsonl(self, version_id: str) -> str:
        """Export a dataset version as JSONL."""
        lines = []
        for triplet in self.get_version_triplets(version_id):
            lines.append(json.dumps(triplet.to_dict()))
        return "\n".join(lines)

    def export_version_slime(self, version_id: str) -> list[dict[str, Any]]:
        """Export a dataset version in SLIME format."""
        data = []
        for triplet in self.get_version_triplets(version_id):
            data.append({
                "prompt": triplet.input_query,
                "chosen": self._messages_to_text(triplet.chosen),
                "rejected": self._messages_to_text(triplet.rejected),
                "chosen_score": triplet.confidence,
                "rejected_score": 1.0 - triplet.confidence,
            })
        return data

    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        """Convert messages to text format."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        return "\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM triplets WHERE status = 'active'")
        total_triplets = cursor.fetchone()[0]

        cursor.execute(
            "SELECT quality, COUNT(*) FROM triplets WHERE status = 'active' GROUP BY quality"
        )
        quality_dist = dict(cursor.fetchall())

        cursor.execute(
            "SELECT source, COUNT(*) FROM triplets WHERE status = 'active' GROUP BY source"
        )
        source_dist = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM dataset_versions")
        total_versions = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(confidence) FROM triplets WHERE status = 'active'")
        avg_confidence = cursor.fetchone()[0] or 0.0

        self._close_conn(conn)

        return {
            "total_triplets": total_triplets,
            "quality_distribution": quality_dist,
            "source_distribution": source_dist,
            "total_versions": total_versions,
            "avg_confidence": avg_confidence,
        }

    def list_versions(self, limit: int = 100) -> list[DatasetVersion]:
        """List all dataset versions."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM dataset_versions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )

        versions = []
        for row in cursor.fetchall():
            versions.append(DatasetVersion(
                version_id=row[0],
                name=row[1],
                description=row[2],
                triplet_count=row[3],
                created_at=datetime.fromisoformat(row[4]),
                created_by=row[5],
                filters_applied=json.loads(row[6]) if row[6] else {},
                metadata=json.loads(row[7]) if row[7] else {},
            ))

        self._close_conn(conn)
        return versions
