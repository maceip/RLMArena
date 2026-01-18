"""
Telemetry Collector for auditable logging.

Provides StrongDM-style system-level logging of all agent actions
for compliance and debugging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import asyncio
import json
import hashlib
import sqlite3
import uuid


class EventType(Enum):
    """Types of telemetry events."""
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    VERIFICATION_START = "verification_start"
    VERIFICATION_END = "verification_end"
    TOOL_CALL = "tool_call"
    POLICY_CHECK = "policy_check"
    ERROR = "error"
    CORRECTION_ATTEMPT = "correction_attempt"
    MODEL_CALL = "model_call"
    CERTIFICATE_ISSUED = "certificate_issued"


class EventSeverity(Enum):
    """Severity levels for events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TelemetryEvent:
    """A telemetry event for audit logging."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    request_id: Optional[str]
    client_id: Optional[str]
    severity: EventSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "client_id": self.client_id,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class SpanContext:
    """Context for distributed tracing."""

    def __init__(
        self,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())[:16]
        self.parent_span_id = parent_span_id

    def create_child(self) -> "SpanContext":
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
        )


class TelemetryCollector:
    """
    Collects and stores telemetry events for audit logging.

    Features:
    - SQLite-based persistent storage
    - Distributed tracing support
    - Structured event logging
    - Query interface for analysis
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        retention_days: int = 30,
        max_events: int = 1000000,
    ):
        self.db_path = db_path
        self.retention_days = retention_days
        self.max_events = max_events
        self._init_db()
        self._event_count = 0

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                request_id TEXT,
                client_id TEXT,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                details TEXT,
                trace_id TEXT,
                span_id TEXT,
                parent_span_id TEXT,
                duration_ms REAL,
                metadata TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_client ON events(client_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_trace ON events(trace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")

        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO events
            (event_id, event_type, timestamp, request_id, client_id, severity,
             message, details, trace_id, span_id, parent_span_id, duration_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.request_id,
                event.client_id,
                event.severity.value,
                event.message,
                json.dumps(event.details),
                event.trace_id,
                event.span_id,
                event.parent_span_id,
                event.duration_ms,
                json.dumps(event.metadata),
            )
        )

        conn.commit()
        conn.close()

        self._event_count += 1

    def record_request_start(
        self,
        request_id: str,
        client_id: str,
        span_context: Optional[SpanContext] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TelemetryEvent:
        """Record start of a request."""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.REQUEST_START,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            client_id=client_id,
            severity=EventSeverity.INFO,
            message=f"Request {request_id} started",
            trace_id=span_context.trace_id if span_context else None,
            span_id=span_context.span_id if span_context else None,
            metadata=metadata or {},
        )

        self.record(event)
        return event

    def record_request_end(
        self,
        request_id: str,
        client_id: str,
        success: bool,
        duration_ms: float,
        span_context: Optional[SpanContext] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TelemetryEvent:
        """Record end of a request."""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.REQUEST_END,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            client_id=client_id,
            severity=EventSeverity.INFO if success else EventSeverity.ERROR,
            message=f"Request {request_id} {'completed' if success else 'failed'}",
            duration_ms=duration_ms,
            trace_id=span_context.trace_id if span_context else None,
            span_id=span_context.span_id if span_context else None,
            metadata=metadata or {},
        )

        self.record(event)
        return event

    def record_verification(
        self,
        request_id: str,
        plugin_name: str,
        passed: bool,
        duration_ms: float,
        details: Optional[dict[str, Any]] = None,
        span_context: Optional[SpanContext] = None,
    ) -> TelemetryEvent:
        """Record a verification event."""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.VERIFICATION_END,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            client_id=None,
            severity=EventSeverity.INFO if passed else EventSeverity.WARNING,
            message=f"Verification {plugin_name}: {'passed' if passed else 'failed'}",
            details=details or {},
            duration_ms=duration_ms,
            trace_id=span_context.trace_id if span_context else None,
            span_id=span_context.span_id if span_context else None,
        )

        self.record(event)
        return event

    def record_error(
        self,
        request_id: Optional[str],
        error: str,
        details: Optional[dict[str, Any]] = None,
        severity: EventSeverity = EventSeverity.ERROR,
        span_context: Optional[SpanContext] = None,
    ) -> TelemetryEvent:
        """Record an error event."""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ERROR,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            client_id=None,
            severity=severity,
            message=error,
            details=details or {},
            trace_id=span_context.trace_id if span_context else None,
            span_id=span_context.span_id if span_context else None,
        )

        self.record(event)
        return event

    def query_by_request(self, request_id: str) -> list[TelemetryEvent]:
        """Query events by request ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM events WHERE request_id = ? ORDER BY timestamp",
            (request_id,)
        )

        events = [self._row_to_event(row) for row in cursor.fetchall()]
        conn.close()

        return events

    def query_by_trace(self, trace_id: str) -> list[TelemetryEvent]:
        """Query events by trace ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM events WHERE trace_id = ? ORDER BY timestamp",
            (trace_id,)
        )

        events = [self._row_to_event(row) for row in cursor.fetchall()]
        conn.close()

        return events

    def query_by_client(
        self,
        client_id: str,
        limit: int = 100,
    ) -> list[TelemetryEvent]:
        """Query events by client ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM events WHERE client_id = ? ORDER BY timestamp DESC LIMIT ?",
            (client_id, limit)
        )

        events = [self._row_to_event(row) for row in cursor.fetchall()]
        conn.close()

        return events

    def query_errors(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[TelemetryEvent]:
        """Query error events."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if since:
            cursor.execute(
                """
                SELECT * FROM events
                WHERE severity IN ('error', 'critical') AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (since.isoformat(), limit)
            )
        else:
            cursor.execute(
                """
                SELECT * FROM events
                WHERE severity IN ('error', 'critical')
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,)
            )

        events = [self._row_to_event(row) for row in cursor.fetchall()]
        conn.close()

        return events

    def _row_to_event(self, row: tuple) -> TelemetryEvent:
        """Convert database row to TelemetryEvent."""
        return TelemetryEvent(
            event_id=row[0],
            event_type=EventType(row[1]),
            timestamp=datetime.fromisoformat(row[2]),
            request_id=row[3],
            client_id=row[4],
            severity=EventSeverity(row[5]),
            message=row[6],
            details=json.loads(row[7]) if row[7] else {},
            trace_id=row[8],
            span_id=row[9],
            parent_span_id=row[10],
            duration_ms=row[11],
            metadata=json.loads(row[12]) if row[12] else {},
        )

    def get_stats(self) -> dict[str, Any]:
        """Get telemetry statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]

        cursor.execute(
            "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
        )
        by_type = dict(cursor.fetchall())

        cursor.execute(
            "SELECT severity, COUNT(*) FROM events GROUP BY severity"
        )
        by_severity = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(DISTINCT request_id) FROM events")
        unique_requests = cursor.fetchone()[0]

        conn.close()

        return {
            "total_events": total_events,
            "by_type": by_type,
            "by_severity": by_severity,
            "unique_requests": unique_requests,
            "retention_days": self.retention_days,
        }

    def cleanup_old_events(self) -> int:
        """Remove events older than retention period."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cutoff = datetime.utcnow().replace(
            day=datetime.utcnow().day - self.retention_days
        )

        cursor.execute(
            "DELETE FROM events WHERE timestamp < ?",
            (cutoff.isoformat(),)
        )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted
