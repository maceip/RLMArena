"""
Cost Tracker for API accounting and quota management.

Tracks usage, costs, and quotas for VaaS billing and resource management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
import json
import sqlite3
import uuid


class UsageType(Enum):
    """Types of usage to track."""
    API_CALL = "api_call"
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"
    VERIFICATION = "verification"
    EXECUTION = "execution"
    STORAGE = "storage"


class BillingPeriod(Enum):
    """Billing period types."""
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"


@dataclass
class UsageRecord:
    """Record of resource usage."""
    record_id: str
    client_id: str
    usage_type: UsageType
    quantity: float
    unit: str
    cost_usd: float
    timestamp: datetime
    request_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "client_id": self.client_id,
            "usage_type": self.usage_type.value,
            "quantity": self.quantity,
            "unit": self.unit,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "metadata": self.metadata,
        }


@dataclass
class QuotaConfig:
    """Quota configuration for a client."""
    client_id: str
    max_requests_per_day: int = 10000
    max_tokens_per_day: int = 1000000
    max_cost_per_day_usd: float = 100.0
    max_concurrent_requests: int = 10
    tier: str = "standard"


@dataclass
class UsageSummary:
    """Summary of usage for a period."""
    client_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    total_tokens_input: int
    total_tokens_output: int
    total_verifications: int
    total_cost_usd: float
    by_type: dict[str, float] = field(default_factory=dict)


class PricingConfig:
    """Pricing configuration for different resources."""

    def __init__(self):
        self.prices = {
            UsageType.API_CALL: 0.001,  # $0.001 per call
            UsageType.TOKEN_INPUT: 0.000003,  # $3 per 1M tokens
            UsageType.TOKEN_OUTPUT: 0.000015,  # $15 per 1M tokens
            UsageType.VERIFICATION: 0.0001,  # $0.0001 per verification
            UsageType.EXECUTION: 0.001,  # $0.001 per execution second
            UsageType.STORAGE: 0.00001,  # $0.01 per MB per day
        }

        self.tier_multipliers = {
            "free": 0.0,
            "standard": 1.0,
            "premium": 1.5,
            "enterprise": 0.8,  # Volume discount
        }

    def calculate_cost(
        self,
        usage_type: UsageType,
        quantity: float,
        tier: str = "standard",
    ) -> float:
        """Calculate cost for usage."""
        base_price = self.prices.get(usage_type, 0.0)
        multiplier = self.tier_multipliers.get(tier, 1.0)
        return base_price * quantity * multiplier


class CostTracker:
    """
    Tracks API costs and manages quotas.

    Features:
    - Usage recording and aggregation
    - Quota enforcement
    - Cost calculation
    - Billing period summaries
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        pricing: Optional[PricingConfig] = None,
    ):
        self.db_path = db_path
        self.pricing = pricing or PricingConfig()
        self._quotas: dict[str, QuotaConfig] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_records (
                record_id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                usage_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                timestamp TEXT NOT NULL,
                request_id TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_client ON usage_records(client_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_records(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_type ON usage_records(usage_type)
        """)

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

    def set_quota(self, config: QuotaConfig) -> None:
        """Set quota configuration for a client."""
        self._quotas[config.client_id] = config

    def get_quota(self, client_id: str) -> Optional[QuotaConfig]:
        """Get quota configuration for a client."""
        return self._quotas.get(client_id)

    def record_usage(
        self,
        client_id: str,
        usage_type: UsageType,
        quantity: float,
        unit: str,
        request_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a usage event."""
        quota = self._quotas.get(client_id)
        tier = quota.tier if quota else "standard"

        cost = self.pricing.calculate_cost(usage_type, quantity, tier)

        record = UsageRecord(
            record_id=str(uuid.uuid4()),
            client_id=client_id,
            usage_type=usage_type,
            quantity=quantity,
            unit=unit,
            cost_usd=cost,
            timestamp=datetime.utcnow(),
            request_id=request_id,
            metadata=metadata or {},
        )

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO usage_records
            (record_id, client_id, usage_type, quantity, unit, cost_usd,
             timestamp, request_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.client_id,
                record.usage_type.value,
                record.quantity,
                record.unit,
                record.cost_usd,
                record.timestamp.isoformat(),
                record.request_id,
                json.dumps(record.metadata),
            )
        )

        conn.commit()
        self._close_conn(conn)

        return record

    def record_request(
        self,
        client_id: str,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        num_verifications: int = 0,
        execution_seconds: float = 0.0,
    ) -> list[UsageRecord]:
        """Record usage for a complete request."""
        records = []

        # API call
        records.append(self.record_usage(
            client_id,
            UsageType.API_CALL,
            1.0,
            "calls",
            request_id,
        ))

        # Input tokens
        if input_tokens > 0:
            records.append(self.record_usage(
                client_id,
                UsageType.TOKEN_INPUT,
                input_tokens,
                "tokens",
                request_id,
            ))

        # Output tokens
        if output_tokens > 0:
            records.append(self.record_usage(
                client_id,
                UsageType.TOKEN_OUTPUT,
                output_tokens,
                "tokens",
                request_id,
            ))

        # Verifications
        if num_verifications > 0:
            records.append(self.record_usage(
                client_id,
                UsageType.VERIFICATION,
                num_verifications,
                "verifications",
                request_id,
            ))

        # Execution time
        if execution_seconds > 0:
            records.append(self.record_usage(
                client_id,
                UsageType.EXECUTION,
                execution_seconds,
                "seconds",
                request_id,
            ))

        return records

    def check_quota(
        self,
        client_id: str,
        usage_type: Optional[UsageType] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if client is within quota limits.

        Returns (within_quota, reason_if_exceeded).
        """
        quota = self._quotas.get(client_id)
        if not quota:
            return True, None  # No quota set, allow

        # Get today's usage
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        conn = self._get_conn()
        cursor = conn.cursor()

        # Check request count
        cursor.execute(
            """
            SELECT COUNT(*) FROM usage_records
            WHERE client_id = ? AND usage_type = ? AND timestamp >= ?
            """,
            (client_id, UsageType.API_CALL.value, today_start.isoformat())
        )
        request_count = cursor.fetchone()[0]

        if request_count >= quota.max_requests_per_day:
            self._close_conn(conn)
            return False, f"Daily request limit ({quota.max_requests_per_day}) exceeded"

        # Check token count
        cursor.execute(
            """
            SELECT SUM(quantity) FROM usage_records
            WHERE client_id = ? AND usage_type IN (?, ?) AND timestamp >= ?
            """,
            (
                client_id,
                UsageType.TOKEN_INPUT.value,
                UsageType.TOKEN_OUTPUT.value,
                today_start.isoformat(),
            )
        )
        token_count = cursor.fetchone()[0] or 0

        if token_count >= quota.max_tokens_per_day:
            self._close_conn(conn)
            return False, f"Daily token limit ({quota.max_tokens_per_day}) exceeded"

        # Check cost
        cursor.execute(
            """
            SELECT SUM(cost_usd) FROM usage_records
            WHERE client_id = ? AND timestamp >= ?
            """,
            (client_id, today_start.isoformat())
        )
        total_cost = cursor.fetchone()[0] or 0.0

        self._close_conn(conn)

        if total_cost >= quota.max_cost_per_day_usd:
            return False, f"Daily cost limit (${quota.max_cost_per_day_usd}) exceeded"

        return True, None

    def get_usage_summary(
        self,
        client_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> UsageSummary:
        """Get usage summary for a client and period."""
        if period_start is None:
            period_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        if period_end is None:
            period_end = datetime.utcnow()

        conn = self._get_conn()
        cursor = conn.cursor()

        # Get totals by type
        cursor.execute(
            """
            SELECT usage_type, SUM(quantity), SUM(cost_usd)
            FROM usage_records
            WHERE client_id = ? AND timestamp >= ? AND timestamp < ?
            GROUP BY usage_type
            """,
            (client_id, period_start.isoformat(), period_end.isoformat())
        )

        by_type = {}
        total_requests = 0
        total_tokens_input = 0
        total_tokens_output = 0
        total_verifications = 0
        total_cost = 0.0

        for row in cursor.fetchall():
            usage_type, quantity, cost = row
            by_type[usage_type] = cost
            total_cost += cost

            if usage_type == UsageType.API_CALL.value:
                total_requests = int(quantity)
            elif usage_type == UsageType.TOKEN_INPUT.value:
                total_tokens_input = int(quantity)
            elif usage_type == UsageType.TOKEN_OUTPUT.value:
                total_tokens_output = int(quantity)
            elif usage_type == UsageType.VERIFICATION.value:
                total_verifications = int(quantity)

        self._close_conn(conn)

        return UsageSummary(
            client_id=client_id,
            period_start=period_start,
            period_end=period_end,
            total_requests=total_requests,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            total_verifications=total_verifications,
            total_cost_usd=total_cost,
            by_type=by_type,
        )

    def get_monthly_invoice(
        self,
        client_id: str,
        year: int,
        month: int,
    ) -> dict[str, Any]:
        """Generate monthly invoice for a client."""
        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)

        summary = self.get_usage_summary(client_id, period_start, period_end)

        quota = self._quotas.get(client_id)
        tier = quota.tier if quota else "standard"

        return {
            "client_id": client_id,
            "period": f"{year}-{month:02d}",
            "tier": tier,
            "line_items": [
                {
                    "description": "API Calls",
                    "quantity": summary.total_requests,
                    "unit": "calls",
                    "cost": summary.by_type.get(UsageType.API_CALL.value, 0.0),
                },
                {
                    "description": "Input Tokens",
                    "quantity": summary.total_tokens_input,
                    "unit": "tokens",
                    "cost": summary.by_type.get(UsageType.TOKEN_INPUT.value, 0.0),
                },
                {
                    "description": "Output Tokens",
                    "quantity": summary.total_tokens_output,
                    "unit": "tokens",
                    "cost": summary.by_type.get(UsageType.TOKEN_OUTPUT.value, 0.0),
                },
                {
                    "description": "Verifications",
                    "quantity": summary.total_verifications,
                    "unit": "verifications",
                    "cost": summary.by_type.get(UsageType.VERIFICATION.value, 0.0),
                },
            ],
            "subtotal": summary.total_cost_usd,
            "tax": 0.0,  # Add tax logic as needed
            "total": summary.total_cost_usd,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get cost tracker statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM usage_records")
        total_records = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT client_id) FROM usage_records")
        unique_clients = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(cost_usd) FROM usage_records")
        total_revenue = cursor.fetchone()[0] or 0.0

        self._close_conn(conn)

        return {
            "total_records": total_records,
            "unique_clients": unique_clients,
            "total_revenue_usd": total_revenue,
            "quotas_configured": len(self._quotas),
        }
