"""
Verification Gateway - HTTP-compatible API layer for VaaS.

This module provides the API gateway for the Verifier-as-a-Service,
handling request routing, rate limiting, and response formatting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import hashlib
import json
import time
import uuid


class RequestPriority(Enum):
    """Priority levels for requests."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RequestStatus(Enum):
    """Status of a verification request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    burst_window_seconds: float = 1.0


@dataclass
class QuotaConfig:
    """Configuration for request quotas."""
    max_tokens_per_request: int = 8192
    max_variations_per_request: int = 5
    max_concurrent_requests: int = 10
    max_execution_time_seconds: float = 120.0


@dataclass
class RequestContext:
    """Context for a verification request."""
    request_id: str
    client_id: str
    timestamp: datetime
    priority: RequestPriority = RequestPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    trace_id: Optional[str] = None


@dataclass
class ResponseEnvelope:
    """Standardized response envelope for VaaS."""
    request_id: str
    status: RequestStatus
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    verification_result: Optional[dict[str, Any]] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "verification_result": self.verification_result,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        refill_interval: float = 1.0,
    ):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        if elapsed >= self.refill_interval:
            refill_count = int(elapsed / self.refill_interval) * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + refill_count)
            self.last_refill = now


class RateLimiter:
    """
    Rate limiter with multiple time windows.

    Tracks requests per minute, hour, and day with burst protection.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._client_buckets: dict[str, dict[str, TokenBucket]] = {}
        self._request_history: dict[str, list[datetime]] = {}

    def _get_buckets(self, client_id: str) -> dict[str, TokenBucket]:
        """Get or create token buckets for a client."""
        if client_id not in self._client_buckets:
            self._client_buckets[client_id] = {
                "minute": TokenBucket(
                    self.config.requests_per_minute,
                    self.config.requests_per_minute / 60,
                ),
                "hour": TokenBucket(
                    self.config.requests_per_hour,
                    self.config.requests_per_hour / 3600,
                ),
                "day": TokenBucket(
                    self.config.requests_per_day,
                    self.config.requests_per_day / 86400,
                ),
                "burst": TokenBucket(
                    self.config.burst_size,
                    self.config.burst_size / self.config.burst_window_seconds,
                    self.config.burst_window_seconds,
                ),
            }
        return self._client_buckets[client_id]

    async def check_rate_limit(self, client_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.

        Returns (allowed, reason_if_denied).
        """
        buckets = self._get_buckets(client_id)

        # Check burst limit first
        if not await buckets["burst"].acquire():
            return False, "Burst limit exceeded"

        # Check minute limit
        if not await buckets["minute"].acquire():
            return False, "Per-minute limit exceeded"

        # Check hour limit
        if not await buckets["hour"].acquire():
            return False, "Per-hour limit exceeded"

        # Check day limit
        if not await buckets["day"].acquire():
            return False, "Daily limit exceeded"

        return True, None

    def get_client_usage(self, client_id: str) -> dict[str, Any]:
        """Get current usage for a client."""
        if client_id not in self._client_buckets:
            return {"requests_remaining": self.config.requests_per_minute}

        buckets = self._client_buckets[client_id]
        return {
            "minute_remaining": int(buckets["minute"].tokens),
            "hour_remaining": int(buckets["hour"].tokens),
            "day_remaining": int(buckets["day"].tokens),
            "burst_remaining": int(buckets["burst"].tokens),
        }


class RequestQueue:
    """Priority queue for verification requests."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues: dict[RequestPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=max_size // 4)
            for priority in RequestPriority
        }
        self._total_count = 0

    async def put(self, context: RequestContext, request_data: dict[str, Any]) -> bool:
        """Add request to queue."""
        try:
            queue = self._queues[context.priority]
            await asyncio.wait_for(
                queue.put((context, request_data)),
                timeout=1.0,
            )
            self._total_count += 1
            return True
        except (asyncio.TimeoutError, asyncio.QueueFull):
            return False

    async def get(self) -> tuple[RequestContext, dict[str, Any]]:
        """Get highest priority request."""
        # Check queues in priority order
        for priority in reversed(RequestPriority):
            queue = self._queues[priority]
            if not queue.empty():
                item = await queue.get()
                self._total_count -= 1
                return item

        # Wait on all queues
        done, pending = await asyncio.wait(
            [asyncio.create_task(q.get()) for q in self._queues.values()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending
        for task in pending:
            task.cancel()

        # Return first completed
        for task in done:
            self._total_count -= 1
            return task.result()

        # Should not reach here
        raise RuntimeError("Queue get failed")

    def size(self) -> int:
        """Get total queue size."""
        return self._total_count


class ResultCache:
    """Cache for verification results."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[ResponseEnvelope, datetime]] = {}
        self._access_times: dict[str, datetime] = {}

    def _compute_key(self, request_data: dict[str, Any]) -> str:
        """Compute cache key from request data."""
        content = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, request_data: dict[str, Any]) -> Optional[ResponseEnvelope]:
        """Get cached result if available and not expired."""
        key = self._compute_key(request_data)

        if key not in self._cache:
            return None

        result, cached_at = self._cache[key]
        if (datetime.utcnow() - cached_at).total_seconds() > self.ttl_seconds:
            del self._cache[key]
            return None

        self._access_times[key] = datetime.utcnow()
        return result

    def put(self, request_data: dict[str, Any], result: ResponseEnvelope) -> None:
        """Cache a result."""
        key = self._compute_key(request_data)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest = min(self._access_times, key=self._access_times.get)
            del self._cache[oldest]
            del self._access_times[oldest]

        self._cache[key] = (result, datetime.utcnow())
        self._access_times[key] = datetime.utcnow()

    def invalidate(self, request_data: dict[str, Any]) -> bool:
        """Invalidate cached result."""
        key = self._compute_key(request_data)
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
            return True
        return False

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


class VerificationGateway:
    """
    Verification Gateway - API layer for VaaS.

    Handles:
    - Request validation and routing
    - Rate limiting and quotas
    - Response caching
    - Telemetry and logging
    """

    def __init__(
        self,
        rate_limit_config: Optional[RateLimitConfig] = None,
        quota_config: Optional[QuotaConfig] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        self._rate_limiter = RateLimiter(rate_limit_config)
        self._quota_config = quota_config or QuotaConfig()
        self._request_queue = RequestQueue()
        self._cache = ResultCache(ttl_seconds=cache_ttl_seconds) if enable_caching else None
        self._active_requests: dict[str, RequestContext] = {}
        self._request_handlers: dict[str, Callable] = {}
        self._metrics: dict[str, int] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "cached_responses": 0,
        }
        self._semaphore = asyncio.Semaphore(self._quota_config.max_concurrent_requests)

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a request handler."""
        self._request_handlers[name] = handler

    async def handle_request(
        self,
        client_id: str,
        request_type: str,
        request_data: dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
    ) -> ResponseEnvelope:
        """
        Handle a verification request.

        Args:
            client_id: Client identifier for rate limiting
            request_type: Type of verification request
            request_data: The request payload
            priority: Request priority
            timeout_seconds: Optional timeout override

        Returns:
            ResponseEnvelope with verification result
        """
        self._metrics["total_requests"] += 1
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Create context
        context = RequestContext(
            request_id=request_id,
            client_id=client_id,
            timestamp=datetime.utcnow(),
            priority=priority,
            timeout_seconds=timeout_seconds or self._quota_config.max_execution_time_seconds,
        )

        # Check rate limit
        allowed, reason = await self._rate_limiter.check_rate_limit(client_id)
        if not allowed:
            self._metrics["rate_limited_requests"] += 1
            return ResponseEnvelope(
                request_id=request_id,
                status=RequestStatus.RATE_LIMITED,
                error=reason,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Check cache
        if self._cache:
            cached = self._cache.get(request_data)
            if cached:
                self._metrics["cached_responses"] += 1
                cached.request_id = request_id
                cached.latency_ms = (time.time() - start_time) * 1000
                return cached

        # Validate request
        validation_error = self._validate_request(request_data)
        if validation_error:
            self._metrics["failed_requests"] += 1
            return ResponseEnvelope(
                request_id=request_id,
                status=RequestStatus.FAILED,
                error=validation_error,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Get handler
        handler = self._request_handlers.get(request_type)
        if not handler:
            self._metrics["failed_requests"] += 1
            return ResponseEnvelope(
                request_id=request_id,
                status=RequestStatus.FAILED,
                error=f"Unknown request type: {request_type}",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Process request with concurrency limit
        try:
            async with self._semaphore:
                self._active_requests[request_id] = context

                result = await asyncio.wait_for(
                    handler(context, request_data),
                    timeout=context.timeout_seconds,
                )

                response = ResponseEnvelope(
                    request_id=request_id,
                    status=RequestStatus.COMPLETED,
                    data=result.get("data"),
                    verification_result=result.get("verification"),
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata=result.get("metadata", {}),
                )

                self._metrics["successful_requests"] += 1

                # Cache result
                if self._cache:
                    self._cache.put(request_data, response)

                return response

        except asyncio.TimeoutError:
            self._metrics["failed_requests"] += 1
            return ResponseEnvelope(
                request_id=request_id,
                status=RequestStatus.TIMEOUT,
                error=f"Request timed out after {context.timeout_seconds}s",
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self._metrics["failed_requests"] += 1
            return ResponseEnvelope(
                request_id=request_id,
                status=RequestStatus.FAILED,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

        finally:
            self._active_requests.pop(request_id, None)

    def _validate_request(self, request_data: dict[str, Any]) -> Optional[str]:
        """Validate request against quotas."""
        messages = request_data.get("messages", [])

        # Check token count (rough estimate)
        total_tokens = sum(
            len(m.get("content", "")) // 4
            for m in messages
            if isinstance(m.get("content"), str)
        )

        if total_tokens > self._quota_config.max_tokens_per_request:
            return f"Token count {total_tokens} exceeds limit {self._quota_config.max_tokens_per_request}"

        # Check variations
        num_variations = request_data.get("num_variations", 1)
        if num_variations > self._quota_config.max_variations_per_request:
            return f"Variation count {num_variations} exceeds limit {self._quota_config.max_variations_per_request}"

        return None

    def get_metrics(self) -> dict[str, Any]:
        """Get gateway metrics."""
        return {
            **self._metrics,
            "active_requests": len(self._active_requests),
            "queue_size": self._request_queue.size(),
            "cache_stats": self._cache.stats() if self._cache else None,
        }

    def get_client_usage(self, client_id: str) -> dict[str, Any]:
        """Get usage statistics for a client."""
        return self._rate_limiter.get_client_usage(client_id)


class HealthCheck:
    """Health check for the gateway."""

    def __init__(self, gateway: VerificationGateway):
        self.gateway = gateway
        self._start_time = datetime.utcnow()

    def check(self) -> dict[str, Any]:
        """Run health check."""
        metrics = self.gateway.get_metrics()

        is_healthy = (
            metrics["active_requests"] < 100 and
            metrics.get("failed_requests", 0) / max(metrics["total_requests"], 1) < 0.5
        )

        return {
            "status": "healthy" if is_healthy else "degraded",
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "total_requests": metrics["total_requests"],
            "error_rate": metrics["failed_requests"] / max(metrics["total_requests"], 1),
            "active_requests": metrics["active_requests"],
            "timestamp": datetime.utcnow().isoformat(),
        }
