"""
VaaS Router for request routing and load balancing.

Provides sglang-compatible request routing with support for:
- Model selection and load balancing
- Bracket-based routing
- Fallback chains
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import asyncio
import hashlib
import random
import time


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"


class ModelTier(Enum):
    """Model performance tiers."""
    PREMIUM = "premium"  # Highest quality, slowest
    STANDARD = "standard"  # Balanced
    FAST = "fast"  # Lower quality, fastest


@dataclass
class ModelEndpoint:
    """Configuration for a model endpoint."""
    name: str
    url: str
    tier: ModelTier
    weight: float = 1.0
    max_concurrent: int = 100
    timeout_seconds: float = 60.0
    healthy: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterConfig:
    """Configuration for the VaaS router."""
    default_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    health_check_interval_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_base: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: float = 60.0


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    endpoint: ModelEndpoint
    strategy_used: LoadBalanceStrategy
    fallback_used: bool = False
    decision_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker for endpoint protection."""

    def __init__(
        self,
        threshold: int = 5,
        timeout_seconds: float = 60.0,
    ):
        self.threshold = threshold
        self.timeout_seconds = timeout_seconds
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def record_failure(self, endpoint_name: str) -> None:
        """Record a failure for an endpoint."""
        self._failures[endpoint_name] = self._failures.get(endpoint_name, 0) + 1

        if self._failures[endpoint_name] >= self.threshold:
            self._open_until[endpoint_name] = time.time() + self.timeout_seconds

    def record_success(self, endpoint_name: str) -> None:
        """Record a success for an endpoint."""
        self._failures[endpoint_name] = 0
        self._open_until.pop(endpoint_name, None)

    def is_open(self, endpoint_name: str) -> bool:
        """Check if circuit is open (blocking requests)."""
        open_until = self._open_until.get(endpoint_name, 0)
        if time.time() < open_until:
            return True

        # Circuit closed, reset
        if endpoint_name in self._open_until:
            del self._open_until[endpoint_name]
            self._failures[endpoint_name] = 0

        return False

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        now = time.time()
        return {
            endpoint: {
                "failures": self._failures.get(endpoint, 0),
                "open": now < open_until,
                "closes_in_seconds": max(0, open_until - now),
            }
            for endpoint, open_until in self._open_until.items()
        }


class VaaSRouter:
    """
    VaaS request router with load balancing and fallback support.

    Features:
    - Multiple load balancing strategies
    - Health checking
    - Circuit breaker protection
    - Bracket-based routing
    """

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        endpoints: Optional[list[ModelEndpoint]] = None,
    ):
        self.config = config or RouterConfig()
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._current_connections: dict[str, int] = {}
        self._round_robin_index = 0
        self._circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout_seconds,
        ) if self.config.enable_circuit_breaker else None
        self._bracket_routes: dict[str, list[str]] = {}

        if endpoints:
            for endpoint in endpoints:
                self.add_endpoint(endpoint)

    def add_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Add an endpoint to the router."""
        self._endpoints[endpoint.name] = endpoint
        self._current_connections[endpoint.name] = 0

    def remove_endpoint(self, name: str) -> bool:
        """Remove an endpoint from the router."""
        if name in self._endpoints:
            del self._endpoints[name]
            del self._current_connections[name]
            return True
        return False

    def set_bracket_route(self, bracket: str, endpoint_names: list[str]) -> None:
        """Set preferred endpoints for a bracket."""
        self._bracket_routes[bracket] = endpoint_names

    def _get_available_endpoints(
        self,
        tier: Optional[ModelTier] = None,
        bracket: Optional[str] = None,
    ) -> list[ModelEndpoint]:
        """Get available endpoints filtered by tier and bracket."""
        endpoints = list(self._endpoints.values())

        # Filter by health
        endpoints = [e for e in endpoints if e.healthy]

        # Filter by circuit breaker
        if self._circuit_breaker:
            endpoints = [
                e for e in endpoints
                if not self._circuit_breaker.is_open(e.name)
            ]

        # Filter by bracket preference
        if bracket and bracket in self._bracket_routes:
            preferred = self._bracket_routes[bracket]
            preferred_endpoints = [e for e in endpoints if e.name in preferred]
            if preferred_endpoints:
                endpoints = preferred_endpoints

        # Filter by tier
        if tier:
            tier_endpoints = [e for e in endpoints if e.tier == tier]
            if tier_endpoints:
                endpoints = tier_endpoints

        return endpoints

    def route(
        self,
        request_id: str,
        tier: Optional[ModelTier] = None,
        bracket: Optional[str] = None,
        strategy: Optional[LoadBalanceStrategy] = None,
    ) -> Optional[RoutingDecision]:
        """
        Route a request to an endpoint.

        Args:
            request_id: Unique request identifier
            tier: Preferred model tier
            bracket: Verification bracket (e.g., "open_coder")
            strategy: Override load balancing strategy

        Returns:
            RoutingDecision or None if no endpoints available
        """
        start_time = time.time()
        strategy = strategy or self.config.default_strategy

        endpoints = self._get_available_endpoints(tier, bracket)
        if not endpoints:
            return None

        # Select endpoint based on strategy
        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            endpoint = self._round_robin_select(endpoints)
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            endpoint = self._least_connections_select(endpoints)
        elif strategy == LoadBalanceStrategy.WEIGHTED:
            endpoint = self._weighted_select(endpoints)
        elif strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            endpoint = self._consistent_hash_select(endpoints, request_id)
        else:
            endpoint = random.choice(endpoints)

        decision_time = (time.time() - start_time) * 1000

        return RoutingDecision(
            endpoint=endpoint,
            strategy_used=strategy,
            decision_time_ms=decision_time,
            metadata={
                "available_endpoints": len(endpoints),
                "tier": tier.value if tier else None,
                "bracket": bracket,
            },
        )

    def _round_robin_select(self, endpoints: list[ModelEndpoint]) -> ModelEndpoint:
        """Select using round-robin."""
        endpoint = endpoints[self._round_robin_index % len(endpoints)]
        self._round_robin_index += 1
        return endpoint

    def _least_connections_select(self, endpoints: list[ModelEndpoint]) -> ModelEndpoint:
        """Select endpoint with fewest connections."""
        return min(
            endpoints,
            key=lambda e: self._current_connections.get(e.name, 0)
        )

    def _weighted_select(self, endpoints: list[ModelEndpoint]) -> ModelEndpoint:
        """Select based on weights."""
        total_weight = sum(e.weight for e in endpoints)
        r = random.uniform(0, total_weight)

        cumulative = 0.0
        for endpoint in endpoints:
            cumulative += endpoint.weight
            if r <= cumulative:
                return endpoint

        return endpoints[-1]

    def _consistent_hash_select(
        self,
        endpoints: list[ModelEndpoint],
        key: str,
    ) -> ModelEndpoint:
        """Select using consistent hashing."""
        hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return endpoints[hash_val % len(endpoints)]

    def record_request_start(self, endpoint_name: str) -> None:
        """Record start of a request to an endpoint."""
        self._current_connections[endpoint_name] = (
            self._current_connections.get(endpoint_name, 0) + 1
        )

    def record_request_end(
        self,
        endpoint_name: str,
        success: bool,
    ) -> None:
        """Record end of a request to an endpoint."""
        self._current_connections[endpoint_name] = max(
            0, self._current_connections.get(endpoint_name, 0) - 1
        )

        if self._circuit_breaker:
            if success:
                self._circuit_breaker.record_success(endpoint_name)
            else:
                self._circuit_breaker.record_failure(endpoint_name)

    def set_endpoint_health(self, name: str, healthy: bool) -> None:
        """Update endpoint health status."""
        if name in self._endpoints:
            self._endpoints[name].healthy = healthy

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "endpoints": {
                name: {
                    "healthy": endpoint.healthy,
                    "tier": endpoint.tier.value,
                    "current_connections": self._current_connections.get(name, 0),
                    "weight": endpoint.weight,
                }
                for name, endpoint in self._endpoints.items()
            },
            "circuit_breaker": (
                self._circuit_breaker.get_status()
                if self._circuit_breaker else None
            ),
            "bracket_routes": self._bracket_routes,
        }


class RequestExecutor:
    """Executes requests with retry and fallback logic."""

    def __init__(
        self,
        router: VaaSRouter,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ):
        self.router = router
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    async def execute(
        self,
        request_id: str,
        handler: Callable,
        tier: Optional[ModelTier] = None,
        bracket: Optional[str] = None,
    ) -> tuple[Any, RoutingDecision]:
        """
        Execute a request with retry and fallback.

        Args:
            request_id: Unique request identifier
            handler: Async function to execute with endpoint
            tier: Preferred model tier
            bracket: Verification bracket

        Returns:
            Tuple of (result, routing_decision)
        """
        last_error = None

        for attempt in range(self.max_retries):
            decision = self.router.route(request_id, tier, bracket)

            if decision is None:
                raise RuntimeError("No available endpoints")

            self.router.record_request_start(decision.endpoint.name)

            try:
                result = await asyncio.wait_for(
                    handler(decision.endpoint),
                    timeout=decision.endpoint.timeout_seconds,
                )
                self.router.record_request_end(decision.endpoint.name, True)
                return result, decision

            except Exception as e:
                last_error = e
                self.router.record_request_end(decision.endpoint.name, False)

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.backoff_base * (2 ** attempt))

        raise last_error or RuntimeError("Request failed after retries")
