"""
Enhanced Shadow Loop with sglang integration.

This module implements the Shadow Loop architecture for parallel trajectory
generation and verification using RadixAttention-based routing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable, Protocol
from datetime import datetime
import asyncio
import json
import random
import hashlib


class VariationStrategy(Enum):
    """Strategies for generating trajectory variations."""
    TEMPERATURE_SWEEP = "temperature_sweep"
    PROMPT_PERTURBATION = "prompt_perturbation"
    MODEL_ENSEMBLE = "model_ensemble"
    TOP_K_VARIATION = "top_k_variation"
    NUCLEUS_VARIATION = "nucleus_variation"
    COMBINED = "combined"


class RoutingStrategy(Enum):
    """Strategies for routing requests to models."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    HASH_BASED = "hash_based"
    ADAPTIVE = "adaptive"


@dataclass
class GenerationConfig:
    """Configuration for a single generation request."""
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 4096
    stop_sequences: list[str] = field(default_factory=list)
    model: Optional[str] = None
    system_prompt_suffix: Optional[str] = None


@dataclass
class ShadowLoopConfig:
    """Configuration for the Shadow Loop."""
    num_variations: int = 4
    variation_strategy: VariationStrategy = VariationStrategy.TEMPERATURE_SWEEP
    routing_strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    timeout_seconds: float = 60.0
    parallel_generation: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 600
    min_valid_responses: int = 1
    temperature_range: tuple[float, float] = (0.3, 1.0)
    enable_radix_attention: bool = True
    max_prefix_cache_tokens: int = 8192


@dataclass
class GenerationResult:
    """Result of a single generation."""
    id: str
    content: str
    messages: list[dict[str, Any]]
    config: GenerationConfig
    latency_ms: float
    tokens_generated: int
    cache_hit: bool = False
    model_used: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShadowLoopResult:
    """Result of a complete shadow loop execution."""
    request_id: str
    variations: list[GenerationResult]
    best_variation_index: Optional[int] = None
    best_score: Optional[float] = None
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class RadixAttentionCache:
    """
    Mock RadixAttention prefix cache for efficient KV reuse.

    In production, this would interface with sglang's RadixAttention system.
    """

    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self._cache: dict[str, dict[str, Any]] = {}
        self._access_times: dict[str, datetime] = {}

    def _compute_prefix_hash(self, messages: list[dict[str, Any]]) -> str:
        """Compute hash for message prefix."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_prefix(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[Optional[str], int]:
        """
        Find the longest cached prefix for messages.

        Returns (cache_key, prefix_length) if found, else (None, 0).
        """
        # Try progressively shorter prefixes
        for i in range(len(messages), 0, -1):
            prefix = messages[:i]
            prefix_hash = self._compute_prefix_hash(prefix)

            if prefix_hash in self._cache:
                self._access_times[prefix_hash] = datetime.utcnow()
                return prefix_hash, i

        return None, 0

    def store_prefix(
        self,
        messages: list[dict[str, Any]],
        kv_cache: Optional[Any] = None,
    ) -> str:
        """Store a prefix in the cache."""
        prefix_hash = self._compute_prefix_hash(messages)

        self._cache[prefix_hash] = {
            "messages": messages,
            "kv_cache": kv_cache,
            "tokens": sum(len(m.get("content", "")) // 4 for m in messages),
        }
        self._access_times[prefix_hash] = datetime.utcnow()

        self._evict_if_needed()

        return prefix_hash

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is too large."""
        total_tokens = sum(e.get("tokens", 0) for e in self._cache.values())

        while total_tokens > self.max_tokens and self._cache:
            oldest_key = min(self._access_times, key=self._access_times.get)
            total_tokens -= self._cache[oldest_key].get("tokens", 0)
            del self._cache[oldest_key]
            del self._access_times[oldest_key]

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "total_tokens": sum(e.get("tokens", 0) for e in self._cache.values()),
            "max_tokens": self.max_tokens,
        }


class MockSGLangRouter:
    """
    Mock sglang router for request distribution.

    In production, this would interface with sglang's router.
    """

    def __init__(
        self,
        models: Optional[list[str]] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
    ):
        self.models = models or ["default-model"]
        self.routing_strategy = routing_strategy
        self._request_counts: dict[str, int] = {m: 0 for m in self.models}
        self._current_index = 0
        self._radix_cache = RadixAttentionCache()

    def select_model(self, request_hash: Optional[str] = None) -> str:
        """Select a model based on routing strategy."""
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            model = self.models[self._current_index % len(self.models)]
            self._current_index += 1
            return model

        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return min(self._request_counts, key=self._request_counts.get)

        elif self.routing_strategy == RoutingStrategy.RANDOM:
            return random.choice(self.models)

        elif self.routing_strategy == RoutingStrategy.HASH_BASED:
            if request_hash:
                index = int(request_hash, 16) % len(self.models)
                return self.models[index]
            return self.models[0]

        else:  # ADAPTIVE
            # Simple adaptive: prefer models with recent cache hits
            return random.choice(self.models)

    def record_request(self, model: str) -> None:
        """Record a request to a model."""
        self._request_counts[model] = self._request_counts.get(model, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "request_counts": self._request_counts.copy(),
            "cache_stats": self._radix_cache.stats(),
        }


class ModelPredictor(Protocol):
    """Protocol for model prediction interface."""

    async def generate(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig,
    ) -> GenerationResult:
        ...


class MockModelPredictor:
    """Mock model predictor for testing."""

    def __init__(self, latency_ms: float = 100.0):
        self.latency_ms = latency_ms
        self._call_count = 0

    async def generate(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate a mock response."""
        self._call_count += 1

        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Generate mock response
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        response_content = f"Mock response to: {user_content[:50]}... (temp={config.temperature})"

        response_messages = messages + [
            {"role": "assistant", "content": response_content}
        ]

        return GenerationResult(
            id=f"gen_{self._call_count}",
            content=response_content,
            messages=response_messages,
            config=config,
            latency_ms=self.latency_ms,
            tokens_generated=len(response_content) // 4,
            model_used=config.model or "mock-model",
        )


class ShadowLoop:
    """
    Shadow Loop for parallel trajectory generation and verification.

    The Shadow Loop:
    1. Intercepts incoming requests
    2. Generates multiple variations using different configs
    3. Uses RadixAttention for efficient prefix sharing
    4. Runs verification on all variations
    5. Returns the best verified response
    """

    def __init__(
        self,
        config: Optional[ShadowLoopConfig] = None,
        predictor: Optional[ModelPredictor] = None,
        router: Optional[MockSGLangRouter] = None,
    ):
        self.config = config or ShadowLoopConfig()
        self._predictor = predictor or MockModelPredictor()
        self._router = router or MockSGLangRouter()
        self._radix_cache = RadixAttentionCache(self.config.max_prefix_cache_tokens)
        self._request_count = 0

    def _generate_variation_configs(
        self,
        base_messages: list[dict[str, Any]],
    ) -> list[GenerationConfig]:
        """Generate configuration variations based on strategy."""
        configs = []
        n = self.config.num_variations

        if self.config.variation_strategy == VariationStrategy.TEMPERATURE_SWEEP:
            temp_min, temp_max = self.config.temperature_range
            for i in range(n):
                temp = temp_min + (temp_max - temp_min) * i / max(n - 1, 1)
                configs.append(GenerationConfig(temperature=temp))

        elif self.config.variation_strategy == VariationStrategy.TOP_K_VARIATION:
            top_k_values = [10, 30, 50, 100][:n]
            for top_k in top_k_values:
                configs.append(GenerationConfig(top_k=top_k))

        elif self.config.variation_strategy == VariationStrategy.NUCLEUS_VARIATION:
            top_p_values = [0.8, 0.9, 0.95, 0.99][:n]
            for top_p in top_p_values:
                configs.append(GenerationConfig(top_p=top_p))

        elif self.config.variation_strategy == VariationStrategy.PROMPT_PERTURBATION:
            perturbations = [
                None,
                "\nThink step by step.",
                "\nBe concise and precise.",
                "\nConsider edge cases carefully.",
            ][:n]
            for suffix in perturbations:
                configs.append(GenerationConfig(system_prompt_suffix=suffix))

        elif self.config.variation_strategy == VariationStrategy.MODEL_ENSEMBLE:
            for i in range(n):
                model = self._router.select_model()
                configs.append(GenerationConfig(model=model))

        else:  # COMBINED
            for i in range(n):
                temp_min, temp_max = self.config.temperature_range
                temp = temp_min + (temp_max - temp_min) * i / max(n - 1, 1)
                top_p = 0.9 + 0.05 * (i % 2)
                configs.append(GenerationConfig(temperature=temp, top_p=top_p))

        return configs

    async def generate_variations(
        self,
        messages: list[dict[str, Any]],
        configs: Optional[list[GenerationConfig]] = None,
    ) -> list[GenerationResult]:
        """Generate multiple trajectory variations."""
        if configs is None:
            configs = self._generate_variation_configs(messages)

        # Check for cached prefix
        cache_key, prefix_len = self._radix_cache.get_prefix(messages)

        if self.config.parallel_generation:
            tasks = [
                self._generate_single(messages, config, cache_key)
                for config in configs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, GenerationResult)]
            return valid_results
        else:
            results = []
            for config in configs:
                try:
                    result = await self._generate_single(messages, config, cache_key)
                    results.append(result)
                except Exception:
                    continue
            return results

    async def _generate_single(
        self,
        messages: list[dict[str, Any]],
        config: GenerationConfig,
        cache_key: Optional[str] = None,
    ) -> GenerationResult:
        """Generate a single variation."""
        # Apply any prompt perturbation
        if config.system_prompt_suffix:
            modified_messages = messages.copy()
            if modified_messages and modified_messages[-1].get("role") == "user":
                modified_messages[-1] = {
                    **modified_messages[-1],
                    "content": modified_messages[-1].get("content", "") + config.system_prompt_suffix,
                }
            messages = modified_messages

        result = await self._predictor.generate(messages, config)
        result.cache_hit = cache_key is not None

        # Store in cache for future requests
        if self.config.enable_caching:
            self._radix_cache.store_prefix(result.messages)

        return result

    async def run(
        self,
        messages: list[dict[str, Any]],
        scorer: Optional[Callable[[GenerationResult], float]] = None,
    ) -> ShadowLoopResult:
        """
        Execute the full shadow loop.

        Args:
            messages: Input messages for generation
            scorer: Optional function to score results (higher is better)

        Returns:
            ShadowLoopResult with all variations and best selection
        """
        self._request_count += 1
        request_id = f"shadow_{self._request_count}"
        start_time = datetime.utcnow()

        # Generate variations
        variations = await self.generate_variations(messages)

        if not variations:
            return ShadowLoopResult(
                request_id=request_id,
                variations=[],
                total_latency_ms=0.0,
                metadata={"error": "No valid variations generated"},
            )

        # Score variations if scorer provided
        best_index = None
        best_score = None

        if scorer is not None:
            scores = [scorer(v) for v in variations]
            best_index = max(range(len(scores)), key=lambda i: scores[i])
            best_score = scores[best_index]
        else:
            # Default: pick first valid result
            best_index = 0

        end_time = datetime.utcnow()
        total_latency = (end_time - start_time).total_seconds() * 1000

        return ShadowLoopResult(
            request_id=request_id,
            variations=variations,
            best_variation_index=best_index,
            best_score=best_score,
            total_latency_ms=total_latency,
            cache_hits=sum(1 for v in variations if v.cache_hit),
            timestamp=start_time,
            metadata={
                "num_variations": len(variations),
                "variation_strategy": self.config.variation_strategy.value,
                "router_stats": self._router.get_stats(),
            },
        )


class LightningArena:
    """
    Optimized tournament runner for shadow verification.

    Uses efficient comparison strategies to minimize LLM calls
    while maintaining ranking quality.
    """

    def __init__(
        self,
        judge: Optional[Callable] = None,
        max_comparisons: Optional[int] = None,
    ):
        self._judge = judge
        self.max_comparisons = max_comparisons
        self._comparison_count = 0

    async def rank_variations(
        self,
        variations: list[GenerationResult],
        input_query: str,
    ) -> list[tuple[int, float]]:
        """
        Rank variations by quality.

        Returns list of (index, score) tuples sorted by score descending.
        """
        if not variations:
            return []

        n = len(variations)

        if n == 1:
            return [(0, 1.0)]

        # Use anchor-based comparison for efficiency
        # Compare all to the first result
        scores = [0.0] * n

        for i in range(1, n):
            if self.max_comparisons and self._comparison_count >= self.max_comparisons:
                break

            score = await self._compare_pair(
                variations[0],
                variations[i],
                input_query,
            )
            self._comparison_count += 1

            # Score relative to anchor
            scores[0] += score
            scores[i] += 1.0 - score

        # Normalize scores
        max_score = max(scores) if scores else 1.0
        normalized = [(i, s / max(max_score, 1e-6)) for i, s in enumerate(scores)]

        # Sort by score descending
        return sorted(normalized, key=lambda x: x[1], reverse=True)

    async def _compare_pair(
        self,
        a: GenerationResult,
        b: GenerationResult,
        input_query: str,
    ) -> float:
        """Compare two variations, return score for A (0.0 to 1.0)."""
        if self._judge is None:
            # Mock comparison based on simple heuristics
            return self._mock_compare(a, b)

        return await self._judge(a, b, input_query)

    def _mock_compare(self, a: GenerationResult, b: GenerationResult) -> float:
        """Mock comparison for testing."""
        # Prefer lower temperatures (more deterministic)
        temp_a = a.config.temperature
        temp_b = b.config.temperature

        # Prefer longer, more detailed responses
        len_a = len(a.content)
        len_b = len(b.content)

        score = 0.5

        if temp_a < temp_b:
            score += 0.1
        elif temp_b < temp_a:
            score -= 0.1

        if len_a > len_b:
            score += 0.1
        elif len_b > len_a:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def reset_stats(self) -> None:
        """Reset comparison statistics."""
        self._comparison_count = 0

    def get_stats(self) -> dict[str, Any]:
        """Get arena statistics."""
        return {
            "total_comparisons": self._comparison_count,
            "max_comparisons": self.max_comparisons,
        }
