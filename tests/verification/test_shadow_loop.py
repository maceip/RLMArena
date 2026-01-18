"""
Tests for Shadow Loop and parallel generation.
"""

import pytest

from qqr.verification.shadow_loop import (
    ShadowLoop,
    ShadowLoopConfig,
    VariationStrategy,
    GenerationConfig,
    LightningArena,
    RadixAttentionCache,
    MockModelPredictor,
)


class TestRadixAttentionCache:
    """Tests for RadixAttention prefix cache."""

    @pytest.fixture
    def cache(self):
        return RadixAttentionCache(max_tokens=1000)

    def test_store_and_retrieve_prefix(self, cache):
        messages = [
            {"role": "user", "content": "Hello world"},
        ]

        cache.store_prefix(messages)
        cache_key, prefix_len = cache.get_prefix(messages)

        assert cache_key is not None
        assert prefix_len == 1

    def test_partial_prefix_match(self, cache):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]

        # Store first two messages
        cache.store_prefix(messages[:2])

        # Query with all three - should match first two
        cache_key, prefix_len = cache.get_prefix(messages)
        assert cache_key is not None
        assert prefix_len == 2

    def test_cache_eviction(self, cache):
        cache = RadixAttentionCache(max_tokens=100)

        # Store many prefixes
        for i in range(20):
            cache.store_prefix([{"role": "user", "content": f"Message {i}" * 10}])

        stats = cache.stats()
        assert stats["total_tokens"] <= stats["max_tokens"]


class TestShadowLoop:
    """Tests for Shadow Loop."""

    @pytest.fixture
    def shadow_loop(self):
        return ShadowLoop(
            config=ShadowLoopConfig(
                num_variations=3,
                variation_strategy=VariationStrategy.TEMPERATURE_SWEEP,
                parallel_generation=True,
            ),
        )

    @pytest.mark.asyncio
    async def test_generate_variations(self, shadow_loop):
        messages = [
            {"role": "user", "content": "Write a hello world program"},
        ]

        variations = await shadow_loop.generate_variations(messages)

        assert len(variations) == 3
        for v in variations:
            assert v.content is not None
            assert v.config.temperature is not None

    @pytest.mark.asyncio
    async def test_run_shadow_loop(self, shadow_loop):
        messages = [
            {"role": "user", "content": "What is Python?"},
        ]

        result = await shadow_loop.run(messages)

        assert result.request_id is not None
        assert len(result.variations) > 0
        assert result.best_variation_index is not None

    @pytest.mark.asyncio
    async def test_run_with_scorer(self, shadow_loop):
        messages = [
            {"role": "user", "content": "Explain recursion"},
        ]

        def scorer(v):
            # Prefer longer responses
            return len(v.content)

        result = await shadow_loop.run(messages, scorer=scorer)

        assert result.best_score is not None
        # Best should have highest length
        best = result.variations[result.best_variation_index]
        for i, v in enumerate(result.variations):
            if i != result.best_variation_index:
                assert len(best.content) >= len(v.content)

    @pytest.mark.asyncio
    async def test_temperature_sweep_strategy(self):
        loop = ShadowLoop(
            config=ShadowLoopConfig(
                num_variations=4,
                variation_strategy=VariationStrategy.TEMPERATURE_SWEEP,
                temperature_range=(0.1, 0.9),
            ),
        )

        messages = [{"role": "user", "content": "Test"}]
        variations = await loop.generate_variations(messages)

        temperatures = [v.config.temperature for v in variations]
        assert min(temperatures) >= 0.1
        assert max(temperatures) <= 0.9


class TestLightningArena:
    """Tests for Lightning Arena tournament."""

    @pytest.fixture
    def arena(self):
        return LightningArena(max_comparisons=10)

    @pytest.mark.asyncio
    async def test_rank_single_variation(self, arena):
        variations = [
            type('MockResult', (), {'content': 'Response', 'config': GenerationConfig()})()
        ]

        rankings = await arena.rank_variations(variations, "Test query")

        assert len(rankings) == 1
        assert rankings[0] == (0, 1.0)

    @pytest.mark.asyncio
    async def test_rank_multiple_variations(self, arena):
        variations = [
            type('MockResult', (), {
                'content': 'Short',
                'config': GenerationConfig(temperature=0.5)
            })(),
            type('MockResult', (), {
                'content': 'This is a longer response with more detail',
                'config': GenerationConfig(temperature=0.3)
            })(),
        ]

        rankings = await arena.rank_variations(variations, "Test query")

        assert len(rankings) == 2
        # Rankings should be sorted by score descending
        assert rankings[0][1] >= rankings[1][1]

    def test_arena_stats(self, arena):
        stats = arena.get_stats()
        assert "total_comparisons" in stats
        assert "max_comparisons" in stats
