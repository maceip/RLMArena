"""
Shadow Arena: VaaS API Proxy

The Shadow Arena intercepts LLM requests, generates multiple variations,
runs them through the verification and tournament pipeline, and returns
only the highest-scoring, verified response.

This transforms RLMArena from a research tool into a production-ready
reliability engine (API proxy / quality gateway).
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from .composite_judge import CompositeJudge
from .plugin import AggregateVerificationResult, VerificationPlugin
from .self_correction import SelfCorrectionEngine


@dataclass
class ShadowArenaConfig:
    """Configuration for the Shadow Arena"""
    num_variations: int = 4
    max_correction_attempts: int = 2
    enable_self_correction: bool = True
    parallel_generation: bool = True
    return_all_results: bool = False
    timeout_seconds: float = 60.0
    min_valid_responses: int = 1


@dataclass
class ShadowArenaResult:
    """Result from Shadow Arena processing"""
    best_response: dict | None
    best_score: float
    all_responses: list[dict] = field(default_factory=list)
    verification_results: list[AggregateVerificationResult] = field(default_factory=list)
    tournament_scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.best_response is not None

    @property
    def valid_count(self) -> int:
        return sum(1 for v in self.verification_results if v.all_passed)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "best_response": self.best_response,
            "best_score": self.best_score,
            "valid_count": self.valid_count,
            "total_variations": len(self.all_responses),
            "metadata": self.metadata,
        }


class ShadowArena:
    """
    The Shadow Arena API Proxy for VaaS.

    Flow:
    1. Interception: Receive user query via MCP-compatible interface
    2. Parallel Generation: Spawn multiple response variations
    3. Verification: Run hard checks on all variations
    4. Self-Correction: Optionally retry failed variations
    5. Tournament: Run verified responses through arena ranking
    6. Selection: Return highest-scoring verified response

    Usage:
        plugins = [CodeLinterPlugin(), SafetyFilterPlugin()]
        judge = CompositeJudge(llm_judge, plugins)
        arena = ShadowArena(
            judge=judge,
            generate_fn=my_llm_generate_function,
            config=ShadowArenaConfig(num_variations=4),
        )

        result = await arena.process_query(
            query="Write a Python function to...",
            context={"tools": available_tools},
        )
        print(result.best_response)
    """

    def __init__(
        self,
        judge: CompositeJudge,
        generate_fn: Callable,
        config: ShadowArenaConfig | None = None,
        correction_engine: SelfCorrectionEngine | None = None,
    ):
        """
        Initialize the Shadow Arena.

        Args:
            judge: CompositeJudge for verification and tournament
            generate_fn: Async function to generate LLM responses
            config: Arena configuration
            correction_engine: Optional self-correction engine
        """
        self.judge = judge
        self.generate_fn = generate_fn
        self.config = config or ShadowArenaConfig()
        self.correction_engine = correction_engine or SelfCorrectionEngine(
            max_correction_attempts=self.config.max_correction_attempts
        )

    async def generate_variations(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        num_variations: int | None = None,
    ) -> list[list[dict]]:
        """
        Generate multiple response variations for a query.

        Args:
            query: The user query
            context: Optional context (tools, system prompt, etc.)
            num_variations: Override for number of variations

        Returns:
            List of message trajectories (each is list of message dicts)
        """
        n = num_variations or self.config.num_variations
        context = context or {}

        if self.config.parallel_generation:
            tasks = [
                self.generate_fn(query, context, variation_idx=i)
                for i in range(n)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for i in range(n):
                try:
                    result = await self.generate_fn(query, context, variation_idx=i)
                    results.append(result)
                except Exception as e:
                    results.append(e)
            return results

    async def verify_and_correct(
        self,
        trajectory: list[dict],
        query: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[list[dict], AggregateVerificationResult]:
        """
        Verify a trajectory and optionally attempt self-correction.

        Args:
            trajectory: The message trajectory
            query: The original query
            context: Optional context

        Returns:
            Tuple of (possibly corrected trajectory, final verification result)
        """
        result = await self.judge.verify_trajectory(trajectory, context)

        if result.all_passed or not self.config.enable_self_correction:
            return trajectory, result

        for attempt in range(self.config.max_correction_attempts):
            if not self.correction_engine.should_attempt_correction(result, attempt):
                break

            correction_msg = self.correction_engine.create_correction_message(result)
            corrected_trajectory = trajectory + [correction_msg]

            try:
                new_response = await self.generate_fn(
                    query,
                    context,
                    messages=corrected_trajectory,
                    is_correction=True,
                )
                trajectory = new_response
                result = await self.judge.verify_trajectory(trajectory, context)

                if result.all_passed:
                    break
            except Exception:
                break

        return trajectory, result

    async def run_tournament(
        self,
        trajectories: list[list[dict]],
        verification_results: list[AggregateVerificationResult],
        query: str,
    ) -> list[float]:
        """
        Run tournament ranking on verified trajectories.

        Args:
            trajectories: All generated trajectories
            verification_results: Verification results for each
            query: The original query

        Returns:
            List of scores (failed verifications get floor score)
        """
        valid_indices = [
            i for i, v in enumerate(verification_results) if v.all_passed
        ]

        if len(valid_indices) == 0:
            return [-1.0] * len(trajectories)

        if len(valid_indices) == 1:
            scores = [-1.0] * len(trajectories)
            scores[valid_indices[0]] = 1.0
            return scores

        scores = [-1.0] * len(trajectories)
        valid_trajectories = [trajectories[i] for i in valid_indices]

        comparison_scores = {}
        for i, idx_a in enumerate(valid_indices):
            for j, idx_b in enumerate(valid_indices):
                if i >= j:
                    continue
                score_a, score_b, _ = await self.judge.bidirectional_compare(
                    valid_trajectories[i],
                    valid_trajectories[j],
                    query=query,
                )
                comparison_scores[(idx_a, idx_b)] = (score_a, score_b)

        for idx in valid_indices:
            total_score = 0.0
            count = 0
            for (a, b), (sa, sb) in comparison_scores.items():
                if a == idx:
                    total_score += sa
                    count += 1
                elif b == idx:
                    total_score += sb
                    count += 1
            scores[idx] = total_score / count if count > 0 else 0.0

        return scores

    async def process_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> ShadowArenaResult:
        """
        Process a query through the full Shadow Arena pipeline.

        Args:
            query: The user query
            context: Optional context (tools, system prompt, etc.)

        Returns:
            ShadowArenaResult with best response and metadata
        """
        try:
            trajectories = await asyncio.wait_for(
                self.generate_variations(query, context),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ShadowArenaResult(
                best_response=None,
                best_score=-1.0,
                metadata={"error": "Generation timeout"},
            )

        valid_trajectories = []
        exceptions = []
        for i, t in enumerate(trajectories):
            if isinstance(t, Exception):
                exceptions.append((i, str(t)))
            else:
                valid_trajectories.append((i, t))

        if not valid_trajectories:
            return ShadowArenaResult(
                best_response=None,
                best_score=-1.0,
                metadata={"error": "All generations failed", "exceptions": exceptions},
            )

        corrected_trajectories = []
        verification_results = []

        for idx, trajectory in valid_trajectories:
            corrected, result = await self.verify_and_correct(trajectory, query, context)
            corrected_trajectories.append((idx, corrected))
            verification_results.append(result)

        all_trajectories = [t for _, t in corrected_trajectories]
        scores = await self.run_tournament(all_trajectories, verification_results, query)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_trajectory = all_trajectories[best_idx]
        best_score = scores[best_idx]

        best_response = None
        if verification_results[best_idx].all_passed:
            for msg in reversed(best_trajectory):
                if msg.get("role") == "assistant":
                    best_response = msg
                    break

        return ShadowArenaResult(
            best_response=best_response,
            best_score=best_score,
            all_responses=[t for _, t in corrected_trajectories] if self.config.return_all_results else [],
            verification_results=verification_results,
            tournament_scores=scores,
            metadata={
                "num_variations": len(trajectories),
                "valid_count": sum(1 for v in verification_results if v.all_passed),
                "exceptions": exceptions,
            },
        )
