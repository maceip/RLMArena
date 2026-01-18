"""
DSPy Optimizer for programmatic judge tuning.

This module uses DSPy to treat the CompositeJudge as a compilable program,
enabling automatic optimization of judge instructions using MIPROv2.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
from datetime import datetime
import json
import random


class Predictor(Protocol):
    """Protocol for DSPy-compatible predictors."""

    def __call__(self, **kwargs) -> dict[str, Any]:
        ...


@dataclass
class Example:
    """A training example for judge optimization."""
    id: str
    input_query: str
    trajectory_a: list[dict[str, Any]]
    trajectory_b: list[dict[str, Any]]
    expected_winner: str  # "a", "b", or "tie"
    confidence: float
    rationales: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_query": self.input_query,
            "trajectory_a": self.trajectory_a,
            "trajectory_b": self.trajectory_b,
            "expected_winner": self.expected_winner,
            "confidence": self.confidence,
            "rationales": self.rationales,
        }


@dataclass
class OptimizationConfig:
    """Configuration for judge optimization."""
    num_candidates: int = 10
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    num_trials: int = 30
    minibatch_size: int = 25
    minibatch_full_eval_steps: int = 10
    requires_permission_to_run: bool = False
    seed: int = 42
    verbose: bool = True


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    best_instructions: str
    best_score: float
    improvement: float
    num_trials: int
    training_examples: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trial_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Signature:
    """
    DSPy-style signature for the judge comparison task.

    This defines the input/output schema for the judge module.
    """

    def __init__(
        self,
        instructions: str = "",
        input_fields: Optional[list[str]] = None,
        output_fields: Optional[list[str]] = None,
    ):
        self.instructions = instructions
        self.input_fields = input_fields or [
            "input_query",
            "trajectory_a",
            "trajectory_b",
        ]
        self.output_fields = output_fields or [
            "winner",
            "confidence",
            "rationale",
        ]

    def to_prompt_template(self) -> str:
        """Convert signature to a prompt template."""
        template = f"{self.instructions}\n\n"
        template += "Input Fields:\n"
        for field in self.input_fields:
            template += f"  - {field}: {{{field}}}\n"
        template += "\nOutput Fields:\n"
        for field in self.output_fields:
            template += f"  - {field}\n"
        return template


class JudgeModule:
    """
    DSPy-style module wrapping the LLM judge comparison logic.

    This module can be optimized using MIPROv2 to tune its instructions.
    """

    def __init__(
        self,
        signature: Optional[Signature] = None,
        predictor: Optional[Predictor] = None,
    ):
        self.signature = signature or Signature(
            instructions=self._default_instructions()
        )
        self._predictor = predictor
        self._compiled = False

    def _default_instructions(self) -> str:
        return """You are an expert judge comparing two AI agent trajectories.
Your task is to determine which trajectory better accomplishes the user's goal
while adhering to best practices, security guidelines, and code quality standards.

Consider the following criteria:
1. Correctness: Does the solution achieve the intended outcome?
2. Security: Are there any security vulnerabilities or policy violations?
3. Efficiency: Is the solution reasonably efficient?
4. Clarity: Is the code/response clear and maintainable?
5. Completeness: Does the solution fully address the user's request?

Provide your judgment as:
- winner: "a", "b", or "tie"
- confidence: 0.0 to 1.0
- rationale: Brief explanation of your reasoning"""

    def forward(
        self,
        input_query: str,
        trajectory_a: list[dict[str, Any]],
        trajectory_b: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute the judge comparison."""
        if self._predictor is None:
            # Mock prediction for testing
            return self._mock_predict(input_query, trajectory_a, trajectory_b)

        return self._predictor(
            input_query=input_query,
            trajectory_a=trajectory_a,
            trajectory_b=trajectory_b,
            instructions=self.signature.instructions,
        )

    def _mock_predict(
        self,
        input_query: str,
        trajectory_a: list[dict[str, Any]],
        trajectory_b: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Mock prediction for testing without an actual LLM."""
        # Simple heuristic based on trajectory length and content
        score_a = len(trajectory_a)
        score_b = len(trajectory_b)

        # Check for error indicators
        a_str = json.dumps(trajectory_a)
        b_str = json.dumps(trajectory_b)

        if "error" in a_str.lower():
            score_a -= 5
        if "error" in b_str.lower():
            score_b -= 5

        if score_a > score_b:
            return {"winner": "a", "confidence": 0.7, "rationale": "Trajectory A appears more complete"}
        elif score_b > score_a:
            return {"winner": "b", "confidence": 0.7, "rationale": "Trajectory B appears more complete"}
        else:
            return {"winner": "tie", "confidence": 0.5, "rationale": "Trajectories appear equivalent"}

    def __call__(self, **kwargs) -> dict[str, Any]:
        return self.forward(**kwargs)


class MockMIPROv2:
    """
    Mock MIPROv2 optimizer for instruction tuning.

    In production, this would use the actual DSPy MIPROv2 optimizer.
    This mock simulates the optimization process for testing.
    """

    def __init__(
        self,
        metric: Callable[[dict[str, Any], Example], float],
        config: Optional[OptimizationConfig] = None,
    ):
        self.metric = metric
        self.config = config or OptimizationConfig()
        self._candidate_instructions = self._generate_candidate_instructions()

    def _generate_candidate_instructions(self) -> list[str]:
        """Generate candidate instruction variations."""
        base = "You are an expert judge comparing two AI agent trajectories."

        variations = [
            # Security-focused
            f"""{base}

CRITICAL: Security is paramount. Any trajectory with security vulnerabilities,
credential exposure, or policy violations should be marked as the loser.

Evaluation priority:
1. Security compliance (most important)
2. Correctness of solution
3. Code quality and best practices
4. Completeness""",

            # Correctness-focused
            f"""{base}

Focus primarily on whether the solution correctly solves the user's problem.
A working solution with minor style issues beats a stylistically perfect
non-working solution.

Evaluation priority:
1. Does it work correctly?
2. Does it fully address the request?
3. Are there any bugs or errors?
4. Is it maintainable?""",

            # Balanced with rationale emphasis
            f"""{base}

Provide detailed reasoning for your judgment. Consider:
- Functional correctness: Does it achieve the goal?
- Security: Any vulnerabilities or risky patterns?
- Efficiency: Reasonable performance?
- Clarity: Is it understandable and maintainable?

Your rationale should cite specific evidence from the trajectories.""",

            # Expert engineer perspective
            f"""{base}

Judge as a senior software engineer would during code review.
Ask yourself: "Which solution would I approve for production?"

Red flags that should cause immediate failure:
- Hardcoded credentials or secrets
- SQL injection or XSS vulnerabilities
- Unvalidated user input
- Missing error handling for critical paths
- Resource leaks""",

            # Structured evaluation
            f"""{base}

Evaluate using this structured approach:

STEP 1: Check for critical failures
- Security vulnerabilities?
- Syntax errors?
- Policy violations?
If either trajectory has critical failures, it loses.

STEP 2: Compare correctness
- Which better achieves the user's goal?

STEP 3: Compare quality
- Code style, efficiency, maintainability

Base your confidence on how clear the difference is between trajectories.""",
        ]

        return variations

    def compile(
        self,
        module: JudgeModule,
        trainset: list[Example],
        valset: Optional[list[Example]] = None,
    ) -> tuple[JudgeModule, OptimizationResult]:
        """
        Compile/optimize the judge module using training examples.

        Returns the optimized module and optimization results.
        """
        random.seed(self.config.seed)

        if valset is None:
            # Split trainset
            random.shuffle(trainset)
            split = int(len(trainset) * 0.8)
            train = trainset[:split]
            val = trainset[split:]
        else:
            train = trainset
            val = valset

        # Evaluate baseline
        baseline_score = self._evaluate(module, val)

        trial_history = []
        best_instructions = module.signature.instructions
        best_score = baseline_score

        # Try each candidate
        for trial, instructions in enumerate(self._candidate_instructions):
            if trial >= self.config.num_trials:
                break

            # Create modified module
            test_module = JudgeModule(
                signature=Signature(instructions=instructions),
                predictor=module._predictor,
            )

            # Evaluate
            score = self._evaluate(test_module, val)

            trial_history.append({
                "trial": trial,
                "score": score,
                "instructions_preview": instructions[:100] + "...",
            })

            if score > best_score:
                best_score = score
                best_instructions = instructions

            if self.config.verbose:
                print(f"Trial {trial + 1}: score={score:.4f} (best={best_score:.4f})")

        # Create optimized module
        optimized_module = JudgeModule(
            signature=Signature(instructions=best_instructions),
            predictor=module._predictor,
        )
        optimized_module._compiled = True

        result = OptimizationResult(
            best_instructions=best_instructions,
            best_score=best_score,
            improvement=best_score - baseline_score,
            num_trials=len(trial_history),
            training_examples=len(train),
            trial_history=trial_history,
            metadata={
                "baseline_score": baseline_score,
                "validation_size": len(val),
                "config": {
                    "num_candidates": self.config.num_candidates,
                    "max_bootstrapped_demos": self.config.max_bootstrapped_demos,
                },
            },
        )

        return optimized_module, result

    def _evaluate(self, module: JudgeModule, examples: list[Example]) -> float:
        """Evaluate module on examples."""
        if not examples:
            return 0.0

        scores = []
        for example in examples:
            prediction = module.forward(
                input_query=example.input_query,
                trajectory_a=example.trajectory_a,
                trajectory_b=example.trajectory_b,
            )
            score = self.metric(prediction, example)
            scores.append(score)

        return sum(scores) / len(scores)


def accuracy_metric(prediction: dict[str, Any], example: Example) -> float:
    """Compute accuracy metric for judge predictions."""
    predicted_winner = prediction.get("winner", "").lower()
    expected_winner = example.expected_winner.lower()

    if predicted_winner == expected_winner:
        # Bonus for high confidence on correct predictions
        confidence = prediction.get("confidence", 0.5)
        return 0.5 + 0.5 * confidence
    else:
        return 0.0


def weighted_accuracy_metric(prediction: dict[str, Any], example: Example) -> float:
    """Compute accuracy weighted by example confidence."""
    base_accuracy = accuracy_metric(prediction, example)
    return base_accuracy * example.confidence


class DSPyJudgeOptimizer:
    """
    High-level interface for optimizing judge modules with DSPy.

    This class manages the optimization lifecycle:
    1. Load gold comparisons from ExpertAlignerService
    2. Convert to DSPy examples
    3. Run MIPROv2 optimization
    4. Export optimized instructions
    """

    def __init__(
        self,
        metric: Optional[Callable] = None,
        config: Optional[OptimizationConfig] = None,
    ):
        self.metric = metric or weighted_accuracy_metric
        self.config = config or OptimizationConfig()
        self._optimizer = MockMIPROv2(self.metric, self.config)
        self._optimization_history: list[OptimizationResult] = []

    def examples_from_gold_comparisons(
        self,
        comparisons: list[dict[str, Any]],
    ) -> list[Example]:
        """Convert gold comparisons to DSPy examples."""
        examples = []

        for comp in comparisons:
            if not comp.get("consensus_winner"):
                continue

            examples.append(Example(
                id=comp["id"],
                input_query=comp["input"],
                trajectory_a=comp["chosen"] if comp["consensus_winner"] == "a" else comp["rejected"],
                trajectory_b=comp["rejected"] if comp["consensus_winner"] == "a" else comp["chosen"],
                expected_winner="a" if comp["consensus_winner"] == "a" else "b",
                confidence=comp.get("confidence", 0.8),
                rationales=comp.get("rationales", []),
            ))

        return examples

    def optimize(
        self,
        module: JudgeModule,
        examples: list[Example],
        validation_split: float = 0.2,
    ) -> tuple[JudgeModule, OptimizationResult]:
        """
        Optimize the judge module using provided examples.

        Args:
            module: The JudgeModule to optimize
            examples: Training examples
            validation_split: Fraction of examples to use for validation

        Returns:
            Tuple of (optimized_module, optimization_result)
        """
        # Split into train/val
        random.seed(self.config.seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - validation_split))
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]

        # Run optimization
        optimized, result = self._optimizer.compile(module, train, val)

        self._optimization_history.append(result)

        return optimized, result

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of all optimization runs."""
        return self._optimization_history.copy()

    def export_optimized_prompt(
        self,
        result: OptimizationResult,
        format: str = "text",
    ) -> str:
        """Export optimized instructions in specified format."""
        if format == "text":
            return result.best_instructions
        elif format == "json":
            return json.dumps({
                "instructions": result.best_instructions,
                "score": result.best_score,
                "improvement": result.improvement,
                "timestamp": result.timestamp.isoformat(),
            }, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")


def create_judge_program(
    predictor: Optional[Predictor] = None,
    instructions: Optional[str] = None,
) -> JudgeModule:
    """Factory function to create a judge program for optimization."""
    signature = Signature(instructions=instructions) if instructions else None
    return JudgeModule(signature=signature, predictor=predictor)
