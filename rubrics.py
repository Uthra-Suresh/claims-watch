# Copyright (c) OpenEnvHackathon
# All rights reserved.
#
# ClaimWatch — Rubric system (OpenEnv RFC 004).

"""
Rubrics for the ClaimWatch environment.

Follows the OpenEnv Rubric system (RFC 004) to provide composable,
outcome-based and process-based rewards suitable for RL training
(GRPO, PPO, etc.).

The rubric system separates reward computation from the environment
logic, making rewards composable and swappable without changing the
environment itself.

Available rubrics:

- ``RoutingAccuracyRubric``: Outcome rubric — 1.0 if the routing
  decision matches ground truth, 0.0 otherwise.
- ``SLAComplianceRubric``: Process rubric — rewards timely processing,
  penalises SLA misses.
- ``FraudDetectionRubric``: Outcome rubric — rewards correct fraud
  flags, penalises missed fraud.
- ``ClaimWatchRubric``: Composite rubric combining outcome + process
  signals with configurable weights.

Usage:

    >>> from rubrics import ClaimWatchRubric
    >>> rubric = ClaimWatchRubric()
    >>> reward = rubric.forward(action, observation)

Custom rubrics can replace any component:

    >>> from rubrics import ClaimWatchRubric, RoutingAccuracyRubric
    >>> rubric = ClaimWatchRubric(
    ...     outcome=RoutingAccuracyRubric(partial_credit=True),
    ...     fraud_weight=0.3,
    ... )
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from openenv.core.rubrics.base import Rubric


class RoutingAccuracyRubric(Rubric):
    """Outcome rubric: reward based on routing decision correctness.

    Returns 1.0 if the agent's decision matches the ground truth,
    0.0 otherwise. When ``partial_credit=True``, returns 0.5 for
    adjacent decisions (e.g. clinical_review when md_review expected).

    The expected routing is injected via ``set_expected()`` at reset time.
    """

    ADJACENT_DECISIONS = {
        ("clinical_review", "md_review"),
        ("md_review", "clinical_review"),
        ("auto_approve", "clinical_review"),
    }

    def __init__(self, partial_credit: bool = False) -> None:
        super().__init__()
        self._expected: Optional[str] = None
        self._partial_credit = partial_credit

    def set_expected(self, expected: Optional[str]) -> None:
        self._expected = expected

    def forward(self, action: Any, observation: Any) -> float:
        if self._expected is None:
            return 0.0
        if not getattr(observation, "done", False):
            return 0.0

        decision = getattr(action, "decision", None)
        if decision is None:
            return 0.0

        decision_str = str(decision).strip().lower()
        expected_str = str(self._expected).strip().lower()

        if decision_str == expected_str:
            return 1.0

        if self._partial_credit:
            pair = (decision_str, expected_str)
            if pair in self.ADJACENT_DECISIONS:
                return 0.5

        return 0.0

    def reset(self) -> None:
        self._expected = None


class SLAComplianceRubric(Rubric):
    """Process rubric: per-step signal based on SLA compliance.

    Returns a small positive reward for processing a claim before its
    SLA deadline, a negative reward for SLA misses, and 0.0 for
    non-critical steps.
    """

    def __init__(
        self,
        on_time_reward: float = 0.10,
        miss_penalty: float = -0.15,
    ) -> None:
        super().__init__()
        self.on_time_reward = on_time_reward
        self.miss_penalty = miss_penalty

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        if not isinstance(metadata, dict):
            return 0.0

        sla_met = metadata.get("sla_met")
        if sla_met is True:
            return self.on_time_reward
        elif sla_met is False:
            return self.miss_penalty
        return 0.0


class FraudDetectionRubric(Rubric):
    """Outcome rubric for fraud detection quality.

    Returns +0.45 for correct fraud flags, -0.35 for missed fraud,
    0.0 for non-fraud claims.
    """

    def __init__(
        self,
        correct_flag_reward: float = 0.45,
        missed_fraud_penalty: float = -0.35,
    ) -> None:
        super().__init__()
        self.correct_flag_reward = correct_flag_reward
        self.missed_fraud_penalty = missed_fraud_penalty

    def forward(self, action: Any, observation: Any) -> float:
        metadata = getattr(observation, "metadata", {})
        if not isinstance(metadata, dict):
            return 0.0

        fraud_result = metadata.get("fraud_result")
        if fraud_result == "correct_flag":
            return self.correct_flag_reward
        elif fraud_result == "missed_fraud":
            return self.missed_fraud_penalty
        return 0.0


class CustomMetricRubric(Rubric):
    """Outcome rubric using a user-provided metric function.

    Mirrors the DSPy GRPO pattern where the user provides
    ``metric(expected, predicted) -> float``.
    """

    def __init__(self, metric_fn: Callable[[str, str], float]) -> None:
        super().__init__()
        self._metric_fn = metric_fn
        self._expected: Optional[str] = None

    def set_expected(self, expected: Optional[str]) -> None:
        self._expected = expected

    def forward(self, action: Any, observation: Any) -> float:
        if self._expected is None:
            return 0.0
        if not getattr(observation, "done", False):
            return 0.0

        metadata = getattr(observation, "metadata", {})
        grader = metadata.get("grader_result", {})
        score = grader.get("score")
        if score is not None:
            return float(score)
        return 0.0

    def reset(self) -> None:
        self._expected = None


class ClaimWatchRubric(Rubric):
    """Composite rubric for the ClaimWatch environment.

    Combines outcome-based reward (routing correctness) with
    process-based reward (SLA compliance) and fraud detection signals.

    The outcome rubric is evaluated on terminal steps (``done=True``).
    The process and fraud rubrics are evaluated on every step.

    Default weights:
        - routing_weight: 0.50 (outcome)
        - sla_weight:     0.30 (process)
        - fraud_weight:   0.20 (outcome)

    Example:
        >>> rubric = ClaimWatchRubric()
        >>> reward = rubric.forward(action, observation)

        >>> rubric = ClaimWatchRubric(
        ...     outcome=RoutingAccuracyRubric(partial_credit=True),
        ...     fraud_weight=0.3,
        ... )
    """

    def __init__(
        self,
        outcome: Optional[Rubric] = None,
        process: Optional[Rubric] = None,
        fraud: Optional[Rubric] = None,
        routing_weight: float = 0.50,
        sla_weight: float = 0.30,
        fraud_weight: float = 0.20,
        failure_reward: float = -0.10,
    ) -> None:
        super().__init__()
        self.outcome = outcome or RoutingAccuracyRubric()
        self.process = process or SLAComplianceRubric()
        self.fraud = fraud or FraudDetectionRubric()
        self.routing_weight = routing_weight
        self.sla_weight = sla_weight
        self.fraud_weight = fraud_weight
        self.failure_reward = failure_reward

    def set_expected(self, expected: Optional[str]) -> None:
        """Pass expected routing to the outcome rubric."""
        if hasattr(self.outcome, "set_expected"):
            self.outcome.set_expected(expected)

    def forward(self, action: Any, observation: Any) -> float:
        done = getattr(observation, "done", False)

        process_reward = self.process(action, observation)
        fraud_reward = self.fraud(action, observation)

        if done:
            metadata = getattr(observation, "metadata", {})
            grader = metadata.get("grader_result", {})
            score = grader.get("score")
            if score is not None:
                return float(score)
            outcome_reward = self.outcome(action, observation)
            total = (
                self.routing_weight * outcome_reward
                + self.sla_weight * process_reward
                + self.fraud_weight * fraud_reward
            )
            return max(-1.0, min(1.0, total))

        # Non-terminal step: process + fraud only
        return max(-1.0, min(1.0, process_reward + fraud_reward))

    def reset(self) -> None:
        self.outcome.reset()
        self.process.reset()
        self.fraud.reset()
