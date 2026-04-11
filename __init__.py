"""ClaimWatch — Insurance Claims Triage RL Environment.

Usage (async):
    >>> from client import ClaimWatchClient
    >>> from env.models import ClaimAction
    >>> async with ClaimWatchClient(base_url="http://localhost:8000") as env:
    ...     result = await env.reset(task_id=1, seed=42)
    ...     result = await env.step(ClaimAction(
    ...         claim_id="CLM-42-00001", decision="auto_approve"
    ...     ))

Usage (sync):
    >>> with ClaimWatchClient(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(task_id=1, seed=42)
"""

from client import ClaimWatchClient
from env.models import (
    ClaimAction,
    ClaimObservation,
    ClaimState,
    Reward,
    RoutingDecision,
    Claim,
    ClaimSnapshot,
)
from env.claim_env import ClaimWatchEnv
from rubrics import (
    ClaimWatchRubric,
    RoutingAccuracyRubric,
    SLAComplianceRubric,
    FraudDetectionRubric,
    CustomMetricRubric,
)

__all__ = [
    # Client
    "ClaimWatchClient",
    # Environment
    "ClaimWatchEnv",
    # Models
    "ClaimAction",
    "ClaimObservation",
    "ClaimState",
    "Reward",
    "RoutingDecision",
    "Claim",
    "ClaimSnapshot",
    # Rubrics
    "ClaimWatchRubric",
    "RoutingAccuracyRubric",
    "SLAComplianceRubric",
    "FraudDetectionRubric",
    "CustomMetricRubric",
]
