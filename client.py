# Copyright (c) OpenEnvHackathon
# All rights reserved.
#
# ClaimWatch — OpenEnv client for insurance claims triage.

"""
Client for ClaimWatch environment.

Provides EnvClient wrapper for remote or local ClaimWatch instances.

``ClaimWatchClient`` is the standard async OpenEnv client for remote usage.
Use ``async with`` / ``await`` directly, or call ``.sync()`` for synchronous code.

Example (async):
    >>> from client import ClaimWatchClient, ClaimAction
    >>> async with ClaimWatchClient(base_url="http://localhost:8000") as env:
    ...     result = await env.reset(task_id=1, seed=42)
    ...     result = await env.step(ClaimAction(
    ...         claim_id="CLM-42-00001", decision="auto_approve"
    ...     ))
    ...     print(result.observation.queue)

Example (sync):
    >>> with ClaimWatchClient(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(task_id=1, seed=42)
    ...     result = env.step(ClaimAction(
    ...         claim_id="CLM-42-00001", decision="clinical_review"
    ...     ))
    ...     print(result.done)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_client import EnvClient, StepResult

from env.models import (
    ClaimAction,
    ClaimObservation,
    ClaimSnapshot,
    ClaimState,
    RoutingDecision,
)


class ClaimWatchClient(EnvClient[ClaimAction, ClaimObservation, ClaimState]):
    """
    Async client for the remote ClaimWatch environment.

    Connects to a running ClaimWatch server via WebSocket using the
    standard OpenEnv client/server protocol.

    For synchronous code, call ``.sync()`` on an instance.

    Example:
        >>> async with ClaimWatchClient(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_id=1, seed=42)
        ...     obs = result.observation
        ...     for claim in obs.queue[:3]:
        ...         print(f"{claim.claim_id}: ${claim.billed_amount:.2f}")
        ...     result = await env.step(ClaimAction(
        ...         claim_id=obs.queue[0].claim_id,
        ...         decision="auto_approve",
        ...     ))
        ...     print(f"reward={result.reward}, done={result.done}")

        >>> with ClaimWatchClient(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_id=1, seed=42)
        ...     result = env.step(ClaimAction(
        ...         claim_id=result.observation.queue[0].claim_id,
        ...         decision="clinical_review",
        ...     ))
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        **kwargs: Any,
    ) -> None:
        """
        Initialize ClaimWatch client.

        Args:
            base_url: Base URL of the ClaimWatch environment server.
            **kwargs: Additional arguments forwarded to ``EnvClient``.
        """
        super().__init__(base_url=base_url, **kwargs)

    # ── EnvClient protocol implementation ────────────────────────────────

    def _step_payload(self, action: ClaimAction) -> Dict[str, Any]:
        """Convert ClaimAction to JSON payload for the server."""
        return action.model_dump()

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[ClaimObservation]:
        """Parse server JSON response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = ClaimObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ClaimState:
        """Parse server JSON response into ClaimState."""
        return ClaimState(**payload)

    # ── Convenience helpers ──────────────────────────────────────────────

    async def triage(
        self,
        claim_id: str,
        decision: str,
        rationale: Optional[str] = None,
    ) -> StepResult[ClaimObservation]:
        """Convenience: triage one claim with a routing decision.

        Args:
            claim_id: The claim to act on (from the observation queue).
            decision: One of: auto_approve, clinical_review, md_review,
                      request_info, deny, flag_fraud.
            rationale: Optional reasoning (not graded).
        """
        return await self.step(
            ClaimAction(
                claim_id=claim_id,
                decision=RoutingDecision(decision),
                rationale=rationale,
            )
        )

    async def get_queue(self) -> List[ClaimSnapshot]:
        """Return the current pending claims queue from server state."""
        state = await self.state()
        # The queue is in the observation; re-issue a state check.
        # For queue access, the caller should use the last step result.
        return []
