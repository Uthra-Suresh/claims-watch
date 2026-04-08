"""ClaimWatch — Core RL environment: reset / step / state."""

from __future__ import annotations

import random as _random_module
from typing import Any, Dict, List, Optional
from openenv.core.env_server.interfaces import Environment

from .models import (
    ClaimAction,
    Claim,
    ClaimSnapshot,
    ClaimState,
    ClaimStatus,
    ClaimObservation,
    Reward,
    RoutingDecision,
    SLA_HOURS,
    claim_to_snapshot,
    DECISION_TO_STATUS,
)
from .generator import generate_claims, recompute_ground_truth
from .policies import RESOURCE_COSTS
from .reward import compute_reward
from .tasks import TASKS, TaskConfig, GRADERS, grade_task3


class ClaimWatchEnv(Environment):
    """OpenEnv-compliant RL environment for insurance claims triage."""

    def __init__(self) -> None:
        super().__init__()
        self._task_config: Optional[TaskConfig] = None
        self._claims: List[Claim] = []
        self._claims_by_id: Dict[str, Claim] = {}
        self._pending_ids: set = set()
        self._decisions: Dict[str, RoutingDecision] = {}
        self._reward_history: List[float] = []
        self._cumulative_reward: float = 0.0

        self._step_number: int = 0
        self._current_day: int = 0
        self._current_hour: int = 8
        self._processed_today: int = 0

        self._md_slots_remaining: int = 0
        self._clinical_slots_remaining: int = 0

        self._policy_version: str = "v1"
        self._policy_update_fired: bool = False

        self._done: bool = True
        self._task_id: int = 1
        self._seed: int = 42

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, seed: int = 42, episode_id: Optional[str] = None, task_id: int = 1, n_claims: Optional[int] = None, **kwargs: Any) -> ClaimObservation:
        """Initialize a new episode."""
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        cfg = TASKS[task_id]
        self._task_config = cfg
        self._task_id = task_id
        self._seed = seed

        claim_count = n_claims or cfg.n_claims

        self._claims = generate_claims(
            n=claim_count,
            seed=seed,
            fraud_rate=cfg.fraud_rate,
            policy_version="v1",
        )

        self._claims_by_id = {c.claim_id: c for c in self._claims}
        self._pending_ids = {c.claim_id for c in self._claims}
        self._decisions = {}
        self._reward_history = []
        self._cumulative_reward = 0.0

        self._step_number = 0
        self._current_day = 0
        self._current_hour = 8
        self._processed_today = 0

        self._md_slots_remaining = cfg.md_slots_per_day
        self._clinical_slots_remaining = cfg.clinical_slots_per_day

        self._policy_version = "v1"
        self._policy_update_fired = False

        self._done = False

        return self._build_observation()

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: ClaimAction, timeout_s: Optional[float] = None, **kwargs: Any) -> ClaimObservation:
        """Process one agent action."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        cfg = self._task_config
        assert cfg is not None

        self._step_number += 1

        # Fire mid-episode mechanics
        self._check_policy_update()

        # Look up claim
        claim = self._claims_by_id.get(action.claim_id)
        if claim is None or claim.claim_id not in self._pending_ids:
            reward = Reward(total=-0.05)
            self._reward_history.append(reward.total)
            self._cumulative_reward += reward.total
            self._advance_clock()
            done = self._check_done()
            obs = self._build_observation()
            obs.done = done
            obs.reward = reward.total
            if done:
                obs.metadata["grader_result"] = self._run_grader()
            return obs

        decision = action.decision

        # Downgrade chain for exhausted slots
        if decision == RoutingDecision.md_review and self._md_slots_remaining <= 0:
            decision = RoutingDecision.clinical_review
        if decision == RoutingDecision.clinical_review and self._clinical_slots_remaining <= 0:
            decision = RoutingDecision.request_info

        # Consume resources
        cost = RESOURCE_COSTS[decision]
        self._md_slots_remaining -= cost.md_slots
        self._clinical_slots_remaining -= cost.clinical_slots

        # Update claim status
        claim.status = DECISION_TO_STATUS.get(decision, ClaimStatus.pending)
        self._decisions[claim.claim_id] = decision
        self._pending_ids.discard(claim.claim_id)
        self._processed_today += 1

        # Compute reward (pass current_hr for SLA tracking)
        reward = compute_reward(
            claim, decision, self._policy_version,
            current_hr=self._current_hour + self._current_day * 24,
        )
        self._reward_history.append(reward.total)
        self._cumulative_reward += reward.total

        # Advance clock
        self._advance_clock()

        # Check done
        done = self._check_done()
        obs = self._build_observation()
        obs.done = done
        obs.reward = reward.total
        if done:
            obs.metadata["grader_result"] = self._run_grader()
        return obs

    # ── State ────────────────────────────────────────────────────────────

    @property
    def state(self) -> ClaimState:
        """Return full environment state for validation."""
        return ClaimState(
            task_id=self._task_id,
            seed=self._seed,
            step_count=self._step_number,
            step_number=self._step_number,
            current_day=self._current_day,
            current_hour=self._current_hour,
            total_claims=len(self._claims),
            pending_count=len(self._pending_ids),
            decided_count=len(self._decisions),
            md_slots_remaining=self._md_slots_remaining,
            clinical_slots_remaining=self._clinical_slots_remaining,
            policy_update_fired=self._policy_update_fired,
            is_done=self._done,
            cumulative_reward=self._cumulative_reward,
        )

    def close(self) -> None:
        """Clean up environment resources."""
        self._done = True

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_observation(self) -> ClaimObservation:
        """Build the agent-visible observation."""
        cfg = self._task_config
        # Sort by SLA remaining (ascending) then billed amount (descending)
        abs_hr = self._current_hour + self._current_day * 24
        pending_claims = [self._claims_by_id[cid] for cid in self._pending_ids]
        pending_claims.sort(
            key=lambda c: (
                max(0.0, c.sla_deadline_hr - max(0, abs_hr - c.arrival_hr)),
                -c.billed_amount,
            )
        )
        visible = pending_claims[:50]
        queue = [claim_to_snapshot(c, current_hr=abs_hr) for c in visible]

        return ClaimObservation(
            current_hour=self._current_hour,
            current_day=self._current_day,
            queue=queue,
            processed_today=self._processed_today,
            md_slots_remaining=self._md_slots_remaining,
            clinical_slots_remaining=self._clinical_slots_remaining,
            policy_update_active=self._policy_version == "v2",
            total_claims_in_episode=len(self._claims),
            task_id=self._task_id,
            step_number=self._step_number,
        )

    def _advance_clock(self) -> None:
        """Advance day every 50 steps."""
        if self._step_number > 0 and self._step_number % 50 == 0:
            self._advance_day()
        self._current_hour = 8 + (self._step_number % 50) % 10

    def _advance_day(self) -> None:
        """Start a new day: reset slots and counters."""
        self._current_day += 1
        self._processed_today = 0
        cfg = self._task_config
        if cfg:
            self._md_slots_remaining = cfg.md_slots_per_day
            self._clinical_slots_remaining = cfg.clinical_slots_per_day
        for cid in self._pending_ids:
            self._claims_by_id[cid].days_in_queue += 1

    def _check_done(self) -> bool:
        """Check episode termination conditions."""
        cfg = self._task_config
        if cfg is None:
            self._done = True
            return True

        if len(self._pending_ids) == 0:
            self._done = True

        return self._done

    def _check_policy_update(self) -> None:
        """Fire policy update v1→v2 at the configured step."""
        cfg = self._task_config
        if cfg is None or self._policy_update_fired:
            return
        if cfg.policy_update_day > 0 and self._step_number >= cfg.policy_update_day:
            self._policy_version = "v2"
            self._policy_update_fired = True
            pending_claims = [self._claims_by_id[cid] for cid in self._pending_ids]
            recompute_ground_truth(pending_claims, "v2")

    def _run_grader(self) -> Dict[str, float]:
        """Run the task-specific grader."""
        cfg = self._task_config
        if cfg is None:
            return {"score": 0.0}

        task_id = cfg.task_id
        if task_id == 3:
            return grade_task3(
                self._claims,
                self._decisions,
            )
        grader = GRADERS.get(task_id)
        if grader is None:
            return {"score": 0.0}
        return grader(self._claims, self._decisions)
