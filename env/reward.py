"""ClaimWatch — Reward computation engine."""

from __future__ import annotations

from .models import Claim, Reward, RoutingDecision, SLA_HOURS


def compute_reward(
    claim: Claim,
    decision: RoutingDecision,
    policy_version: str = "v1",
    current_hr: int = 0,
) -> Reward:
    """Compute a fully-decomposed reward for a single (claim, decision) pair.

    Deadline logic: if current_hr >= arrival_hr + sla_deadline_hr the claim
    has missed its SLA window.
    """
    gt = claim.ground_truth_routing
    r = Reward()

    elapsed = max(0, current_hr - claim.arrival_hr)
    on_time = elapsed < claim.sla_deadline_hr

    if decision == gt:
        # Correct decision — assign positive reward
        if gt == RoutingDecision.auto_approve:
            r.correct_auto_approve = 0.30
        elif gt == RoutingDecision.deny:
            r.correct_denial = 0.40
        elif gt == RoutingDecision.clinical_review:
            r.correct_clinical_route = 0.25
        elif gt == RoutingDecision.request_info:
            r.correct_rfi = 0.20
        elif gt == RoutingDecision.flag_fraud:
            r.correct_fraud_flag = 0.45
        elif gt == RoutingDecision.md_review:
            r.correct_clinical_route = 0.25

        # Deadline bonus/penalty
        if on_time:
            r.deadline_bonus = 0.10
        else:
            r.deadline_miss_penalty = -0.15
    else:
        # Incorrect decision — assign penalties

        # Missed fraud
        if (claim.is_fraud
                and gt == RoutingDecision.flag_fraud
                and decision in (RoutingDecision.auto_approve, RoutingDecision.clinical_review)):
            r.missed_fraud_penalty = -0.35

        # False denial
        if (not claim.is_fraud
                and decision == RoutingDecision.deny
                and gt != RoutingDecision.deny):
            r.false_denial_penalty = -0.60

        # Unnecessary review
        if (gt == RoutingDecision.auto_approve
                and decision in (RoutingDecision.clinical_review, RoutingDecision.md_review)):
            r.unnecessary_review_penalty = -0.20

        # Redundant RFI
        if (decision == RoutingDecision.request_info
                and gt != RoutingDecision.request_info):
            r.redundant_rfi_penalty = -0.15

        # Slot-level mismatch / wrong routing
        if (decision == RoutingDecision.md_review and gt == RoutingDecision.clinical_review):
            r.wrong_routing_penalty = -0.10
        elif (decision == RoutingDecision.clinical_review and gt == RoutingDecision.md_review):
            r.wrong_routing_penalty = -0.10
        elif (r.false_denial_penalty == 0.0 and r.missed_fraud_penalty == 0.0
              and r.unnecessary_review_penalty == 0.0 and r.redundant_rfi_penalty == 0.0):
            r.wrong_routing_penalty = -0.10

    # Compute total (clamped to [-1.0, 1.0])
    total = (
        r.correct_auto_approve
        + r.correct_denial
        + r.correct_clinical_route
        + r.correct_rfi
        + r.correct_fraud_flag
        + r.deadline_bonus
        + r.deadline_miss_penalty
        + r.false_denial_penalty
        + r.unnecessary_review_penalty
        + r.redundant_rfi_penalty
        + r.missed_fraud_penalty
        + r.wrong_routing_penalty
    )
    r.total = max(-1.0, min(1.0, total))

    return r
