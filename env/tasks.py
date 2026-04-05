"""ClaimWatch — Task configurations and grader functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import Claim, RoutingDecision, SLATier, SLA_HOURS


# ── TaskConfig ───────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    task_id: int
    name: str
    difficulty: str
    n_claims: int
    n_hospitals: int
    fraud_rate: float
    md_slots_per_day: int
    clinical_slots_per_day: int
    policy_update_day: int = -1
    seed: int = 42


TASKS: Dict[int, TaskConfig] = {
    1: TaskConfig(
        task_id=1,
        name="routine_triage",
        difficulty="easy",
        n_claims=50,
        n_hospitals=10,
        fraud_rate=0.06,
        md_slots_per_day=20,
        clinical_slots_per_day=40,
        policy_update_day=-1,
        seed=42,
    ),
    2: TaskConfig(
        task_id=2,
        name="multi_hospital_triage",
        difficulty="medium",
        n_claims=300,
        n_hospitals=50,
        fraud_rate=0.08,
        md_slots_per_day=8,
        clinical_slots_per_day=25,
        policy_update_day=150,
        seed=42,
    ),
    3: TaskConfig(
        task_id=3,
        name="full_complexity",
        difficulty="hard",
        n_claims=1000,
        n_hospitals=200,
        fraud_rate=0.10,
        md_slots_per_day=4,
        clinical_slots_per_day=15,
        policy_update_day=500,
        seed=42,
    ),
}


# ── Base metrics helper ──────────────────────────────────────────────────────

def _base_metrics(
    claims: List[Claim],
    decisions: Dict[str, RoutingDecision],
) -> Dict[str, float]:
    """Compute shared grading metrics."""
    total = len(claims)
    if total == 0:
        return {
            "routing_accuracy": 0.0,
            "sla_compliance_rate": 1.0,
            "fraud_detection_rate": 1.0,
            "false_denial_rate": 0.0,
            "unnecessary_review_rate": 0.0,
        }

    correct = 0
    sla_total = 0
    sla_correct = 0
    fraud_total = 0
    fraud_correct = 0
    false_denials = 0
    unnecessary_reviews = 0

    for claim in claims:
        decision = decisions.get(claim.claim_id)
        if decision is None:
            continue

        gt = claim.ground_truth_routing

        if decision == gt:
            correct += 1

        # SLA compliance: urgent/critical claims routed correctly
        if claim.sla_tier in (SLATier.urgent, SLATier.critical):
            sla_total += 1
            if decision == gt:
                sla_correct += 1

        if claim.is_fraud:
            fraud_total += 1
            if decision == RoutingDecision.flag_fraud:
                fraud_correct += 1

        if (not claim.is_fraud
                and decision == RoutingDecision.deny
                and gt != RoutingDecision.deny):
            false_denials += 1

        if (gt == RoutingDecision.auto_approve
                and decision in (RoutingDecision.clinical_review, RoutingDecision.md_review)):
            unnecessary_reviews += 1

    routing_accuracy = correct / total if total else 0.0
    sla_compliance_rate = sla_correct / sla_total if sla_total else 1.0
    fraud_detection_rate = fraud_correct / fraud_total if fraud_total else 1.0
    false_denial_rate = false_denials / total if total else 0.0
    unnecessary_review_rate = unnecessary_reviews / total if total else 0.0

    return {
        "routing_accuracy": routing_accuracy,
        "sla_compliance_rate": sla_compliance_rate,
        "fraud_detection_rate": fraud_detection_rate,
        "false_denial_rate": false_denial_rate,
        "unnecessary_review_rate": unnecessary_review_rate,
    }


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


# ── Grader functions ─────────────────────────────────────────────────────────

def grade_task1(
    claims: List[Claim],
    decisions: Dict[str, RoutingDecision],
) -> Dict[str, float]:
    """Grader for Task 1 — routine_triage."""
    m = _base_metrics(claims, decisions)
    score = _clamp(0.70 * m["routing_accuracy"] + 0.30 * m["sla_compliance_rate"])
    return {**m, "score": score}


def grade_task2(
    claims: List[Claim],
    decisions: Dict[str, RoutingDecision],
) -> Dict[str, float]:
    """Grader for Task 2 — multi_hospital_triage."""
    m = _base_metrics(claims, decisions)
    score = _clamp(
        0.40 * m["routing_accuracy"]
        + 0.35 * m["sla_compliance_rate"]
        + 0.15 * m["fraud_detection_rate"]
        + 0.10 * (1.0 - m["false_denial_rate"])
        - 0.05 * m["unnecessary_review_rate"]
    )
    return {**m, "score": score}


def grade_task3(
    claims: List[Claim],
    decisions: Dict[str, RoutingDecision],
) -> Dict[str, float]:
    """Grader for Task 3 — full_complexity."""
    m = _base_metrics(claims, decisions)

    raw = (
        0.40 * m["routing_accuracy"]
        + 0.30 * m["sla_compliance_rate"]
        + 0.20 * m["fraud_detection_rate"]
        + 0.10 * (1.0 - m["false_denial_rate"])
        - 0.05 * m["unnecessary_review_rate"]
    )

    multiplier = 1.0 if m["sla_compliance_rate"] >= 0.70 else m["sla_compliance_rate"] / 0.70
    score = _clamp(raw * multiplier)

    return {**m, "score": score}


GRADERS = {
    1: grade_task1,
    2: grade_task2,
    3: grade_task3,
}
