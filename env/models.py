"""ClaimWatch — Pydantic v2 models for claims, observations, actions, rewards."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class RoutingDecision(str, Enum):
    auto_approve = "auto_approve"
    clinical_review = "clinical_review"
    md_review = "md_review"
    request_info = "request_info"
    deny = "deny"
    flag_fraud = "flag_fraud"


class ClaimStatus(str, Enum):
    pending = "pending"
    auto_approved = "auto_approved"
    in_clinical_review = "in_clinical_review"
    in_md_review = "in_md_review"
    awaiting_info = "awaiting_info"
    denied = "denied"
    flagged_fraud = "flagged_fraud"


class SLATier(str, Enum):
    routine = "routine"      # 24 hours
    critical = "critical"    # 12 hours
    urgent = "urgent"        # 3 hours


SLA_HOURS = {
    SLATier.routine: 24,
    SLATier.critical: 12,
    SLATier.urgent: 3,
}


class DocumentationType(str, Enum):
    physician_notes = "physician_notes"
    lab_results = "lab_results"
    imaging_report = "imaging_report"
    surgical_notes = "surgical_notes"
    referral_letter = "referral_letter"
    prior_auth_history = "prior_auth_history"
    none = "none"


# ── Status mapping ───────────────────────────────────────────────────────────

DECISION_TO_STATUS = {
    RoutingDecision.auto_approve: ClaimStatus.auto_approved,
    RoutingDecision.clinical_review: ClaimStatus.in_clinical_review,
    RoutingDecision.md_review: ClaimStatus.in_md_review,
    RoutingDecision.request_info: ClaimStatus.awaiting_info,
    RoutingDecision.deny: ClaimStatus.denied,
    RoutingDecision.flag_fraud: ClaimStatus.flagged_fraud,
}


# ── Claim (internal — has hidden ground truth) ───────────────────────────────

class Claim(BaseModel):
    claim_id: str
    hospital_id: str
    procedure_code: str
    procedure_description: str
    diagnosis_code: str
    diagnosis_description: str
    billed_amount: float
    documentation: List[str] = Field(default_factory=list)
    sla_tier: SLATier = SLATier.routine
    sla_deadline_hr: int = 24
    arrival_hr: int = 0
    patient_age: int = 45
    days_in_queue: int = 0
    status: ClaimStatus = ClaimStatus.pending

    # Hidden fields — never exposed to agent
    is_fraud: bool = False
    ground_truth_routing: RoutingDecision = RoutingDecision.clinical_review


# ── ClaimSnapshot (agent-visible — no ground truth) ──────────────────────────

class ClaimSnapshot(BaseModel):
    claim_id: str
    hospital_id: str
    procedure_code: str
    procedure_description: str
    diagnosis_code: str
    diagnosis_description: str
    billed_amount: float
    documentation: List[str] = Field(default_factory=list)
    sla_tier: SLATier = SLATier.routine
    sla_deadline_hr: int = 24
    sla_remaining_hr: float = 24.0
    patient_age: int = 45
    days_in_queue: int = 0
    status: ClaimStatus = ClaimStatus.pending


def claim_to_snapshot(claim: Claim, current_hr: int = 0) -> ClaimSnapshot:
    """Convert a Claim to an agent-visible ClaimSnapshot (strips hidden fields)."""
    elapsed = max(0, current_hr - claim.arrival_hr)
    remaining = max(0.0, claim.sla_deadline_hr - elapsed)
    return ClaimSnapshot(
        claim_id=claim.claim_id,
        hospital_id=claim.hospital_id,
        procedure_code=claim.procedure_code,
        procedure_description=claim.procedure_description,
        diagnosis_code=claim.diagnosis_code,
        diagnosis_description=claim.diagnosis_description,
        billed_amount=claim.billed_amount,
        documentation=claim.documentation,
        sla_tier=claim.sla_tier,
        sla_deadline_hr=claim.sla_deadline_hr,
        sla_remaining_hr=remaining,
        patient_age=claim.patient_age,
        days_in_queue=claim.days_in_queue,
        status=claim.status,
    )


# ── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    current_hour: int = 0
    current_day: int = 0
    queue: List[ClaimSnapshot] = Field(default_factory=list)
    processed_today: int = 0
    md_slots_remaining: int = 0
    clinical_slots_remaining: int = 0
    policy_update_active: bool = False
    total_claims_in_episode: int = 0
    task_id: int = 1
    step_number: int = 0


# ── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    claim_id: str
    decision: RoutingDecision
    rationale: Optional[str] = None


# ── Reward ───────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    correct_auto_approve: float = 0.0
    correct_denial: float = 0.0
    correct_clinical_route: float = 0.0
    correct_rfi: float = 0.0
    correct_fraud_flag: float = 0.0
    deadline_bonus: float = 0.0
    deadline_miss_penalty: float = 0.0
    false_denial_penalty: float = 0.0
    unnecessary_review_penalty: float = 0.0
    redundant_rfi_penalty: float = 0.0
    missed_fraud_penalty: float = 0.0
    wrong_routing_penalty: float = 0.0
    total: float = 0.0

    def __init__(self, **kwargs: object) -> None:
        for field_name in [
            "correct_auto_approve", "correct_denial",
            "correct_clinical_route", "correct_rfi", "correct_fraud_flag",
            "deadline_bonus", "deadline_miss_penalty",
            "false_denial_penalty",
            "unnecessary_review_penalty", "redundant_rfi_penalty",
            "missed_fraud_penalty", "wrong_routing_penalty", "total",
        ]:
            kwargs.setdefault(field_name, 0.0)
        super().__init__(**kwargs)
