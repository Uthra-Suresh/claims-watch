"""ClaimWatch — Seeded claim batch generator."""

from __future__ import annotations

import random as _random_module
from typing import List

from .models import Claim, RoutingDecision, SLATier, SLA_HOURS
from .policies import (
    ALL_DOC_TYPES,
    PROCEDURE_RULES,
    ROUTINE_DIAGNOSIS_CODES,
    URGENT_DIAGNOSIS_CODES,
)


def _compute_ground_truth(
    claim: Claim,
    policy_version: str = "v1",
) -> RoutingDecision:
    """Determine the correct routing for a claim using the priority cascade."""
    rule = PROCEDURE_RULES.get(claim.procedure_code)
    if rule is None:
        return RoutingDecision.clinical_review

    # 1. Fraud → FLAG_FRAUD  (model must detect from patterns, not flags)
    if claim.is_fraud:
        return RoutingDecision.flag_fraud

    # 2. Not covered → DENY
    if not rule.covered:
        return RoutingDecision.deny

    # 3. Required docs missing → REQUEST_INFO
    required = list(rule.required_docs)
    if policy_version == "v2" and rule.v2_changes and "required_docs" in rule.v2_changes:
        required = list(rule.v2_changes["required_docs"])
    for doc in required:
        if doc not in claim.documentation:
            return RoutingDecision.request_info

    # 4. Billed > 1.5 × max_billed → MD_REVIEW
    if rule.max_billed > 0 and claim.billed_amount > 1.5 * rule.max_billed:
        return RoutingDecision.md_review

    # 5. auto_approve_if_docs_complete → AUTO_APPROVE
    if rule.auto_approve_if_docs_complete:
        return RoutingDecision.auto_approve

    # 6. Else → default_routing from procedure rule
    return rule.default_routing


def generate_claims(
    n: int,
    seed: int = 42,
    fraud_rate: float = 0.08,
    policy_version: str = "v1",
) -> List[Claim]:
    """Generate a deterministic batch of claims."""
    rng = _random_module.Random(seed)

    procedure_codes = list(PROCEDURE_RULES.keys())
    urgent_diag_codes = list(URGENT_DIAGNOSIS_CODES.keys())

    claims: List[Claim] = []

    for i in range(n):
        # Hospital ID (simple, no tier)
        hospital_num = rng.randint(1, max(10, n * 2))
        hospital_id = f"HOSP-{hospital_num:04d}"

        # Determine fraud
        is_fraud = rng.random() < fraud_rate

        # Choose procedure
        proc_code = rng.choice(procedure_codes)
        rule = PROCEDURE_RULES[proc_code]

        # SLA tier: start with the procedure's default tier
        sla_tier = rule.sla_tier

        # Choose diagnosis (20% chance of urgent diagnosis)
        is_urgent_diag = rng.random() < 0.20
        if is_urgent_diag:
            diag_code = rng.choice(urgent_diag_codes)
            diag_desc = f"Acute condition ({diag_code})"
            # Diagnosis urgency can upgrade the SLA tier
            diag_tier = URGENT_DIAGNOSIS_CODES[diag_code]
            if SLA_HOURS[diag_tier] < SLA_HOURS[sla_tier]:
                sla_tier = diag_tier
        else:
            diag_code, diag_desc = rng.choice(ROUTINE_DIAGNOSIS_CODES)

        sla_deadline_hr = SLA_HOURS[sla_tier]

        # Arrival hour: spread across working hours
        arrival_hr = rng.randint(0, max(0, sla_deadline_hr - 1))

        # Documentation completeness: 75% complete, 25% missing one doc
        docs_complete = rng.random() < 0.75
        if rule.required_docs:
            if docs_complete:
                documentation = list(rule.required_docs)
            else:
                documentation = list(rule.required_docs)
                if documentation:
                    doc_to_remove = rng.choice(documentation)
                    documentation.remove(doc_to_remove)
        else:
            documentation = []

        # Add some extra docs occasionally
        if rng.random() < 0.3:
            extra_docs = [d for d in ALL_DOC_TYPES if d not in documentation]
            if extra_docs:
                documentation.append(rng.choice(extra_docs))

        # Billed amount
        if rule.max_billed > 0:
            if is_fraud:
                # Fraud claims often have abnormal billing
                billed = rule.max_billed * rng.uniform(2.0, 4.0)
            elif rng.random() < 0.10:
                billed = rule.max_billed * rng.uniform(1.6, 3.0)
            else:
                billed = rule.max_billed * rng.uniform(0.3, 1.2)
        else:
            billed = rng.uniform(50.0, 500.0)
        billed = round(billed, 2)

        patient_age = rng.randint(18, 95)
        days_in_queue = rng.randint(0, 5)

        claim_id = f"CLM-{seed}-{i:05d}"

        claim = Claim(
            claim_id=claim_id,
            hospital_id=hospital_id,
            procedure_code=proc_code,
            procedure_description=rule.description,
            diagnosis_code=diag_code,
            diagnosis_description=diag_desc,
            billed_amount=billed,
            documentation=documentation,
            sla_tier=sla_tier,
            sla_deadline_hr=sla_deadline_hr,
            arrival_hr=arrival_hr,
            patient_age=patient_age,
            days_in_queue=days_in_queue,
            is_fraud=is_fraud,
            ground_truth_routing=RoutingDecision.clinical_review,  # placeholder
        )

        # Compute ground truth routing
        claim.ground_truth_routing = _compute_ground_truth(claim, policy_version)

        claims.append(claim)

    return claims


def recompute_ground_truth(claims: List[Claim], policy_version: str = "v2") -> None:
    """Recompute ground_truth_routing for all claims in-place."""
    for claim in claims:
        claim.ground_truth_routing = _compute_ground_truth(claim, policy_version)
