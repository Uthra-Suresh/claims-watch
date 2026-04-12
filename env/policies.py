"""ClaimWatch — Payer policies, procedure rules, SLA tiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import RoutingDecision, SLATier


# ── Procedure Rule ───────────────────────────────────────────────────────────

@dataclass
class ProcedureRule:
    code: str
    description: str
    required_docs: List[str]
    auto_approve_if_docs_complete: bool
    default_routing: RoutingDecision
    covered: bool
    max_billed: float
    sla_tier: SLATier = SLATier.routine
    v2_changes: Optional[Dict] = None


PROCEDURE_RULES: Dict[str, ProcedureRule] = {
    "99213": ProcedureRule(
        code="99213",
        description="Office/outpatient visit, established patient, low complexity",
        required_docs=["physician_notes"],
        auto_approve_if_docs_complete=True,
        default_routing=RoutingDecision.auto_approve,
        covered=True,
        max_billed=150.0,
        sla_tier=SLATier.routine,
    ),
    "99214": ProcedureRule(
        code="99214",
        description="Office/outpatient visit, established patient, moderate complexity",
        required_docs=["physician_notes", "lab_results"],
        auto_approve_if_docs_complete=True,
        default_routing=RoutingDecision.auto_approve,
        covered=True,
        max_billed=250.0,
        sla_tier=SLATier.routine,
    ),
    "27447": ProcedureRule(
        code="27447",
        description="Total knee replacement",
        required_docs=["physician_notes", "imaging_report"],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.clinical_review,
        covered=True,
        max_billed=35000.0,
        sla_tier=SLATier.critical,
        v2_changes={"required_docs": ["physician_notes", "imaging_report", "prior_auth_history"]},
    ),
    "43239": ProcedureRule(
        code="43239",
        description="Upper GI endoscopy with biopsy",
        required_docs=["physician_notes", "lab_results"],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.clinical_review,
        covered=True,
        max_billed=5000.0,
        sla_tier=SLATier.routine,
    ),
    "33533": ProcedureRule(
        code="33533",
        description="Coronary artery bypass graft (CABG)",
        required_docs=["physician_notes", "imaging_report", "surgical_notes"],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.md_review,
        covered=True,
        max_billed=80000.0,
        sla_tier=SLATier.urgent,
    ),
    "61510": ProcedureRule(
        code="61510",
        description="Craniotomy for excision of brain tumor",
        required_docs=["physician_notes", "imaging_report", "surgical_notes"],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.md_review,
        covered=True,
        max_billed=95000.0,
        sla_tier=SLATier.urgent,
    ),
    "S9083": ProcedureRule(
        code="S9083",
        description="Global fee for emergency department visit — not covered",
        required_docs=[],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.deny,
        covered=False,
        max_billed=0.0,
        sla_tier=SLATier.routine,
    ),
    "99050": ProcedureRule(
        code="99050",
        description="After-hours service — not in standard plan",
        required_docs=[],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.deny,
        covered=False,
        max_billed=0.0,
        sla_tier=SLATier.routine,
    ),
    "99291": ProcedureRule(
        code="99291",
        description="Critical care, first 30-74 minutes",
        required_docs=["physician_notes", "lab_results"],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.md_review,
        covered=True,
        max_billed=600.0,
        sla_tier=SLATier.urgent,
    ),
    "99285": ProcedureRule(
        code="99285",
        description="Emergency department visit, high severity",
        required_docs=["physician_notes"],
        auto_approve_if_docs_complete=False,
        default_routing=RoutingDecision.md_review,
        covered=True,
        max_billed=900.0,
        sla_tier=SLATier.urgent,
    ),
}


# ── Diagnosis codes ──────────────────────────────────────────────────────────

# Diagnoses that inherently require urgent/critical handling
URGENT_DIAGNOSIS_CODES: Dict[str, SLATier] = {
    "I21.3": SLATier.urgent,        # STEMI
    "I46.9": SLATier.urgent,        # Cardiac arrest
    "J96.00": SLATier.urgent,       # Acute respiratory failure
    "G93.1": SLATier.urgent,        # Anoxic brain damage
    "R57.0": SLATier.urgent,        # Cardiogenic shock
    "S06.30XA": SLATier.critical,   # Brain injury
    "I63.50": SLATier.critical,     # Ischemic stroke
}

# Non-urgent diagnosis codes for routine claims
ROUTINE_DIAGNOSIS_CODES = [
    ("M17.11", "Primary osteoarthritis, right knee"),
    ("K21.0", "Gastroesophageal reflux with esophagitis"),
    ("E11.9", "Type 2 diabetes without complications"),
    ("J06.9", "Acute upper respiratory infection"),
    ("M54.5", "Low back pain"),
    ("I10", "Essential hypertension"),
    ("J44.1", "COPD with acute exacerbation"),
    ("K80.20", "Gallstone without obstruction"),
    ("N18.3", "Chronic kidney disease, stage 3"),
    ("G43.909", "Migraine, unspecified"),
]


# ── Resource Costs ───────────────────────────────────────────────────────────

@dataclass
class ResourceCost:
    clinical_slots: int
    md_slots: int


RESOURCE_COSTS: Dict[RoutingDecision, ResourceCost] = {
    RoutingDecision.auto_approve: ResourceCost(clinical_slots=0, md_slots=0),
    RoutingDecision.request_info: ResourceCost(clinical_slots=0, md_slots=0),
    RoutingDecision.clinical_review: ResourceCost(clinical_slots=1, md_slots=0),
    RoutingDecision.md_review: ResourceCost(clinical_slots=0, md_slots=1),
    RoutingDecision.deny: ResourceCost(clinical_slots=0, md_slots=0),
    RoutingDecision.flag_fraud: ResourceCost(clinical_slots=0, md_slots=0),
}


# All possible doc types for generation
ALL_DOC_TYPES = [
    "physician_notes",
    "lab_results",
    "imaging_report",
    "surgical_notes",
    "referral_letter",
    "prior_auth_history",
]
