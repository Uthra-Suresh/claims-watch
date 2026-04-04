from pydantic import BaseModel
from typing import Optional


class ClaimAction(BaseModel):
    action: str  # one of: auto_approve, clinical_review, md_review, request_info, deny


class ClaimObservation(BaseModel):
    claim_id: str = ""
    procedure_code: str = ""
    procedure_name: str = ""
    diagnosis_code: str = ""
    diagnosis_name: str = ""
    hospital_id: str = ""
    patient_acuity: int = 1
    claim_amount: float = 0.0
    policy_id: str = ""
    policy_name: str = ""
    policy_auto_approve_threshold: float = 500.0
    policy_covered_procedures: list[str] = []
    policy_always_review: list[str] = []
    clinical_note: str = ""
    available_actions: list[str] = []
    reward: Optional[float] = None
    done: bool = False
    step: int = 0
    total_claims: int = 0
    cumulative_reward: float = 0.0


class ClaimState(BaseModel):
    task: int = 1
    step: int = 0
    done: bool = False
    cumulative_reward: float = 0.0
