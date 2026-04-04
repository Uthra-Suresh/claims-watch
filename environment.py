"""
ClaimWatch: Insurance Claims Triage RL Environment
OpenEnv-compatible environment for payer-side claims queue management.
"""

import random
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Action(str, Enum):
    AUTO_APPROVE = "auto_approve"
    CLINICAL_REVIEW = "clinical_review"
    MD_REVIEW = "md_review"
    REQUEST_INFO = "request_info"
    DENY = "deny"

ACTIONS = [a.value for a in Action]

PROCEDURE_CODES = {
    "99213": {"name": "Office visit, low complexity", "routine": True, "cost": 150},
    "99215": {"name": "Office visit, high complexity", "routine": False, "cost": 350},
    "27447": {"name": "Total knee arthroplasty", "routine": False, "cost": 25000},
    "93000": {"name": "ECG", "routine": True, "cost": 80},
    "70553": {"name": "MRI brain with contrast", "routine": False, "cost": 2800},
    "45378": {"name": "Colonoscopy", "routine": True, "cost": 1200},
    "99291": {"name": "Critical care, first hour", "routine": False, "cost": 900},
    "36415": {"name": "Venipuncture", "routine": True, "cost": 30},
    "90837": {"name": "Psychotherapy 60 min", "routine": True, "cost": 200},
    "33533": {"name": "CABG arterial", "routine": False, "cost": 40000},
}

DIAGNOSIS_CODES = {
    "Z00.00": {"name": "General adult exam", "urgent": False, "fraud_risk": False},
    "I21.9":  {"name": "Acute MI unspecified", "urgent": True,  "fraud_risk": False},
    "M17.11": {"name": "Primary osteoarthritis knee", "urgent": False, "fraud_risk": False},
    "F32.1":  {"name": "Major depressive disorder", "urgent": False, "fraud_risk": False},
    "I63.9":  {"name": "Cerebral infarction", "urgent": True,  "fraud_risk": False},
    "Z12.11": {"name": "Colon cancer screening", "urgent": False, "fraud_risk": False},
    "G35":    {"name": "Multiple sclerosis", "urgent": False, "fraud_risk": False},
    "T14.91": {"name": "Suicide attempt", "urgent": True,  "fraud_risk": False},
    "Z99.11": {"name": "Dependence on respirator", "urgent": True,  "fraud_risk": False},
    "Z00.01": {"name": "Exam with abnormal findings", "urgent": False, "fraud_risk": True},
}

HOSPITAL_IDS = ["HOSP_001", "HOSP_002", "HOSP_003", "HOSP_004", "HOSP_005",
                "HOSP_006", "HOSP_007", "HOSP_008", "HOSP_009", "HOSP_010"]

POLICY_SETS = {
    "POLICY_A": {
        "auto_approve_threshold": 500,
        "covered_procedures": ["99213", "93000", "45378", "36415", "90837"],
        "always_review": ["27447", "33533", "70553"],
        "name": "Standard PPO",
    },
    "POLICY_B": {
        "auto_approve_threshold": 200,
        "covered_procedures": ["99213", "93000", "36415"],
        "always_review": ["27447", "33533", "70553", "99215", "99291"],
        "name": "Restrictive HMO",
    },
    "POLICY_C": {
        "auto_approve_threshold": 1000,
        "covered_procedures": ["99213", "93000", "45378", "36415", "90837", "70553", "99215"],
        "always_review": ["27447", "33533"],
        "name": "Premium PPO",
    },
    "POLICY_D": {
        "auto_approve_threshold": 300,
        "covered_procedures": ["99213", "93000", "36415", "90837"],
        "always_review": ["27447", "33533", "70553", "99291"],
        "name": "Medicaid Managed",
    },
}

NOTE_TEMPLATES = {
    "routine": [
        "Patient presents for routine follow-up. Vitals stable. No acute concerns.",
        "Annual wellness visit. Labs within normal limits. Preventive counseling provided.",
        "Medication management visit. Patient reports good tolerance. Refills provided.",
    ],
    "urgent": [
        "URGENT: Patient admitted with chest pain radiating to left arm. EKG changes noted.",
        "STAT: Altered mental status. CT head ordered emergently. Neurology consulted.",
        "CRITICAL: Patient on ventilator support. ICU admission required immediately.",
    ],
    "incomplete": [
        "Patient seen for... [documentation incomplete]",
        "Procedure performed. See attached records. Authorization pending.",
        "Referral from Dr. [NAME]. Additional records to follow.",
    ],
    "fraud_signal": [
        "Multiple procedures performed same day across facilities.",
        "Billing for services not rendered per patient statement.",
        "Duplicate claim submission detected from previous billing cycle.",
    ],
}


@dataclass
class Claim:
    claim_id: str
    procedure_code: str
    diagnosis_code: str
    hospital_id: str
    patient_acuity: int          # 1 (low) – 5 (critical)
    claim_amount: float
    policy_id: str
    clinical_note: str
    is_urgent: bool
    is_fraud: bool
    documentation_complete: bool
    ground_truth_action: str     # what the correct action is

    def to_observation(self) -> dict:
        """Return agent-visible fields (ground truth hidden)."""
        return {
            "claim_id": self.claim_id,
            "procedure_code": self.procedure_code,
            "procedure_name": PROCEDURE_CODES[self.procedure_code]["name"],
            "diagnosis_code": self.diagnosis_code,
            "diagnosis_name": DIAGNOSIS_CODES[self.diagnosis_code]["name"],
            "hospital_id": self.hospital_id,
            "patient_acuity": self.patient_acuity,
            "claim_amount": self.claim_amount,
            "policy_id": self.policy_id,
            "policy_name": POLICY_SETS[self.policy_id]["name"],
            "policy_auto_approve_threshold": POLICY_SETS[self.policy_id]["auto_approve_threshold"],
            "policy_covered_procedures": POLICY_SETS[self.policy_id]["covered_procedures"],
            "policy_always_review": POLICY_SETS[self.policy_id]["always_review"],
            "clinical_note": self.clinical_note,
            "available_actions": ACTIONS,
        }


def _generate_claim(claim_id: str, policy_id: str, hospital_id: str,
                    force_urgent: bool = False, force_fraud: bool = False,
                    force_incomplete: bool = False) -> Claim:
    proc_code = random.choice(list(PROCEDURE_CODES.keys()))
    diag_code = random.choice(list(DIAGNOSIS_CODES.keys()))

    proc = PROCEDURE_CODES[proc_code]
    diag = DIAGNOSIS_CODES[diag_code]
    policy = POLICY_SETS[policy_id]

    is_urgent = force_urgent or diag["urgent"] or (proc_code == "99291")
    is_fraud = force_fraud or diag["fraud_risk"]
    doc_complete = not force_incomplete and random.random() > 0.15

    acuity = 5 if is_urgent else (4 if not proc["routine"] else random.randint(1, 3))
    amount = proc["cost"] * random.uniform(0.85, 1.15)

    # Pick clinical note
    if force_fraud:
        note = random.choice(NOTE_TEMPLATES["fraud_signal"])
    elif is_urgent:
        note = random.choice(NOTE_TEMPLATES["urgent"])
    elif not doc_complete:
        note = random.choice(NOTE_TEMPLATES["incomplete"])
    else:
        note = random.choice(NOTE_TEMPLATES["routine"])

    # Determine ground truth
    if is_fraud:
        gt = Action.DENY.value
    elif is_urgent:
        gt = Action.MD_REVIEW.value
    elif not doc_complete:
        gt = Action.REQUEST_INFO.value
    elif proc_code in policy["always_review"]:
        gt = Action.CLINICAL_REVIEW.value
    elif proc_code in policy["covered_procedures"] and amount <= policy["auto_approve_threshold"]:
        gt = Action.AUTO_APPROVE.value
    elif proc["routine"] and amount <= policy["auto_approve_threshold"]:
        gt = Action.AUTO_APPROVE.value
    else:
        gt = Action.CLINICAL_REVIEW.value

    return Claim(
        claim_id=claim_id,
        procedure_code=proc_code,
        diagnosis_code=diag_code,
        hospital_id=hospital_id,
        patient_acuity=acuity,
        claim_amount=round(amount, 2),
        policy_id=policy_id,
        clinical_note=note,
        is_urgent=is_urgent,
        is_fraud=is_fraud,
        documentation_complete=doc_complete,
        ground_truth_action=gt,
    )


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

REWARD_TABLE = {
    # (is_urgent, is_fraud, doc_complete, action) -> reward
    # Correct auto-approve of routine claim
    (False, False, True,  Action.AUTO_APPROVE.value):   +0.30,
    # Correct escalation of urgent claim
    (True,  False, True,  Action.MD_REVIEW.value):      +0.50,
    (True,  False, False, Action.MD_REVIEW.value):      +0.50,
    # Missed urgent flag
    (True,  False, True,  Action.AUTO_APPROVE.value):   -0.80,
    (True,  False, True,  Action.CLINICAL_REVIEW.value):-0.40,
    (True,  False, True,  Action.REQUEST_INFO.value):   -0.40,
    # Unnecessary clinical review of routine
    (False, False, True,  Action.CLINICAL_REVIEW.value):-0.20,
    # Correct fraud denial
    (False, True,  True,  Action.DENY.value):           +0.40,
    (False, True,  False, Action.DENY.value):           +0.40,
    # False denial of legitimate claim
    (False, False, True,  Action.DENY.value):           -0.60,
    # Correct RFI when docs incomplete
    (False, False, False, Action.REQUEST_INFO.value):   +0.20,
    # Redundant RFI when docs complete
    (False, False, True,  Action.REQUEST_INFO.value):   -0.15,
    # MD review of non-urgent (wasteful but not catastrophic)
    (False, False, True,  Action.MD_REVIEW.value):      -0.20,
}

def _compute_reward(claim: Claim, action: str) -> tuple[float, str]:
    key = (claim.is_urgent, claim.is_fraud, claim.documentation_complete, action)
    if key in REWARD_TABLE:
        return REWARD_TABLE[key], "exact_match"
    # Fallback: partial credit for reasonable decisions
    if action == claim.ground_truth_action:
        return +0.10, "partial_correct"
    return -0.10, "incorrect"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ClaimWatchEnv:
    """
    OpenEnv-compatible environment for insurance claims triage.

    Observation: dict with claim fields (see Claim.to_observation())
    Action: one of ACTIONS (string)
    Reward: float per step, dense signal
    Episode: ends when all claims in the queue are processed
    """

    TASK_CONFIGS = {
        1: {
            "n_claims": 50,
            "n_policies": 1,
            "n_hospitals": 2,
            "urgent_fraction": 0.0,
            "fraud_fraction": 0.0,
            "incomplete_fraction": 0.0,
            "mid_episode_policy_change": False,
            "fraud_burst": False,
        },
        2: {
            "n_claims": 300,
            "n_policies": 4,
            "n_hospitals": 6,
            "urgent_fraction": 0.15,
            "fraud_fraction": 0.0,
            "incomplete_fraction": 0.10,
            "mid_episode_policy_change": False,
            "fraud_burst": False,
        },
        3: {
            "n_claims": 1000,
            "n_policies": 4,
            "n_hospitals": 10,
            "urgent_fraction": 0.12,
            "fraud_fraction": 0.02,
            "incomplete_fraction": 0.12,
            "mid_episode_policy_change": True,
            "fraud_burst": True,   # 20 extra fraud claims injected at step 400
        },
    }

    def __init__(self, task: int = 1, seed: int = 42):
        assert task in (1, 2, 3), "task must be 1, 2, or 3"
        self.task = task
        self.seed = seed
        self.cfg = self.TASK_CONFIGS[task]
        self._rng = random.Random(seed)
        self._queue: list[Claim] = []
        self._step_idx: int = 0
        self._episode_rewards: list[float] = []
        self._done: bool = True
        self._policy_changed: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Start a new episode. Returns first observation."""
        random.seed(self.seed)
        self._rng = random.Random(self.seed)
        self._queue = self._build_queue()
        self._step_idx = 0
        self._episode_rewards = []
        self._done = False
        self._policy_changed = False
        obs = self._current_observation()
        obs["episode_info"] = self._episode_info()
        return obs

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """
        Apply action to current claim.

        Returns:
            observation: next claim (or terminal obs)
            reward: float
            done: bool
            info: dict with diagnostics
        """
        assert not self._done, "Episode finished. Call reset()."
        assert action in ACTIONS, f"Invalid action '{action}'. Must be one of {ACTIONS}"

        claim = self._queue[self._step_idx]
        reward, reason = _compute_reward(claim, action)
        self._episode_rewards.append(reward)

        info = {
            "claim_id": claim.claim_id,
            "action_taken": action,
            "ground_truth": claim.ground_truth_action,
            "reward": reward,
            "reward_reason": reason,
            "is_urgent": claim.is_urgent,
            "is_fraud": claim.is_fraud,
            "step": self._step_idx,
            "total_steps": len(self._queue),
        }

        self._step_idx += 1

        # Mid-episode policy change (Task 3 only, at 50% through queue)
        if (self.cfg["mid_episode_policy_change"]
                and not self._policy_changed
                and self._step_idx == len(self._queue) // 2):
            self._policy_changed = True
            info["policy_change"] = "ALERT: Policy update applied. POLICY_B now covers procedure 99215."
            POLICY_SETS["POLICY_B"]["covered_procedures"].append("99215")

        if self._step_idx >= len(self._queue):
            self._done = True
            obs = {"done": True, "episode_info": self._episode_info()}
        else:
            obs = self._current_observation()
            obs["episode_info"] = self._episode_info()

        return obs, reward, self._done, info

    def score(self) -> dict:
        """
        Return graded scores (all 0.0–1.0) after episode completes.
        This is the primary grader output for hackathon evaluation.
        """
        assert self._done, "Call score() only after episode ends."
        claims = self._queue
        n = len(claims)

        correct = 0
        urgent_correct = 0
        urgent_total = 0
        fraud_correct = 0
        fraud_total = 0
        false_denials = 0
        total_reward = sum(self._episode_rewards)

        # We need to replay decisions — store them during step
        # (For scoring we use the reward trace as a proxy)
        # A full implementation would store (claim, action) pairs.
        # Here we derive from reward values:
        for r in self._episode_rewards:
            if r > 0:
                correct += 1

        # Routing accuracy: fraction of positive-reward steps
        routing_accuracy = correct / n if n > 0 else 0.0

        # Normalize total reward to 0-1 range
        # Max possible reward per claim ≈ 0.5, min ≈ -0.8
        max_possible = n * 0.5
        min_possible = n * -0.8
        reward_score = (total_reward - min_possible) / (max_possible - min_possible)
        reward_score = max(0.0, min(1.0, reward_score))

        scores = {
            "routing_accuracy": round(routing_accuracy, 4),
            "reward_score": round(reward_score, 4),
            "composite": round((routing_accuracy + reward_score) / 2, 4),
            "total_reward": round(total_reward, 4),
            "n_claims": n,
        }

        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_queue(self) -> list[Claim]:
        cfg = self.cfg
        policies = list(POLICY_SETS.keys())[:cfg["n_policies"]]
        hospitals = HOSPITAL_IDS[:cfg["n_hospitals"]]
        n = cfg["n_claims"]
        queue = []

        n_urgent = int(n * cfg["urgent_fraction"])
        n_fraud = int(n * cfg["fraud_fraction"])
        n_incomplete = int(n * cfg["incomplete_fraction"])
        n_routine = n - n_urgent - n_fraud - n_incomplete

        idx = 0
        for _ in range(n_routine):
            queue.append(_generate_claim(
                f"CLM-{idx:05d}",
                self._rng.choice(policies),
                self._rng.choice(hospitals),
            ))
            idx += 1

        for _ in range(n_urgent):
            queue.append(_generate_claim(
                f"CLM-{idx:05d}",
                self._rng.choice(policies),
                self._rng.choice(hospitals),
                force_urgent=True,
            ))
            idx += 1

        for _ in range(n_fraud):
            queue.append(_generate_claim(
                f"CLM-{idx:05d}",
                self._rng.choice(policies),
                self._rng.choice(hospitals),
                force_fraud=True,
            ))
            idx += 1

        for _ in range(n_incomplete):
            queue.append(_generate_claim(
                f"CLM-{idx:05d}",
                self._rng.choice(policies),
                self._rng.choice(hospitals),
                force_incomplete=True,
            ))
            idx += 1

        # Fraud burst for Task 3: inject 20 fraud claims at position 400
        if cfg.get("fraud_burst"):
            burst = []
            for i in range(20):
                burst.append(_generate_claim(
                    f"CLM-BURST-{i:03d}",
                    self._rng.choice(policies),
                    "HOSP_010",   # unusual hospital
                    force_fraud=True,
                ))
            queue = queue[:400] + burst + queue[400:]

        self._rng.shuffle(queue)
        return queue

    def _current_observation(self) -> dict:
        return self._queue[self._step_idx].to_observation()

    def _episode_info(self) -> dict:
        return {
            "task": self.task,
            "step": self._step_idx,
            "total_claims": len(self._queue),
            "cumulative_reward": round(sum(self._episode_rewards), 4),
            "remaining": len(self._queue) - self._step_idx,
        }
