# ClaimWatch

## The Problem

Insurance claims processing is one of the most resource-intensive operations in healthcare. Human reviewers manually triage thousands of claims daily — deciding which ones to approve, which need clinical review, which are missing documentation, and which might be fraudulent. This process is:

- **Slow** — Claims sit in queues for hours or days, often missing SLA deadlines for urgent cases like cardiac arrests or strokes.
- **Error-prone** — Reviewers under pressure make costly mistakes: denying legitimate claims (leading to patient harm and legal liability) or approving fraudulent ones (causing financial losses).
- **Unscalable** — Specialist reviewers (physicians, clinical nurses) are limited resources. As claim volume grows, bottlenecks get worse.
- **Inconsistent** — Routing decisions vary by reviewer, time of day, and workload, with no standardized decision framework.

## What ClaimWatch Solves

ClaimWatch is a **production-grade Reinforcement Learning environment** that trains and evaluates AI agents to automate claims triage. It simulates realistic insurance operations — complete with SLA deadlines, limited reviewer capacity, mid-episode policy changes, and embedded fraud — so that agents can learn optimal routing strategies before deployment.

The system generates batches of medical claims with real CPT/ICD-10 codes, varying documentation completeness, and realistic billing patterns. An agent examines each claim and routes it through one of six decision pathways. The environment provides granular reward signals for every correct and incorrect action, enabling agents to learn the tradeoffs between speed, accuracy, cost, and risk.

**Key capabilities:**
- 6 routing decisions with procedure-aware ground truth
- SLA-driven prioritization (routine / critical / urgent deadlines)
- Pattern-based fraud detection (no explicit labels — the agent must learn from billing anomalies)
- Resource-constrained review slots that force strategic allocation
- Mid-episode policy updates that test adaptability
- Decomposed reward signals for interpretable learning

---

## Table of Contents

- [How It Works](#how-it-works)
- [Observation Space](#observation-space)
- [Action Space](#action-space)
- [Routing Decisions Explained](#routing-decisions-explained)
- [SLA Tiers & Deadlines](#sla-tiers--deadlines)
- [Reward & Penalty Mechanism](#reward--penalty-mechanism)
- [Grading & Scoring](#grading--scoring)
- [Resource Constraints](#resource-constraints)
- [Mid-Episode Policy Updates](#mid-episode-policy-updates)
- [Fraud Detection](#fraud-detection)
- [Tasks](#tasks)
- [API Endpoints](#api-endpoints)
- [Quick Start](#quick-start)
- [Docker](#docker)
- [Running the LLM Agent](#running-the-llm-agent)
- [CLI Options](#cli-options)
- [Testing](#testing)

---

## How It Works

1. **Reset** — A task is selected (easy / medium / hard). The environment generates a batch of insurance claims, each with a procedure code, diagnosis, billed amount, documentation, and an SLA deadline.
2. **Step loop** — Every step, the agent receives the top 50 pending claims (sorted by urgency). It picks **one** claim and assigns a routing decision. The environment returns a reward, an updated observation, and whether the episode is done.
3. **Episode ends** when every claim has been processed. At that point, a grader computes a final score.

There is no step limit and no budget — the episode runs until every claim is handled.

---

## Observation Space

Each step, the environment returns this structured observation:

| Field                       | Type            | Description                                         |
|-----------------------------|-----------------|-----------------------------------------------------|
| `current_hour`              | int             | Current hour of the working day (8–17)              |
| `current_day`               | int             | Day counter since episode start                     |
| `queue`                     | list of claims  | Up to 50 pending claims, sorted by SLA urgency      |
| `processed_today`           | int             | Claims processed so far today                       |
| `md_slots_remaining`        | int             | Physician review slots left today                   |
| `clinical_slots_remaining`  | int             | Clinical nurse review slots left today              |
| `policy_update_active`      | bool            | Whether policy v2 is now in effect                  |
| `total_claims_in_episode`   | int             | Total claims in this episode                        |
| `task_id`                   | int             | Current task ID (1, 2, or 3)                        |
| `step_number`               | int             | How many steps taken so far                         |

### Each claim in the queue contains:

| Field                   | Type      | Description                                                  |
|-------------------------|-----------|--------------------------------------------------------------|
| `claim_id`              | string    | Unique identifier (e.g. `CLM-42-00017`)                      |
| `hospital_id`           | string    | Originating hospital                                         |
| `procedure_code`        | string    | CPT/HCPCS code (e.g. `99213`, `33533`)                      |
| `procedure_description` | string    | Human-readable procedure name                                |
| `diagnosis_code`        | string    | ICD-10 code (e.g. `I21.3` for STEMI)                        |
| `diagnosis_description` | string    | Human-readable diagnosis                                     |
| `billed_amount`         | float     | Dollar amount billed                                         |
| `documentation`         | list[str] | Attached documents (e.g. `physician_notes`, `lab_results`)   |
| `sla_tier`              | string    | `routine`, `critical`, or `urgent`                           |
| `sla_deadline_hr`       | int       | Total SLA window in hours                                    |
| `sla_remaining_hr`      | float     | Hours left before the SLA deadline expires                   |
| `patient_age`           | int       | Patient's age                                                |
| `days_in_queue`         | int       | How many days this claim has been waiting                    |
| `status`                | string    | Current status (always `pending` in the queue)               |

> **Note:** The agent never sees the ground-truth routing or fraud labels — it must infer the correct decision from the claim data alone.

---

## Action Space

Each step, the agent submits one action:

```json
{
  "claim_id": "CLM-42-00017",
  "decision": "clinical_review",
  "rationale": "Moderate-complexity procedure with complete docs, needs nurse review"
}
```

| Field       | Required | Description                                          |
|-------------|----------|------------------------------------------------------|
| `claim_id`  | Yes      | Must match a claim ID currently in the queue         |
| `decision`  | Yes      | One of the 6 routing decisions (see below)           |
| `rationale` | No       | Optional explanation (useful for debugging)          |

Submitting an invalid `claim_id` (not in queue or already processed) results in a **-0.05** penalty and the step is wasted.

---

## Routing Decisions Explained

The agent must choose one of these 6 decisions for every claim:

| Decision           | When to Use                                                        | Resource Cost      |
|--------------------|--------------------------------------------------------------------|--------------------|
| `auto_approve`     | Routine, low-cost claims with all required docs present            | None               |
| `clinical_review`  | Moderate-complexity claims needing a clinical nurse to review      | 1 clinical slot    |
| `md_review`        | High-cost or complex claims needing a physician to review          | 1 MD slot          |
| `request_info`     | Required documentation is missing from the claim                   | None               |
| `deny`             | Procedure is not covered under the insurance plan                  | None               |
| `flag_fraud`       | Billing patterns look abnormal (e.g. billed amount far exceeds normal) | None           |

### Decision Priority (how ground truth is computed)

The environment determines the "correct" answer using this priority cascade:

1. **Fraud** — If the claim is fraudulent → `flag_fraud`
2. **Not covered** — Procedure not in the plan → `deny`
3. **Missing docs** — Required documentation absent → `request_info`
4. **Excessive cost** — Billed amount > 1.5× the procedure's normal max → `md_review`
5. **Simple + complete** — Docs complete and procedure is auto-approvable → `auto_approve`
6. **Default** — Fall back to the procedure's default routing (e.g. `clinical_review` or `md_review`)

---

## SLA Tiers & Deadlines

Every claim has a Service Level Agreement (SLA) tier that determines how quickly it must be processed:

| SLA Tier     | Deadline  | Typical Triggers                                       |
|--------------|-----------|--------------------------------------------------------|
| `routine`    | 24 hours  | Standard office visits, minor procedures               |
| `critical`   | 12 hours  | Brain injury, ischemic stroke, major surgery           |
| `urgent`     | 3 hours   | Cardiac arrest, STEMI, acute respiratory failure, CABG |

**How SLA tiers are assigned:**
- Each procedure has a default SLA tier (e.g. knee replacement = `critical`, CABG = `urgent`).
- Certain diagnosis codes (e.g. `I21.3` STEMI, `I46.9` cardiac arrest) can **upgrade** the tier to something more urgent.
- The final tier is whichever is more urgent between the procedure and diagnosis.

**What happens at the deadline?**
- Claims processed **before** the deadline get a **+0.10 bonus**.
- Claims processed **after** the deadline get a **-0.15 penalty**.
- The queue is sorted so the most urgent claims (lowest `sla_remaining_hr`) appear first.

---

## Reward & Penalty Mechanism

Every step produces a decomposed reward clamped to **[-1.0, +1.0]**. Here's exactly how it works:

### Correct Decision Rewards

When the agent's decision matches the ground truth:

| Scenario                              | Reward  |
|---------------------------------------|---------|
| Correctly auto-approved               | **+0.30** |
| Correctly denied                      | **+0.40** |
| Correctly routed to clinical review   | **+0.25** |
| Correctly requested info              | **+0.20** |
| Correctly flagged fraud               | **+0.45** |
| Correctly routed to MD review         | **+0.25** |

Plus the SLA deadline bonus/penalty:
- **On time** (before SLA deadline): **+0.10**
- **Late** (after SLA deadline): **-0.15**

**Best-case single step:** +0.45 (fraud flag) + 0.10 (on time) = **+0.55**

### Incorrect Decision Penalties

When the agent's decision is wrong, penalties depend on the severity of the mistake:

| Mistake                                                      | Penalty   |
|--------------------------------------------------------------|-----------|
| **False denial** — Denying a legitimate, non-fraud claim     | **-0.60** |
| **Missed fraud** — Approving/reviewing a fraudulent claim    | **-0.35** |
| **Unnecessary review** — Sending an auto-approvable claim to review | **-0.20** |
| **Redundant RFI** — Requesting info when docs are complete   | **-0.15** |
| **Wrong routing** — e.g. MD review when clinical was needed  | **-0.10** |

> **Key takeaway:** False denials are the most expensive mistake (-0.60). Catching fraud is the most valuable correct action (+0.45). Agents should prioritize avoiding false denials above all else.

### Reward Breakdown Example

```
Claim: CLM-42-00017 (CABG, urgent SLA, $82,000 billed)
Agent decision: md_review
Ground truth: md_review ✓

Reward breakdown:
  correct_clinical_route = +0.25
  deadline_bonus         = +0.10  (processed within 3hr SLA)
  total                  = +0.35
```

---

## Grading & Scoring

When the episode ends (all claims processed), a **grader** computes a final score from 0.0 to 1.0.

### Metrics Computed

| Metric                     | Description                                                   |
|----------------------------|---------------------------------------------------------------|
| `routing_accuracy`         | Fraction of claims routed to the correct decision             |
| `sla_compliance_rate`      | Fraction of urgent/critical claims routed correctly           |
| `fraud_detection_rate`     | Fraction of fraud claims correctly flagged                    |
| `false_denial_rate`        | Fraction of claims incorrectly denied                         |
| `unnecessary_review_rate`  | Fraction of auto-approvable claims sent to review             |

### Scoring Formulas

**Task 1 — Easy (routine_triage):**
```
score = 0.70 × routing_accuracy + 0.30 × sla_compliance
```

**Task 2 — Medium (multi_hospital_triage):**
```
score = 0.40 × routing_accuracy
      + 0.35 × sla_compliance
      + 0.15 × fraud_detection
      + 0.10 × (1 - false_denial_rate)
      - 0.05 × unnecessary_review_rate
```

**Task 3 — Hard (full_complexity):**
```
score = (0.40 × routing_accuracy
       + 0.30 × sla_compliance
       + 0.20 × fraud_detection
       + 0.10 × (1 - false_denial_rate)
       - 0.05 × unnecessary_review_rate)
       × multiplier

multiplier = 1.0 if sla_compliance ≥ 0.70, else sla_compliance / 0.70
```

> **Task 3 warning:** If SLA compliance drops below 70%, the multiplier collapses the entire score. Maintaining SLA compliance is critical.

---

## Resource Constraints

The agent has limited review capacity each day:

| Resource         | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) |
|------------------|---------------|-----------------|----------------|
| MD slots/day     | 20            | 8               | 4              |
| Clinical slots/day | 40          | 25              | 15             |

- **`md_review`** consumes 1 MD slot. **`clinical_review`** consumes 1 clinical slot.
- All other decisions (`auto_approve`, `deny`, `request_info`, `flag_fraud`) cost nothing.
- A new day starts every 50 steps — slots reset, and pending claims age by +1 day.

**What if slots run out?**
- If `md_review` is requested but MD slots are exhausted → auto-downgraded to `clinical_review`.
- If `clinical_review` is requested but clinical slots are exhausted → auto-downgraded to `request_info`.

This means the agent must be strategic about when to use expensive review slots, especially in harder tasks where slots are scarce.

---

## Mid-Episode Policy Updates

In Tasks 2 and 3, the insurance policy changes mid-episode:

| Task | Policy update fires at step | What changes                                      |
|------|----------------------------|---------------------------------------------------|
| 1    | Never                      | —                                                 |
| 2    | Step 150                   | Some procedures require additional documentation  |
| 3    | Step 500                   | Some procedures require additional documentation  |

When the policy updates from v1 → v2:
- `policy_update_active` flips to `true` in the observation.
- Remaining pending claims have their ground-truth re-evaluated under v2 rules.
- Example: Knee replacement (27447) now also requires `prior_auth_history` — claims missing that doc will need `request_info` instead of `clinical_review`.

The agent should detect `policy_update_active = true` and adapt its routing strategy accordingly.

---

## Fraud Detection

Fraudulent claims are generated at a configurable rate (6–10% depending on task). They are **not** labeled as fraud — the agent must detect them from patterns:

**Signals that suggest fraud:**
- **Abnormal billing:** Billed amount is 2–4× the procedure's normal maximum.
- Example: A routine office visit (`99213`, max $150) billed at $450.

**Why fraud detection matters:**
- Correctly flagging fraud: **+0.45** (highest single reward).
- Missing fraud (approving/reviewing a fraud claim): **-0.35** penalty.
- Fraud detection rate directly affects scores on Tasks 2 and 3.

---

## Tasks

| ID | Name                    | Difficulty | Claims | Hospitals | Fraud Rate | Policy Update |
|----|-------------------------|------------|--------|-----------|------------|---------------|
| 1  | `routine_triage`        | Easy       | 50     | 10        | 6%         | None          |
| 2  | `multi_hospital_triage` | Medium     | 300    | 50        | 8%         | Step 150      |
| 3  | `full_complexity`       | Hard       | 1000   | 200       | 10%        | Step 500      |

---

## API Endpoints

| Method | Path     | Description                                       |
|--------|----------|---------------------------------------------------|
| POST   | `/reset` | Start a new episode (accepts `task_id`, `seed`, `n_claims`) |
| GET    | `/reset` | Same as POST with query params                    |
| POST   | `/step`  | Submit one triage action                          |
| GET    | `/step`  | Same as POST with query params                    |
| GET    | `/state` | Full environment state (for debugging)            |
| GET    | `/tasks` | List available tasks and metadata                 |
| GET    | `/health`| Health check                                      |

### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 42}'
```

### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"claim_id": "CLM-42-00003", "decision": "auto_approve", "rationale": "Low-cost, docs complete"}'
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t claimwatch .
docker run -p 7860:7860 claimwatch
```

## Running the LLM Agent

```bash
# Set required environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token"
export ENV_BASE_URL="http://localhost:7860"

# Run inference
python inference.py
```

## CLI Options

```bash
python inference.py --easy          # Run Task 1 only (50 claims)
python inference.py --medium        # Run Task 2 only (300 claims)
python inference.py --hard          # Run Task 3 only (1000 claims)
python inference.py --n-claims 20   # Override claim count
python inference.py --debug         # Verbose per-step output
```

## Testing

```bash
python test_env.py
```

Runs 6 smoke tests covering: reset/step mechanics, SLA deadline tracking, fraud detection, resource slot exhaustion, policy updates, and grader scoring.
```