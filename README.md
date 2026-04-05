---
title: ClaimWatch
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
license: mit
short_description: Insurance claims triage RL environment — OpenEnv compliant
---
# ClaimWatch

> **OpenEnv-compliant RL environment** for insurance claims triage.  
---

## What is ClaimWatch?

Insurance claims processing is one of the most resource-intensive operations in healthcare. Human reviewers manually triage thousands of claims daily — deciding which ones to approve, which need clinical review, which are missing documentation, and which might be fraudulent.

**ClaimWatch** is a production-grade Reinforcement Learning environment that trains and evaluates AI agents to automate this triage. It simulates realistic insurance operations — complete with SLA deadlines, limited reviewer capacity, mid-episode policy changes, and embedded fraud — so agents can learn optimal routing strategies.

---

## Quick Start

```bash
# 1. Reset the environment (task 1, seed 42)
curl -X POST https://YOUR_USERNAME-claimwatch.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 42}'

# 2. Submit a routing decision
curl -X POST https://YOUR_USERNAME-claimwatch.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"claim_id": "CLM-42-00001", "decision": "clinical_review"}'

# 3. Check state
curl https://YOUR_USERNAME-claimwatch.hf.space/state
```

Interactive API docs: `https://YOUR_USERNAME-claimwatch.hf.space/docs`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode. Accepts `{}` (defaults to task 1, seed 42) |
| `POST` | `/step` | Submit one routing decision for a claim |
| `GET` | `/state` | Get full environment state |
| `GET` | `/tasks` | List all 3 tasks |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Action Space — 6 Routing Decisions

| Decision | When to use |
|----------|-------------|
| `auto_approve` | Routine claim, complete documentation, low cost |
| `clinical_review` | Moderate complexity, needs clinical nurse sign-off |
| `md_review` | High-cost or complex procedure, needs physician |
| `request_info` | Required documentation is missing |
| `deny` | Procedure is not covered under the policy |
| `flag_fraud` | Billing anomaly detected (billed >> expected) |

---

## Observation Space

Each step returns an `Observation` with:

```json
{
  "current_hour": 10,
  "current_day": 0,
  "queue": [
    {
      "claim_id": "CLM-42-00001",
      "procedure_code": "99213",
      "procedure_description": "Office visit",
      "diagnosis_code": "J06.9",
      "billed_amount": 250.00,
      "sla_tier": "routine",
      "sla_remaining_hr": 22.5,
      "documentation": ["physician_notes"],
      "hospital_id": "HOSP-001",
      "patient_age": 45,
      "days_in_queue": 0,
      "status": "pending"
    }
  ],
  "md_slots_remaining": 20,
  "clinical_slots_remaining": 40,
  "policy_update_active": false,
  "total_claims_in_episode": 50,
  "task_id": 1,
  "step_number": 1
}
```

---

## Reward Structure

| Outcome | Reward |
|---------|--------|
| Correct fraud flag | +0.45 |
| Correct denial | +0.40 |
| Correct auto-approve | +0.30 |
| Correct clinical route | +0.25 |
| Correct request-info | +0.20 |
| SLA met bonus | +0.10 |
| Missed fraud (approved fraud claim) | −0.35 |
| False denial (denied legit claim) | −0.60 |
| SLA deadline missed | −0.20 |
| Unnecessary review of routine claim | −0.20 |
| Wrong routing (general) | −0.10 |

All rewards are clamped to `[-1.0, 1.0]`.

---

## Tasks

| ID | Name | Difficulty | Claims | Notes |
|----|------|------------|--------|-------|
| 1 | `routine_triage` | Easy | 50 | No policy changes, ample slots |
| 2 | `multi_hospital_triage` | Medium | 300 | Policy v1→v2 flip at step 150 |
| 3 | `full_complexity` | Hard | 1000 | Tight slots, fraud bursts, policy update at step 500 |

### Grader Formulas

**Task 1 (Easy):**
```
score = 0.70 × routing_accuracy + 0.30 × sla_compliance
```

**Task 2 (Medium):**
```
score = 0.40 × routing_accuracy + 0.35 × sla_compliance
      + 0.15 × fraud_detection + 0.10 × (1 − false_denial_rate)
```

**Task 3 (Hard):**
```
raw   = 0.35 × routing_accuracy + 0.30 × sla_compliance
      + 0.20 × fraud_detection  + 0.15 × (1 − false_denial_rate)
score = raw × urgency_multiplier   # collapses if SLA recall < 0.70
```

---

## SLA Tiers

| Tier | Deadline | Example |
|------|----------|---------|
| `routine` | 24 hours | Office visits |
| `critical` | 12 hours | Surgical procedures |
| `urgent` | 3 hours | Cardiac arrest, stroke, ICU |

---

## Run Locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/claimwatch
cd claimwatch
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the baseline inference agent
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

---

## Docker

```bash
docker build -t claimwatch .
docker run -p 7860:7860 claimwatch

# Verify
curl -X POST -d '{}' -H "Content-Type: application/json" \
  http://localhost:7860/reset
```

---

## Run the Inference Agent

```bash
# All 3 tasks (default)
python inference.py

# Single task
python inference.py --easy
python inference.py --medium
python inference.py --hard

# Debug mode (verbose output)
python inference.py --debug

# Quick smoke test (5 claims per task)
python inference.py --n-claims 5
```

**Expected log format:**
```
[START] task=routine_triage env=claimwatch model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=auto_approve(claim=CLM-42-000) reward=0.30 done=false error=null
[STEP] step=2 action=flag_fraud(claim=CLM-42-001) reward=0.45 done=false error=null
[END] success=true steps=50 score=0.720 rewards=0.30,0.45,...
```

---

## File Structure

```
claimwatch/
├── env/
│   ├── models.py        ← Pydantic v2 models (Claim, Observation, Reward)
│   ├── policies.py      ← CPT codes, fraud signals, hospital tiers
│   ├── generator.py     ← Seeded claim generator
│   ├── reward.py        ← compute_reward() function
│   ├── tasks.py         ← 3 TaskConfigs + grader functions
│   └── claim_env.py     ← ClaimWatchEnv (reset/step/state)
├── server/
│   └── app.py           ← FastAPI application
├── inference.py         ← Baseline LLM agent
├── openenv.yaml         ← OpenEnv spec metadata
├── Dockerfile
└── requirements.txt
```

---

## OpenEnv Compliance

- ✅ `POST /reset` — accepts `{}` with defaults, returns HTTP 200
- ✅ `POST /step` — returns `(observation, reward, done, info)`
- ✅ `GET /state` — returns full environment state dict
- ✅ `openenv.yaml` — valid spec with 3 tasks
- ✅ Grader scores in `[0.0, 1.0]`
- ✅ Deterministic given same seed
- ✅ Runs on 2 vCPU / 8 GB RAM

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | Yes | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | Yes | — | HuggingFace / API key |
| `ENV_BASE_URL` | No | `http://localhost:7860` | Environment server URL |