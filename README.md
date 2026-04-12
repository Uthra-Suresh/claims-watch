---
title: ClaimWatch
emoji: "🏥"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
pinned: false
license: mit
short_description: RL environment for insurance claims triage
---

# ClaimWatch

ClaimWatch is an OpenEnv-compatible reinforcement learning environment for insurance claims triage. It models the operational problem faced by payers and utilization-management teams: every claim is a routing decision under time pressure, reviewer capacity constraints, documentation uncertainty, fraud risk, and changing policy rules.

## Quick Start

### Using the Client SDK (Recommended)

Async:

```python
from client import ClaimWatchClient
from models import ClaimAction

async with ClaimWatchClient(base_url="http://localhost:8000") as env:
    result = await env.reset(task_id=1, seed=42)
    obs = result.observation
    print(f"Queue: {len(obs.queue)} claims")

    result = await env.step(ClaimAction(
        claim_id=obs.queue[0].claim_id,
        decision="auto_approve",
    ))
    print(f"Reward: {result.reward}, Done: {result.done}")
```

Sync:

```python
with ClaimWatchClient(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id=1, seed=42)
    result = env.step(ClaimAction(
        claim_id=result.observation.queue[0].claim_id,
        decision="clinical_review",
    ))
    print(result.observation.queue)
```

### Using HTTP Directly

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 42}'

curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"claim_id": "CLM-42-00001", "decision": "clinical_review"}'

curl http://localhost:8000/state
```

## Industry Problem

Healthcare claims operations are not just a classification problem.

Teams have to decide, claim by claim:

- whether a routine case can be auto-approved
- whether a nurse reviewer or physician reviewer should spend limited capacity on it
- whether documentation is missing and more information is required
- whether a claim should be denied under policy rules
- whether the pattern looks fraudulent enough to escalate immediately

The hard part is that these decisions are coupled. A bad decision does not only hurt one claim. It also burns scarce review slots, increases queue time for urgent cases, and raises the chance of SLA misses later in the episode. Static rules and one-shot classifiers usually miss that operational tradeoff.

## Why RL Fits This Problem

This is a strong reinforcement learning use case because the agent is optimizing a sequence of decisions, not a single label.

RL is the right tool here because the environment has:

- sequential decision-making across an episode
- delayed consequences through SLA misses and capacity depletion
- constrained resources such as physician and clinical-review slots
- non-stationarity through policy changes in harder tasks
- competing objectives: accuracy, speed, fraud detection, and avoidance of false denials

In other words, the agent needs a policy, not just a prediction. ClaimWatch is designed to benchmark exactly that.

## Current Architecture

Main modules:

- [client.py](client.py): Remote async OpenEnv client (`ClaimWatchClient`)
- [env/claim_env.py](env/claim_env.py): Core RL environment (`ClaimWatchEnv`)
- [env/models.py](env/models.py): Pydantic v2 models for actions, observations, state
- [env/generator.py](env/generator.py): Seeded claim batch generator
- [env/reward.py](env/reward.py): Decomposed per-step reward computation
- [env/tasks.py](env/tasks.py): Task configurations and grader functions
- [env/policies.py](env/policies.py): Payer policies and procedure rules
- [rubrics.py](rubrics.py): Rubric-based rewards (OpenEnv RFC 004)
- [server/app.py](server/app.py): OpenEnv HTTP server app and env factory
- [inference.py](inference.py): Baseline LLM agent

## Environment Design

### Action Space

The agent chooses one routing decision per step:

| Decision | Meaning |
|----------|---------|
| `auto_approve` | Approve a routine claim with sufficient evidence |
| `clinical_review` | Route to clinical reviewer |
| `md_review` | Route to physician reviewer |
| `request_info` | Ask for missing documentation |
| `deny` | Deny a non-covered procedure |
| `flag_fraud` | Escalate suspected fraud |

### Observation Space

The environment returns a structured observation containing queue state and operating context.

Important implementation details from the current code:

- the visible queue is capped at `50` pending claims
- claims are sorted by shortest SLA remaining time, then highest billed amount
- observations include `processed_today`, `md_slots_remaining`, `clinical_slots_remaining`, `policy_update_active`, `task_id`, and `step_number`
- hidden labels such as fraud status and ground-truth routing are never exposed to the agent

Example observation payload:

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
      "diagnosis_description": "Acute upper respiratory infection",
      "billed_amount": 250.0,
      "documentation": ["physician_notes"],
      "sla_tier": "routine",
      "sla_deadline_hr": 24,
      "sla_remaining_hr": 22.5,
      "hospital_id": "HOSP-001",
      "patient_age": 45,
      "days_in_queue": 0,
      "status": "pending"
    }
  ],
  "processed_today": 1,
  "md_slots_remaining": 20,
  "clinical_slots_remaining": 40,
  "policy_update_active": false,
  "total_claims_in_episode": 20,
  "task_id": 1,
  "step_number": 1
}
```

### Reward Structure

Per-step rewards are clamped to `[-1.0, 1.0]`.

| Outcome | Reward |
|---------|--------|
| Correct fraud flag | `+0.45` |
| Correct denial | `+0.40` |
| Correct auto-approve | `+0.30` |
| Correct clinical or MD review route | `+0.25` |
| Correct request-info | `+0.20` |
| SLA met bonus | `+0.10` |
| Missed fraud | `-0.35` |
| False denial | `-0.60` |
| Deadline miss penalty | `-0.15` |
| Unnecessary review of routine claim | `-0.20` |
| Redundant request-info | `-0.15` |
| Other wrong routing | `-0.10` |

### Tasks

| ID | Name | Difficulty | Claims | Key dynamics |
|----|------|------------|--------|--------------|
| 1 | `routine_triage` | Easy | `20` | Stable policy, ample review capacity |
| 2 | `multi_hospital_triage` | Medium | `30` | Policy update at step `25`, tighter capacity |
| 3 | `full_complexity` | Hard | `50` | Policy update at step `50`, low capacity, stronger urgency pressure |

### Grader Formulas

Task 1:

```text
score = clamp(0.70 * routing_accuracy + 0.30 * sla_compliance_rate)
```

Task 2:

```text
score = clamp(
    0.40 * routing_accuracy
  + 0.35 * sla_compliance_rate
  + 0.15 * fraud_detection_rate
  + 0.10 * (1 - false_denial_rate)
  - 0.05 * unnecessary_review_rate
)
```

Task 3:

```text
raw = (
    0.40 * routing_accuracy
  + 0.30 * sla_compliance_rate
  + 0.20 * fraud_detection_rate
  + 0.10 * (1 - false_denial_rate)
  - 0.05 * unnecessary_review_rate
)

urgency_multiplier = 1.0 if sla_compliance_rate >= 0.70 else sla_compliance_rate / 0.70
score = clamp(raw * urgency_multiplier)
```

## Rewards (Rubric System — RFC 004)

ClaimWatch supports the OpenEnv Rubric system for composable, swappable rewards suitable for RL training (GRPO, PPO, etc.):

```python
from rubrics import ClaimWatchRubric, RoutingAccuracyRubric

# Default composite rubric
rubric = ClaimWatchRubric()
reward = rubric.forward(action, observation)

# Custom rubric with partial credit
rubric = ClaimWatchRubric(
    outcome=RoutingAccuracyRubric(partial_credit=True),
    fraud_weight=0.3,
)
```

Available rubric classes:
- `RoutingAccuracyRubric` — Outcome rubric: 1.0 for correct routing, optional 0.5 for adjacent decisions
- `SLAComplianceRubric` — Process rubric: +0.10 for on-time processing, -0.15 for SLA misses
- `FraudDetectionRubric` — Outcome rubric: +0.45 for correct fraud flags, -0.35 for missed fraud
- `ClaimWatchRubric` — Composite rubric combining all the above with configurable weights
- `CustomMetricRubric` — User-provided `metric(expected, predicted) -> float`

## Run Locally

### Python

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t claimwatch .
docker run -p 8000:8000 claimwatch
```

## Run The Baseline Agent

The repository includes a baseline agent in `inference.py` that calls an OpenAI-compatible model endpoint and interacts with the environment over WebSocket.

Example environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:8000"
```

Example runs:

```bash
python inference.py
python inference.py --easy
python inference.py --medium
python inference.py --hard
python inference.py --n-claims 5
python inference.py --debug
```

Current CLI defaults:

- `--easy` -> task 1 with `20` claims
- `--medium` -> task 2 with `30` claims
- `--hard` -> task 3 with `50` claims
- no task flags -> all tasks, `100` claims total

Expected log shape:

```text
[START] task=routine_triage env=claimwatch model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=auto_approve(claim=CLM-42-000) reward=0.30 done=false error=null
[STEP] step=2 action=flag_fraud(claim=CLM-42-001) reward=0.45 done=false error=null
[END] success=true steps=20 score=0.720 rewards=0.30,0.45,...
```

Note: the baseline script currently reports `score` as average per-step reward clamped into `[0, 1]`. That is separate from the task grader score produced by the environment at episode end.

## Baseline Scores

Reproducible grader scores with `seed=42` on the default claim counts per task.

| Task | Difficulty | Claims | Oracle | Random | Qwen2.5-72B Agent |
|------|------------|--------|--------|--------|--------------------|
| 1 — `routine_triage` | Easy | 20 | 1.000 | 0.312 | 0.447 |
| 2 — `multi_hospital_triage` | Medium | 30 | 1.000 | 0.374 | 0.519 |
| 3 — `full_complexity` | Hard | 50 | 1.000 | 0.057 | 0.113 |

**Oracle** uses the ground-truth routing decision for every claim — the theoretical maximum.
**Random** picks uniformly from the 6 possible decisions (`seed=12345`).
**Qwen2.5-72B** is the LLM baseline agent from `inference.py` using `Qwen/Qwen2.5-72B-Instruct` via Hugging Face inference.

The scores confirm that the graders meaningfully separate agent quality: oracle achieves near-perfect scores, random performs poorly (especially on Task 3 where the urgency multiplier penalizes low SLA compliance), and the LLM agent lands in between with room for improvement through RL fine-tuning.

To reproduce:

```bash
# Oracle + Random (no server needed)
python baseline_scores.py

# LLM agent (requires running server + HF API key)
uvicorn server.app:app --host 0.0.0.0 --port 8000   # terminal 1
python inference.py --easy                             # terminal 2
python inference.py --medium
python inference.py --hard
```

## Repository Layout

```text
claimwatch/
|- client.py             ← OpenEnv client (ClaimWatchClient)
|- rubrics.py            ← Rubric system (RFC 004)
|- env/
|  |- models.py          ← Pydantic v2 models
|  |- policies.py        ← Payer policies and procedure rules
|  |- generator.py       ← Seeded claim batch generator
|  |- reward.py          ← Per-step reward computation
|  |- tasks.py           ← Task configs and grader functions
|  `- claim_env.py       ← Core RL environment
|- server/
|  `- app.py             ← OpenEnv HTTP server
|- inference.py           ← Baseline LLM agent
|- openenv.yaml           ← OpenEnv spec config
|- Dockerfile
`- requirements.txt
```
