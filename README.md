---
title: ClaimWatch
emoji: "🏥"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
license: mit
short_description: Insurance claims triage RL environment for routing, SLA, and fraud decisions
---

# ClaimWatch

ClaimWatch is an OpenEnv-compatible reinforcement learning environment for insurance claims triage. It models the operational problem faced by payers and utilization-management teams: every claim is a routing decision under time pressure, reviewer capacity constraints, documentation uncertainty, fraud risk, and changing policy rules.

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

## How ClaimWatch Solves It

ClaimWatch turns claims triage into a measurable control problem.

- State: the agent sees the live queue, SLA remaining time, billed amount, documentation, resource availability, policy state, and current episode progress.
- Actions: the agent chooses one of six routing decisions for the current claim.
- Rewards: the agent gets dense feedback for correct routing, fraud catches, SLA success, and penalties for false denials, missed fraud, unnecessary reviews, and other routing errors.
- Evaluation: task-specific graders score performance in `[0, 1]` using routing accuracy, SLA compliance, fraud detection, false denial rate, and unnecessary review rate.

The environment deliberately escalates from simple routing to harder operating conditions:

- Task 1: stable policy, high capacity, routine triage
- Task 2: more hospitals, tighter review capacity, mid-episode policy update
- Task 3: high complexity, low capacity, stronger fraud pressure, urgency-sensitive grading

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

## API And Interaction Model

ClaimWatch is served through OpenEnv's generated FastAPI application.

### HTTP Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit one action |
| `GET` | `/state` | Inspect current environment state |
| `GET` | `/schema` | Retrieve action, observation, and state schemas |
| `GET` | `/metadata` | Retrieve environment metadata |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

### Persistent Agent Sessions

For automated multi-step agents, prefer the WebSocket endpoint at `/ws`. The current baseline agent in `inference.py` uses that path so the episode state stays consistent across steps.

### Correct HTTP Payload Shapes

Reset accepts extra environment-specific fields such as `task_id` and `n_claims`.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 42}'
```

Step requests must wrap the action inside an `action` object:

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"claim_id": "CLM-42-00001", "decision": "clinical_review"}}'
```

Read the current state:

```bash
curl http://localhost:7860/state
```

## Run Locally

### Python

Use the same interpreter for install and execution. On Windows, the project-local environment is typically `envs\\Scripts\\python.exe`.

```bash
python -m venv envs
envs\Scripts\python -m pip install -r requirements.txt
envs\Scripts\python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t claimwatch .
docker run -p 7860:7860 claimwatch
```

## Run The Baseline Agent

The repository includes a baseline agent in `inference.py` that calls an OpenAI-compatible model endpoint and interacts with the environment over WebSocket.

Example environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:7860"
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

## Repository Layout

```text
claimwatch/
|- env/
|  |- models.py
|  |- policies.py
|  |- generator.py
|  |- reward.py
|  |- tasks.py
|  `- claim_env.py
|- server/
|  `- app.py
|- inference.py
|- openenv.yaml
|- Dockerfile
`- requirements.txt
```

