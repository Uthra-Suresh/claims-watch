# ClaimWatch

**Insurance claims triage RL environment — OpenEnv compatible**

An agent sits inside a payer (insurance company) and manages an inbound claims queue each episode. It must route each claim to the correct disposition: auto-approve, clinical review, MD review, request more information, or deny.

---

## Quick start (local, no Docker)

```bash
# 1. Clone / copy these files into a folder
cd claimwatch/

# 2. Install dependencies (only needed for baseline.py)
pip install requests

# 3. Start the environment server
python server.py
# → ClaimWatch environment running on http://localhost:8000

# 4. In another terminal, run the baseline agent
python baseline.py --task 1          # easy: 50 claims
python baseline.py --task 2          # medium: 300 claims
python baseline.py --task 3          # hard: 1000 claims + fraud burst
python baseline.py --all             # all three tasks
python baseline.py --task 2 --verbose  # see per-step decisions
```

---

## Docker setup

```bash
# Build
docker build -t claimwatch .

# Run (server inside container)
docker run -p 8000:8000 claimwatch

# Run baseline against containerised server
python baseline.py --all
```

---

## API reference

All requests/responses are JSON.

### `GET /health`
Returns `{"status": "ok"}` — use for Docker healthcheck.

### `GET /info`
Returns environment metadata: task list, action space, observation keys, reward range.

### `POST /reset`
Start a new episode.

**Request:**
```json
{
  "task": 1,
  "seed": 42
}
```

**Response:**
```json
{
  "session_id": "uuid-string",
  "observation": { ...claim fields... }
}
```

### `POST /step`
Take one action on the current claim.

**Request:**
```json
{
  "session_id": "uuid-string",
  "action": "auto_approve"
}
```

Valid actions: `auto_approve`, `clinical_review`, `md_review`, `request_info`, `deny`

**Response:**
```json
{
  "observation": { ...next claim... },
  "reward": 0.3,
  "done": false,
  "info": {
    "claim_id": "CLM-00001",
    "action_taken": "auto_approve",
    "ground_truth": "auto_approve",
    "reward": 0.3,
    "is_urgent": false,
    "is_fraud": false,
    "step": 1,
    "total_steps": 50
  }
}
```

When `done: true`, the response also includes:
```json
{
  "score": {
    "routing_accuracy": 0.82,
    "reward_score": 0.74,
    "composite": 0.78,
    "total_reward": 12.4,
    "n_claims": 50
  }
}
```

---

## Observation schema

Each observation is a dict with these keys:

| Key | Type | Description |
|---|---|---|
| `claim_id` | str | Unique claim identifier |
| `procedure_code` | str | CPT procedure code |
| `procedure_name` | str | Human-readable procedure name |
| `diagnosis_code` | str | ICD-10 diagnosis code |
| `diagnosis_name` | str | Human-readable diagnosis name |
| `hospital_id` | str | Submitting hospital identifier |
| `patient_acuity` | int | 1 (low) to 5 (critical) |
| `claim_amount` | float | Billed amount in USD |
| `policy_id` | str | Applicable payer policy |
| `policy_name` | str | Human-readable policy name |
| `policy_auto_approve_threshold` | float | Max amount for auto-approval |
| `policy_covered_procedures` | list[str] | Procedures covered by policy |
| `policy_always_review` | list[str] | Procedures requiring clinical review |
| `clinical_note` | str | Free-text clinical documentation |
| `available_actions` | list[str] | Always the full action list |
| `episode_info` | dict | Step count, cumulative reward, remaining claims |

---

## Reward function

| Situation | Action | Reward |
|---|---|---|
| Routine claim, docs complete | `auto_approve` | +0.30 |
| Urgent claim | `md_review` | +0.50 |
| Urgent claim | `auto_approve` | −0.80 |
| Urgent claim | `clinical_review` | −0.40 |
| Routine claim, docs complete | `clinical_review` | −0.20 |
| Fraud claim | `deny` | +0.40 |
| Legitimate claim | `deny` | −0.60 |
| Incomplete docs | `request_info` | +0.20 |
| Complete docs | `request_info` | −0.15 |
| Non-urgent claim | `md_review` | −0.20 |

---

## Tasks

| Task | Claims | Policies | Urgent % | Fraud | Special |
|---|---|---|---|---|---|
| 1 (Easy) | 50 | 1 | 0% | No | Baseline classification |
| 2 (Medium) | 300 | 4 | 15% | No | Urgency detection |
| 3 (Hard) | 1000 | 4 | 12% | Yes | Fraud burst at step 400, mid-episode policy change |

---

## Scores (all 0.0–1.0)

- **routing_accuracy**: fraction of steps with positive reward
- **reward_score**: normalised total reward (worst possible = 0.0, best = 1.0)
- **composite**: average of the two above

Scores are fully deterministic given the same `seed`. The baseline agent with `seed=42` produces reproducible reference scores.

---

## Connecting an LLM agent

```python
import requests, json

BASE = "http://localhost:8000"

def llm_agent(obs: dict) -> str:
    """Replace with your LLM call."""
    prompt = f"""You are a medical claims reviewer. Triage this claim:

Procedure: {obs['procedure_name']} ({obs['procedure_code']})
Diagnosis:  {obs['diagnosis_name']} ({obs['diagnosis_code']})
Amount:     ${obs['claim_amount']}
Acuity:     {obs['patient_acuity']}/5
Policy:     {obs['policy_name']}
Covered:    {obs['policy_covered_procedures']}
Always review: {obs['policy_always_review']}
Note:       {obs['clinical_note']}

Choose exactly one action: auto_approve, clinical_review, md_review, request_info, deny
Respond with only the action name."""
    # response = your_llm_call(prompt)
    # return response.strip()
    return "clinical_review"  # placeholder

r = requests.post(f"{BASE}/reset", json={"task": 1, "seed": 42})
session_id = r.json()["session_id"]
obs = r.json()["observation"]

while True:
    action = llm_agent(obs)
    r = requests.post(f"{BASE}/step", json={"session_id": session_id, "action": action})
    result = r.json()
    if result["done"]:
        print(result["score"])
        break
    obs = result["observation"]
```

---

## File structure

```
claimwatch/
├── environment.py   # Core env logic, claim generation, reward function
├── server.py        # HTTP server (OpenEnv interface)
├── baseline.py      # Rule-based baseline agent
├── Dockerfile       # Container build
├── requirements.txt
└── README.md
```
