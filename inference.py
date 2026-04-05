"""
ClaimWatch — Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the
  root directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

import requests
from openai import OpenAI

load_dotenv()

# ── Log file setup ───────────────────────────────────────────────────────────
_log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
_log_file = open(f"logs/run_{_log_ts}.log", "a", encoding="utf-8")


def log(msg: str) -> None:
    """Print to stdout and append to the run log file."""
    print(msg, flush=True)
    _log_file.write(msg + "\n")
    _log_file.flush()


def debug(msg: str) -> None:
    """Print only when --debug is active; always write to log file."""
    _log_file.write(f"[DEBUG] {msg}\n")
    _log_file.flush()
    if DEBUG:
        print(msg, flush=True)

# ── Environment Variables ────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "claimwatch"
SUCCESS_THRESHOLD = 0.40

# ── Runtime flags (set by parse_args) ────────────────────────────────────────
DEBUG = False

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert insurance claims triage specialist. Your role is to route incoming medical claims through the correct decision pathway.

You have 6 possible routing decisions:
1. auto_approve — For routine, low-cost claims with complete documentation
2. clinical_review — For moderate-complexity claims needing clinical nurse review
3. md_review — For high-cost/complex claims needing physician review
4. request_info — When required documentation is missing
5. deny — For non-covered procedures or services not in the plan
6. flag_fraud — When billing patterns look abnormal (e.g. billed amount far exceeds normal range)

Each claim has an SLA tier (routine=24hr, critical=12hr, urgent=3hr) and remaining SLA hours.
Prioritise claims whose SLA is about to expire.

DECISION PRIORITY (highest to lowest):
1. FRAUD: Abnormal billing patterns (billed >> expected) → flag_fraud
2. COVERAGE: Non-covered procedures → deny
3. DOCUMENTATION: Missing required docs → request_info
4. COST: Extremely high billed amounts (>1.5x normal) → md_review
5. ROUTINE: Simple claims with complete docs → auto_approve
6. DEFAULT: Route based on procedure complexity

You must respond with valid JSON containing:
- claim_id: the exact claim ID to act on
- decision: one of the 6 routing decisions
- rationale: brief explanation

Always prioritise claims with the lowest SLA remaining hours first."""


# ── Helper functions ─────────────────────────────────────────────────────────

def env_reset(task_id: int = 1, seed: int = 42, n_claims: Optional[int] = None) -> Dict[str, Any]:
    """POST /reset to the environment."""
    body: Dict[str, Any] = {"task_id": task_id, "seed": seed}
    if n_claims is not None:
        body["n_claims"] = n_claims
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(claim_id: str, decision: str, rationale: str = "") -> Dict[str, Any]:
    """POST /step to the environment."""
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"claim_id": claim_id, "decision": decision, "rationale": rationale},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def build_claim_prompt(obs: Dict[str, Any]) -> str:
    """Build a prompt from the top 5 claims in the observation queue."""
    queue = obs.get("queue", [])
    top_claims = queue[:5]

    if not top_claims:
        return "No claims in queue."

    lines = [
        f"Current state: Day {obs.get('current_day', 0)}, Hour {obs.get('current_hour', 0)}",
        f"MD slots remaining: {obs.get('md_slots_remaining', 0)}",
        f"Clinical slots remaining: {obs.get('clinical_slots_remaining', 0)}",
        f"Policy update active: {obs.get('policy_update_active', False)}",
        "",
        "Top claims to triage (pick ONE to act on):",
    ]

    for i, claim in enumerate(top_claims, 1):
        lines.append(f"\n--- Claim {i} ---")
        lines.append(f"ID: {claim['claim_id']}")
        lines.append(f"Procedure: {claim['procedure_code']} — {claim.get('procedure_description', '')}")
        lines.append(f"Diagnosis: {claim['diagnosis_code']} — {claim.get('diagnosis_description', '')}")
        lines.append(f"Billed: ${claim['billed_amount']:.2f}")
        lines.append(f"SLA: {claim.get('sla_tier', 'routine')} (remaining: {claim.get('sla_remaining_hr', 24):.1f}hr)")
        lines.append(f"Documentation: {', '.join(claim.get('documentation', [])) or 'none'}")
        lines.append(f"Hospital: {claim['hospital_id']}")
        lines.append(f"Patient age: {claim.get('patient_age', 'N/A')}")
        lines.append(f"Days in queue: {claim.get('days_in_queue', 0)}")

    lines.append("\nRespond with JSON: {\"claim_id\": \"...\", \"decision\": \"...\", \"rationale\": \"...\"}")
    return "\n".join(lines)


def fallback_heuristic(obs: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Fallback heuristic when LLM fails."""
    queue = obs.get("queue", [])
    if not queue:
        return None

    claim = queue[0]
    claim_id = claim["claim_id"]
    sla_remaining = claim.get("sla_remaining_hr", 24.0)
    billed = claim.get("billed_amount", 0)
    docs = claim.get("documentation", [])

    if billed > 10000 and len(docs) == 0:
        decision = "flag_fraud"
    elif sla_remaining <= 3 and billed > 5000:
        decision = "md_review"
    elif len(docs) == 0:
        decision = "request_info"
    else:
        decision = "clinical_review"

    return {"claim_id": claim_id, "decision": decision}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON from raw text, handling markdown fences."""
    text = text.strip()
    if not text:
        return None
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def call_llm(client: OpenAI, prompt: str) -> Optional[Dict[str, Any]]:
    """Call the LLM and parse the JSON response."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Try with response_format first, fall back to plain call
    for use_json_format in (True, False):
        try:
            kwargs: Dict[str, Any] = dict(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
            )
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if not content:
                if use_json_format:
                    continue  # retry without json format
                return None

            parsed = _extract_json(content)
            if parsed is not None:
                return parsed

            if use_json_format:
                continue  # retry without json format
            debug(f"LLM returned unparseable content: {content[:200]}")
            return None
        except Exception as exc:
            debug(f"LLM call error (json_format={use_json_format}): {type(exc).__name__}: {exc}")
            if use_json_format:
                continue
    return None


# ── Debug helpers ────────────────────────────────────────────────────────────

def _dbg_claim(claim: Dict[str, Any]) -> str:
    """One-line summary of a claim for debug output."""
    return (
        f"  {claim['claim_id'][:12]} | sla={claim.get('sla_tier', '?')} "
        f"rem={claim.get('sla_remaining_hr', 0):.1f}hr "
        f"${claim.get('billed_amount', 0):.0f} "
        f"docs={len(claim.get('documentation', []))}"
    )


def _dbg_reward(reward: Dict[str, Any]) -> str:
    """Compact reward breakdown."""
    parts = [f"{k}={v:.2f}" for k, v in reward.items() if k != "total" and v != 0.0]
    return f"total={reward.get('total', 0):.2f} ({', '.join(parts)})" if parts else f"total={reward.get('total', 0):.2f}"


def _dbg_state(obs: Dict[str, Any]) -> str:
    """Compact resource state."""
    return (
        f"  queue={len(obs.get('queue', []))} md={obs.get('md_slots_remaining', '?')} "
        f"clin={obs.get('clinical_slots_remaining', '?')}"
    )


# ── Main inference loop ──────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: int) -> Dict[str, Any]:
    """Run a single task and return the result."""
    log(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    n_claims_override = os.environ.get("_CLAIMWATCH_N_CLAIMS")
    obs = env_reset(
        task_id=task_id, seed=42,
        n_claims=int(n_claims_override) if n_claims_override else None,
    )
    rewards: List[float] = []
    step_count = 0

    debug(f"Task {task_id}: claims={len(obs.get('queue', []))}")

    n = 0
    while True:
        n += 1
        queue = obs.get("queue", [])
        if not queue:
            break

        prompt = build_claim_prompt(obs)

        if DEBUG:
            debug(f"--- step {n} ---")
            debug(_dbg_state(obs))
            for c in queue[:3]:
                debug(_dbg_claim(c))

        parsed = call_llm(client, prompt)

        if parsed is None or "claim_id" not in parsed or "decision" not in parsed:
            parsed = fallback_heuristic(obs)
            debug("LLM failed → fallback heuristic")
            if parsed is None:
                break

        claim_id = parsed["claim_id"]
        decision = parsed["decision"]
        rationale = parsed.get("rationale", "")
        error_msg = "null"

        try:
            result = env_step(claim_id, decision, rationale)
            reward_val = result.get("reward", {}).get("total", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", obs)
            rewards.append(reward_val)
            step_count = n

            step_line = (
                f"[STEP] step={n} action={decision}(claim={claim_id[:12]}) "
                f"reward={reward_val:.2f} done={str(done).lower()} error={error_msg}"
            )
            log(step_line)

            if DEBUG:
                debug(f"  reward: {_dbg_reward(result.get('reward', {}))}")

            if done:
                info = result.get("info", {})
                grader = info.get("grader_result", {})
                score = grader.get("score", 0.0)
                success = score >= SUCCESS_THRESHOLD
                rewards_str = ",".join(f"{r:.2f}" for r in rewards)
                end_line = (
                    f"[END] success={str(success).lower()} steps={step_count} "
                    f"score={score:.3f} rewards={rewards_str}"
                )
                log(end_line)
                return {"task_id": task_id, "score": score, "steps": step_count, "success": success}

        except Exception as e:
            error_msg = str(e)
            step_line = (
                f"[STEP] step={n} action={decision}(claim={claim_id[:12]}) "
                f"reward=0.00 done=false error={error_msg}"
            )
            log(step_line)
            step_count = n
            rewards.append(0.0)

    # Episode ended without done signal (max steps or empty queue)
    score = 0.0
    success = False
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    end_line = (
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={score:.3f} rewards={rewards_str}"
    )
    log(end_line)
    return {"task_id": task_id, "score": score, "steps": step_count, "success": success}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ClaimWatch inference agent")
    p.add_argument("--easy", action="store_true", help="Run task 1 (easy)")
    p.add_argument("--medium", action="store_true", help="Run task 2 (medium)")
    p.add_argument("--hard", action="store_true", help="Run task 3 (hard)")
    p.add_argument("--n-claims", type=int, default=None, help="Override claim count for all tasks")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    return p.parse_args()


def main() -> None:
    """Run selected tasks (default: all 3)."""
    global DEBUG

    args = parse_args()

    # Debug mode: enable debug flag (console sink already filters to DEBUG-only)
    if args.debug:
        DEBUG = True

    # Determine which tasks to run
    task_ids: List[int] = []
    if args.easy:
        task_ids.append(1)
    if args.medium:
        task_ids.append(2)
    if args.hard:
        task_ids.append(3)
    if not task_ids:
        task_ids = [1, 2, 3]

    # Apply --n-claims override if given (passed in /reset body)
    if args.n_claims is not None:
        os.environ["_CLAIMWATCH_N_CLAIMS"] = str(args.n_claims)
        log(f"Overriding claim count → {args.n_claims}")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    results = []
    for task_id in task_ids:
        result = run_task(client, task_id)
        results.append(result)

    print("\n=== Summary ===")
    for r in results:
        print(f"Task {r['task_id']}: score={r['score']:.3f} steps={r['steps']} success={r['success']}")


if __name__ == "__main__":
    main()
