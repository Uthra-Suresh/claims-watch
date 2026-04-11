
"""ClaimWatch inference agent.

STDOUT FORMAT
    [START] task=<task_id> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<decision>(claim=<id>) reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import websocket
from urllib.parse import urlparse
from openai import OpenAI

import requests

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
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
BENCHMARK = "claimwatch"

# ── Runtime flags (set by parse_args) ────────────────────────────────────────
DEBUG = False


# ── Structured log helpers (matching reference format) ───────────────────────

def log_start(task_id: int) -> None:
    log(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")


def log_step(step: int, decision: str, claim_id: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    log(
        f"[STEP] step={step} action={decision}(claim={claim_id[:12]}) "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}"
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    log(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}"
    )


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

# ── WebSocket helpers ────────────────────────────────────────────────────────

def build_ws_url(env_base_url: str) -> str:
    """Convert ENV_BASE_URL into the OpenEnv WebSocket endpoint."""
    base = normalize_env_base_url(env_base_url).rstrip("/")

    if base.startswith("http://"):
        ws_base = "ws://" + base[len("http://"):]
    elif base.startswith("https://"):
        ws_base = "wss://" + base[len("https://"):]
    elif base.startswith(("ws://", "wss://")):
        ws_base = base
    else:
        ws_base = "ws://" + base

    return f"{ws_base}/ws"


def env_connect() -> websocket.WebSocket:
    """Open a persistent WebSocket session to the environment."""
    ws_url = build_ws_url(ENV_BASE_URL)
    debug(f"Connecting to environment WebSocket: {ws_url}")
    return websocket.create_connection(ws_url, timeout=30)


def _ws_send_and_receive(session: websocket.WebSocket, message: Dict[str, Any]) -> Dict[str, Any]:
    """Send one WebSocket message and return the inner OpenEnv response payload."""
    session.send(json.dumps(message))

    raw = session.recv()
    debug(f"WS recv: {raw}")
    response = json.loads(raw)

    response_type = response.get("type")
    if response_type == "error":
        data = response.get("data", {})
        raise RuntimeError(
            f"Server error: {data.get('message', 'Unknown error')} "
            f"(code: {data.get('code', 'UNKNOWN')})"
        )

    if response_type not in {"observation", "state"}:
        raise RuntimeError(f"Unexpected WS response type: {response_type!r}")

    return response.get("data", {})


def env_reset(
    session: websocket.WebSocket,
    task_id: int = 1,
    seed: int = 42,
    n_claims: Optional[int] = None,
) -> Dict[str, Any]:
    """Reset the environment over the active WebSocket session."""
    body: Dict[str, Any] = {"task_id": task_id, "seed": seed}
    if n_claims is not None:
        body["n_claims"] = n_claims

    return _ws_send_and_receive(
        session,
        {
            "type": "reset",
            "data": body,
        },
    )


def env_step(
    session: websocket.WebSocket,
    claim_id: str,
    decision: str,
    rationale: str = "",
) -> Dict[str, Any]:
    """Execute one step over the active WebSocket session."""
    return _ws_send_and_receive(
        session,
        {
            "type": "step",
            "data": {
                "claim_id": claim_id,
                "decision": decision,
                "rationale": rationale,
            },
        },
    )


def env_state(session: websocket.WebSocket) -> Dict[str, Any]:
    """Request the current environment state over WebSocket."""
    return _ws_send_and_receive(
        session,
        {
            "type": "state",
        },
    )


def env_close(session: Optional[websocket.WebSocket]) -> None:
    """Close the active WebSocket session."""
    if session is None:
        return

    try:
        session.send(json.dumps({"type": "close"}))
    except Exception:
        pass

    try:
        session.close()
    except Exception:
        pass


# ── Observation helpers ───────────────────────────────────────────────────────

def unwrap_obs(result: Dict[str, Any], prev_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the observation dict from a step/reset response."""
    return result.get("observation", prev_obs)


def get_reward(result: Dict[str, Any], obs: Dict[str, Any]) -> float:
    """Extract scalar reward from a step response.

    OpenEnv StepResponse.reward is Optional[float] at the top level.
    ClaimObservation also carries obs.reward as a fallback.
    """
    r = result.get("reward")
    if isinstance(r, (int, float)) and r is not None:
        return float(r)
    # Fallback: reward embedded in the observation itself
    r = obs.get("reward")
    if isinstance(r, (int, float)) and r is not None:
        return float(r)
    return 0.0


def get_done(result: Dict[str, Any], obs: Dict[str, Any]) -> bool:
    """Extract done flag — check both top-level result and observation."""
    if result.get("done"):
        return True
    return bool(obs.get("done", False))


def get_grader_score(obs: Dict[str, Any]) -> Optional[float]:
    """Extract grader score from observation metadata (set when done=True)."""
    grader = obs.get("metadata", {}).get("grader_result", {})
    if not grader:
        return None
    return grader.get("score")


def normalize_env_base_url(raw_url: str) -> str:
    """Normalize common ENV_BASE_URL mistakes into a valid base URL."""
    candidate = (raw_url or "").strip()
    if not candidate:
        return "http://localhost:8000"

    if "lllocalhost" in candidate.lower():
        candidate = candidate.replace("lllocalhost", "localhost")
        candidate = candidate.replace("LLLOCALHOST", "localhost")

    lowered = candidate.lower()
    if lowered.startswith("https:") and not lowered.startswith("https://"):
        candidate = "https://" + candidate[6:].lstrip("/")
    elif lowered.startswith("http:") and not lowered.startswith("http://"):
        candidate = "http://" + candidate[5:].lstrip("/")
    elif "://" not in candidate:
        candidate = f"http://{candidate.lstrip('/')}"

    parsed = urlparse(candidate)
    if not parsed.netloc and parsed.path:
        parsed = urlparse(f"{parsed.scheme}://{parsed.path.lstrip('/')}")

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid ENV_BASE_URL: {raw_url!r}")

    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def validate_action(obs: Dict[str, Any], parsed: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Validate and normalize an action before sending it to /step."""
    queue = obs.get("queue", [])
    if not queue:
        return None

    valid_claim_ids = {claim.get("claim_id") for claim in queue}
    valid_decisions = {
        "auto_approve",
        "clinical_review",
        "md_review",
        "request_info",
        "deny",
        "flag_fraud",
    }

    claim_id = str(parsed.get("claim_id", "")).strip()
    decision = str(parsed.get("decision", "")).strip().lower()
    rationale = str(parsed.get("rationale", "")).strip()

    if claim_id not in valid_claim_ids:
        debug(f"Invalid claim_id from model: {claim_id!r}")
        return None

    if decision not in valid_decisions:
        debug(f"Invalid decision from model: {decision!r}")
        return None

    return {"claim_id": claim_id, "decision": decision, "rationale": rationale}


# ── Prompt / LLM helpers ──────────────────────────────────────────────────────

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
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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
                    continue
                return None

            parsed = _extract_json(content)
            if parsed is not None:
                return parsed

            if use_json_format:
                continue
            debug(f"LLM returned unparseable content: {content[:200]}")
            return None
        except Exception as exc:
            debug(f"LLM call error (json_format={use_json_format}): {type(exc).__name__}: {exc}")
            if use_json_format:
                continue
    return None


# ── Debug helpers ────────────────────────────────────────────────────────────

def _dbg_claim(claim: Dict[str, Any]) -> str:
    return (
        f"  {claim['claim_id'][:12]} | sla={claim.get('sla_tier', '?')} "
        f"rem={claim.get('sla_remaining_hr', 0):.1f}hr "
        f"${claim.get('billed_amount', 0):.0f} "
        f"docs={len(claim.get('documentation', []))}"
    )


def _dbg_state(obs: Dict[str, Any]) -> str:
    return (
        f"  queue={len(obs.get('queue', []))} md={obs.get('md_slots_remaining', '?')} "
        f"clin={obs.get('clinical_slots_remaining', '?')}"
    )


# ── Main inference loop ───────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: int, n_claims: Optional[int] = None) -> Dict[str, Any]:
    """Run a single task and return the result."""
    log_start(task_id)
    session = env_connect()
    try:
        reset_result = env_reset(session, task_id=task_id, seed=42, n_claims=n_claims)
        obs = unwrap_obs(reset_result, {})

        total_claims = obs.get("total_claims_in_episode", n_claims or 0)
        max_steps = int(total_claims) if total_claims else len(obs.get("queue", []))

        rewards: List[float] = []
        step_count = 0
        completed_episode = False

        for n in range(1, max_steps + 1):
            queue = obs.get("queue", [])
            if not queue:
                debug(f"Step {n}: empty queue — episode complete")
                completed_episode = True
                break

            debug(f"--- step {n}/{max_steps} ---")
            debug(_dbg_state(obs))
            for c in queue[:3]:
                debug(_dbg_claim(c))

            prompt = build_claim_prompt(obs)
            parsed = call_llm(client, prompt)

            if parsed is None or "claim_id" not in parsed or "decision" not in parsed:
                parsed = fallback_heuristic(obs)
                debug(f"Step {n}: LLM failed → fallback heuristic")
            else:
                parsed = validate_action(obs, parsed)

            if parsed is None:
                parsed = fallback_heuristic(obs)
                if parsed is None:
                    log(f"[STEP] step={n} action=noop(claim=none) reward=0.00 done=false error=fallback_failed")
                    break

            claim_id = parsed["claim_id"]
            decision = parsed["decision"]
            rationale = parsed.get("rationale", "")
            step_error: Optional[str] = None

            try:
                result = env_step(session, claim_id, decision, rationale)
                obs = unwrap_obs(result, obs)
                reward_val = get_reward(result, obs)
                done = get_done(result, obs)
                rewards.append(reward_val)
                step_count = n

                log_step(n, decision, claim_id, reward_val, done, step_error)

                if done:
                    completed_episode = True
                    break

            except Exception as e:
                step_error = str(e)
                log_step(n, decision, claim_id, 0.0, False, step_error)
                rewards.append(0.0)
                step_count = n
        else:
            # Loop exhausted max_steps without done signal
            log(f"  WARNING: reached max_steps={max_steps} without done signal")

        final_score = sum(rewards) / len(rewards) if rewards else 0.0
        final_score = max(0.0, min(1.0, final_score))
        success = completed_episode

        log_end(success, step_count, final_score, rewards)
        return {"task_id": task_id, "score": final_score, "steps": step_count, "success": success}
    finally:
        env_close(session)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ClaimWatch inference agent")
    p.add_argument("--easy", action="store_true", help="Run task 1 (easy, 20 claims)")
    p.add_argument("--medium", action="store_true", help="Run task 2 (medium, 30 claims)")
    p.add_argument("--hard", action="store_true", help="Run task 3 (hard, 50 claims)")
    p.add_argument("--n-claims", type=int, default=None, help="Override claim count for all tasks")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug output")
    return p.parse_args()


def main() -> None:
    """Run selected tasks (default: all 3 -> 100 claims total)."""
    global DEBUG
    global ENV_BASE_URL

    args = parse_args()
    if args.debug:
        DEBUG = True

    ENV_BASE_URL = normalize_env_base_url(ENV_BASE_URL)

    task_ids: List[int] = []
    if args.easy:
        task_ids.append(1)
    if args.medium:
        task_ids.append(2)
    if args.hard:
        task_ids.append(3)
    if not task_ids:
        task_ids = [1, 2, 3]  # default: all tasks = 20 + 30 + 50 = 100 claims

    if args.n_claims is not None:
        log(f"Overriding claim count → {args.n_claims} per task")

    if not HF_TOKEN:
        log("WARNING: No API key found. Set HF_TOKEN or API_KEY env var.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    start_time = time.time()
    results = []
    for task_id in task_ids:
        result = run_task(client, task_id, n_claims=args.n_claims)
        results.append(result)

    elapsed = time.time() - start_time
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    debug(f"\n=== Summary (elapsed={elapsed:.1f}s) ===")
    for r in results:
        debug(f"  Task {r['task_id']}: score={r['score']:.3f} steps={r['steps']} success={r['success']}")
    debug(f"  Average score: {avg_score:.3f}")


if __name__ == "__main__":
    main()
