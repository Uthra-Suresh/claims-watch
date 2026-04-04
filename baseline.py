"""
ClaimWatch baseline agent.

Uses ClaimWatchClient (plain WebSocket, no openenv-core) to talk to the server.

Usage:
    python baseline.py --task 1
    python baseline.py --task 2
    python baseline.py --task 3
    python baseline.py --all
    python baseline.py --all --verbose
"""

import argparse
import asyncio
from client import ClaimWatchClient
from models import ClaimAction

BASE_URL = "ws://localhost:8000/ws"


# ---------------------------------------------------------------------------
# Rule-based agent
# ---------------------------------------------------------------------------

def rule_agent(obs: dict) -> str:
    note      = obs.get("clinical_note", "").lower()
    acuity    = obs.get("patient_acuity", 1)
    amount    = obs.get("claim_amount", 0)
    proc      = obs.get("procedure_code", "")
    always_review = obs.get("policy_always_review", [])
    covered   = obs.get("policy_covered_procedures", [])
    threshold = obs.get("policy_auto_approve_threshold", 500)

    if any(k in note for k in ["duplicate", "not rendered", "multiple procedures same day"]):
        return "deny"
    if any(k in note for k in ["urgent", "stat", "critical", "chest pain", "altered mental"]) or acuity >= 5:
        return "md_review"
    if any(k in note for k in ["incomplete", "to follow", "pending"]):
        return "request_info"
    if proc in always_review:
        return "clinical_review"
    if proc in covered and amount <= threshold:
        return "auto_approve"
    return "clinical_review"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_task(task: int, seed: int = 42, verbose: bool = False) -> dict:
    async with ClaimWatchClient(BASE_URL) as client:
        data = await client.reset(task=task, seed=seed)
        obs = data["observation"]
        step = 0

        while True:
            action = rule_agent(obs)
            result = await client.step(ClaimAction(action=action))

            if verbose and step % 50 == 0:
                info = result.get("info", {})
                print(
                    f"  step {step:4d} | "
                    f"action={action:20s} | "
                    f"reward={result.get('reward', 0):+.2f} | "
                    f"gt={info.get('ground_truth', '?')}"
                )

            if result.get("done"):
                score = result.get("score", {})
                return {
                    "task": task,
                    "steps": step + 1,
                    "score": score,
                }

            obs = result["observation"]
            step += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="ClaimWatch rule-based baseline")
    parser.add_argument("--task", type=int, choices=[1, 2, 3])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.all and args.task is None:
        parser.error("Specify --task 1/2/3 or --all")

    tasks = [1, 2, 3] if args.all else [args.task]

    print(f"\nClaimWatch Baseline Agent  (seed={args.seed})")
    print("=" * 55)

    for task in tasks:
        print(f"\nTask {task}...")
        result = await run_task(task, seed=args.seed, verbose=args.verbose)
        score = result.get("score", {})
        print(f"  steps            : {result['steps']}")
        print(f"  routing_accuracy : {score.get('routing_accuracy', 'n/a')}")
        print(f"  reward_score     : {score.get('reward_score', 'n/a')}")
        print(f"  composite        : {score.get('composite', 'n/a')}")
        print(f"  total_reward     : {score.get('total_reward', 'n/a')}")

    print(f"\n{'=' * 55}")
    print("Done. Scores are reproducible with the same --seed.")


if __name__ == "__main__":
    asyncio.run(main())
