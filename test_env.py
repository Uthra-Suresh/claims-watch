"""ClaimWatch — 6 smoke tests. All must pass before submission."""

from __future__ import annotations

import sys
import traceback

from env.models import Action, Claim, Observation, Reward, RoutingDecision, SLATier, SLA_HOURS
from env.claim_env import ClaimWatchEnv
from env.generator import generate_claims
from env.reward import compute_reward
from env.tasks import TASKS, grade_task1, grade_task2, grade_task3


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ── Test 1: Reward Engine ────────────────────────────────────────────────────

def test_reward_engine() -> None:
    """Correct fraud flag > 0; missed fraud < -0.3; deadline bonus applied; all combos in [-1, 1]."""
    # Correct fraud flag
    claim_fraud = Claim(
        claim_id="TEST-FRAUD-001",
        hospital_id="HOSP-0001",
        procedure_code="99291",
        procedure_description="Critical care",
        diagnosis_code="I21.3",
        diagnosis_description="STEMI",
        billed_amount=2400.0,
        documentation=["physician_notes", "lab_results"],
        sla_tier=SLATier.urgent,
        sla_deadline_hr=3,
        arrival_hr=0,
        is_fraud=True,
        ground_truth_routing=RoutingDecision.flag_fraud,
    )
    r = compute_reward(claim_fraud, RoutingDecision.flag_fraud, current_hr=1)
    _assert(r.total > 0, f"Correct fraud flag should be positive, got {r.total}")
    _assert(r.correct_fraud_flag == 0.45, f"Expected 0.45, got {r.correct_fraud_flag}")
    _assert(r.deadline_bonus == 0.10, f"On-time bonus expected 0.10, got {r.deadline_bonus}")

    # Missed fraud
    r2 = compute_reward(claim_fraud, RoutingDecision.auto_approve, current_hr=1)
    _assert(r2.total < -0.3, f"Missed fraud should be < -0.3, got {r2.total}")

    # Deadline miss penalty: correct decision but late
    claim_late = Claim(
        claim_id="TEST-LATE-001",
        hospital_id="HOSP-0001",
        procedure_code="99213",
        procedure_description="Office visit",
        diagnosis_code="M54.5",
        diagnosis_description="Low back pain",
        billed_amount=100.0,
        documentation=["physician_notes"],
        sla_tier=SLATier.routine,
        sla_deadline_hr=24,
        arrival_hr=0,
        is_fraud=False,
        ground_truth_routing=RoutingDecision.auto_approve,
    )
    r3 = compute_reward(claim_late, RoutingDecision.auto_approve, current_hr=25)
    _assert(r3.deadline_miss_penalty == -0.15, f"Late penalty expected -0.15, got {r3.deadline_miss_penalty}")
    _assert(r3.deadline_bonus == 0.0, f"Should have no bonus when late, got {r3.deadline_bonus}")

    # All combos (6 decisions × 30 claim variants) in [-1, 1]
    decisions = list(RoutingDecision)
    claims_variants = generate_claims(30, seed=99)
    for claim in claims_variants:
        for dec in decisions:
            r = compute_reward(claim, dec)
            _assert(-1.0 <= r.total <= 1.0, f"Reward {r.total} out of [-1,1] for {claim.claim_id} / {dec}")

    print("  ✓ test_reward_engine passed")


# ── Test 2: Graders ──────────────────────────────────────────────────────────

def test_graders() -> None:
    """Oracle > random; gap > 0.5; all scores in [0, 1]."""
    for task_id in [1, 2, 3]:
        cfg = TASKS[task_id]
        claims = generate_claims(
            n=cfg.n_claims,
            seed=cfg.seed,
            fraud_rate=cfg.fraud_rate,
        )

        oracle = {c.claim_id: c.ground_truth_routing for c in claims}

        import random
        rng = random.Random(12345)
        all_decs = list(RoutingDecision)
        rand_decisions = {c.claim_id: rng.choice(all_decs) for c in claims}

        if task_id == 1:
            oracle_result = grade_task1(claims, oracle)
            rand_result = grade_task1(claims, rand_decisions)
        elif task_id == 2:
            oracle_result = grade_task2(claims, oracle)
            rand_result = grade_task2(claims, rand_decisions)
        else:
            oracle_result = grade_task3(claims, oracle)
            rand_result = grade_task3(claims, rand_decisions)

        oracle_score = oracle_result["score"]
        rand_score = rand_result["score"]

        _assert(0.0 <= oracle_score <= 1.0, f"Task {task_id} oracle score {oracle_score} out of [0,1]")
        _assert(0.0 <= rand_score <= 1.0, f"Task {task_id} random score {rand_score} out of [0,1]")
        _assert(oracle_score > rand_score, f"Task {task_id} oracle {oracle_score} should beat random {rand_score}")
        gap = oracle_score - rand_score
        _assert(gap > 0.5, f"Task {task_id} gap {gap:.3f} should be > 0.5")

    print("  ✓ test_graders passed")


# ── Test 3: Environment Episodes ─────────────────────────────────────────────

def test_environment_episodes() -> None:
    """Reset returns correct claim count; reward bounds; state keys."""
    env = ClaimWatchEnv()

    for task_id in [1, 2, 3]:
        obs = env.reset(task_id=task_id, seed=42)
        cfg = TASKS[task_id]

        _assert(obs.total_claims_in_episode == cfg.n_claims,
                f"Task {task_id}: expected {cfg.n_claims} claims, got {obs.total_claims_in_episode}")
        _assert(len(obs.queue) <= 50, f"Queue should be capped at 50, got {len(obs.queue)}")
        _assert(obs.step_number == 0, "Step should be 0 after reset")

        if obs.queue:
            claim = obs.queue[0]
            action = Action(claim_id=claim.claim_id, decision=RoutingDecision.clinical_review)
            obs2, reward, done, info = env.step(action)
            _assert(-1.0 <= reward.total <= 1.0, f"Reward {reward.total} out of bounds")

        state = env.state()
        required_keys = [
            "task_id", "seed", "step_number", "current_day", "current_hour",
            "total_claims", "pending_count", "decided_count",
            "md_slots_remaining", "clinical_slots_remaining",
            "policy_update_fired", "done", "reward_history_len",
            "cumulative_reward",
        ]
        for key in required_keys:
            _assert(key in state, f"State missing key: {key}")

    print("  ✓ test_environment_episodes passed")


# ── Test 4: Mid-Episode Mechanics ────────────────────────────────────────────

def test_mid_episode_mechanics() -> None:
    """Policy update fires at step 30 (Task 2)."""
    env = ClaimWatchEnv()

    obs = env.reset(task_id=2, seed=42)
    for i in range(30):
        if not obs.queue:
            break
        claim = obs.queue[0]
        action = Action(claim_id=claim.claim_id, decision=RoutingDecision.clinical_review)
        obs_new, reward, done, info = env.step(action)
        obs = Observation(**obs_new.model_dump()) if not done else obs_new
        if done:
            break

    state = env.state()
    _assert(state["policy_update_fired"], "Policy update should have fired by step 30")

    print("  ✓ test_mid_episode_mechanics passed")


# ── Test 5: Resource Constraints ─────────────────────────────────────────────

def test_resource_constraints() -> None:
    """Slots deplete correctly; downgrade chain works."""
    env = ClaimWatchEnv()
    obs = env.reset(task_id=1, seed=42)

    initial_md = obs.md_slots_remaining
    initial_clinical = obs.clinical_slots_remaining

    md_used = 0
    for _ in range(min(initial_md + 5, len(obs.queue))):
        if not obs.queue:
            break
        claim = obs.queue[0]
        action = Action(claim_id=claim.claim_id, decision=RoutingDecision.md_review)
        obs, reward, done, info = env.step(action)
        md_used += 1
        if done:
            break

    state = env.state()
    if md_used >= initial_md:
        _assert(state["md_slots_remaining"] <= 0,
                f"MD slots should be depleted after {md_used} uses, got {state['md_slots_remaining']}")

    # Test downgrade: reset and exhaust MD slots, then try MD_REVIEW
    obs = env.reset(task_id=2, seed=42)
    cfg = TASKS[2]

    for i in range(cfg.md_slots_per_day):
        if not obs.queue:
            break
        claim = obs.queue[0]
        action = Action(claim_id=claim.claim_id, decision=RoutingDecision.md_review)
        obs, reward, done, info = env.step(action)
        if done:
            break

    if obs.queue and not done:
        claim = obs.queue[0]
        pre_clinical = obs.clinical_slots_remaining
        action = Action(claim_id=claim.claim_id, decision=RoutingDecision.md_review)
        obs, reward, done, info = env.step(action)
        _assert(obs.clinical_slots_remaining < pre_clinical,
                "MD_REVIEW should downgrade to clinical_review when MD slots exhausted")

    print("  ✓ test_resource_constraints passed")


# ── Test 6: OpenEnv Spec Compliance ──────────────────────────────────────────

def test_openenv_spec_compliance() -> None:
    """Observation has 10 fields; step returns 4-tuple; episode terminates."""
    env = ClaimWatchEnv()
    obs = env.reset(task_id=1, seed=42)

    obs_dict = obs.model_dump()
    expected_fields = [
        "current_hour", "current_day", "queue", "processed_today",
        "md_slots_remaining", "clinical_slots_remaining",
        "policy_update_active",
        "total_claims_in_episode",
        "task_id", "step_number",
    ]
    for f in expected_fields:
        _assert(f in obs_dict, f"Observation missing field: {f}")
    _assert(len(expected_fields) == 10, "Should check 10 fields")

    # Step returns 4-tuple
    if obs.queue:
        claim = obs.queue[0]
        action = Action(claim_id=claim.claim_id, decision=RoutingDecision.auto_approve)
        result = env.step(action)
        _assert(len(result) == 4, f"Step should return 4-tuple, got {len(result)}")
        obs_r, reward_r, done_r, info_r = result
        _assert(isinstance(obs_r, Observation), "First element should be Observation")
        _assert(isinstance(reward_r, Reward), "Second element should be Reward")
        _assert(isinstance(done_r, bool), "Third element should be bool")
        _assert(isinstance(info_r, dict), "Fourth element should be dict")

    # Episode terminates (all claims processed)
    obs = env.reset(task_id=1, seed=42)
    steps = 0
    done = False
    while True:
        if not obs.queue:
            break
        claim = obs.queue[0]
        action = Action(claim_id=claim.claim_id, decision=RoutingDecision.auto_approve)
        obs, reward, done, info = env.step(action)
        steps += 1
        if done:
            _assert("grader_result" in info, "Done episode must include grader_result in info")
            _assert(0.0 <= info["grader_result"]["score"] <= 1.0, "Score must be in [0, 1]")
            break
    _assert(done, "Episode should terminate")

    print("  ✓ test_openenv_spec_compliance passed")


# ── Runner ───────────────────────────────────────────────────────────────────

def main() -> None:
    tests = [
        ("test_reward_engine", test_reward_engine),
        ("test_graders", test_graders),
        ("test_environment_episodes", test_environment_episodes),
        ("test_mid_episode_mechanics", test_mid_episode_mechanics),
        ("test_resource_constraints", test_resource_constraints),
        ("test_openenv_spec_compliance", test_openenv_spec_compliance),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} FAILED: {e}")
            traceback.print_exc()

    total = passed + failed
    print(f"\n{'=' * 40}")
    if failed == 0:
        print(f"ALL {total}/{total} TESTS PASSED ✓")
    else:
        print(f"{passed}/{total} tests passed, {failed} FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
