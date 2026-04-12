# baseline_scores.py
from env.generator import generate_claims
from env.tasks import TASKS, grade_task1, grade_task2, grade_task3
from models import RoutingDecision
import random

GRADERS = {1: grade_task1, 2: grade_task2, 3: grade_task3}

rng = random.Random(12345)
all_decs = list(RoutingDecision)

for task_id in [1, 2, 3]:
    cfg = TASKS[task_id]
    claims = generate_claims(n=cfg.n_claims, seed=cfg.seed, fraud_rate=cfg.fraud_rate)

    # Oracle: perfect decisions
    oracle_decs = {c.claim_id: c.ground_truth_routing for c in claims}
    oracle_result = GRADERS[task_id](claims, oracle_decs)

    # Random: uniformly random decisions
    rand_decs = {c.claim_id: rng.choice(all_decs) for c in claims}
    rand_result = GRADERS[task_id](claims, rand_decs)

    print(f"Task {task_id}: oracle={oracle_result['score']:.3f}  random={rand_result['score']:.3f}")