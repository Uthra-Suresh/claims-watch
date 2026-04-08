"""ClaimWatch — OpenEnv-compliant RL environment server."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

from env.claim_env import ClaimWatchEnv
from env.models import ClaimAction, ClaimObservation, Reward, RoutingDecision


DESCRIPTION = """
## ClaimWatch — Insurance Claims Triage RL Environment

A production-grade Reinforcement Learning environment that simulates
daily operations of an insurance company's claims intake triage team.

### Quick Start

1. **Reset** the environment: `POST /reset`
2. **Step** through claims: `POST /step` with a claim\_id and decision
3. **Check state**: `GET /state`

### Routing Decisions

| Decision | Description |
|---|---|
| `auto_approve` | Approve routine claims with complete docs |
| `clinical_review` | Route to clinical nurse review |
| `md_review` | Route to physician review (high-cost) |
| `request_info` | Request missing documentation |
| `deny` | Deny non-covered procedures |
| `flag_fraud` | Flag suspected fraud |

### Tasks

| ID | Name | Difficulty | Claims |
|----|------|------------|--------|
| 1 | routine_triage | easy | 20 |
| 2 | multi_hospital_triage | medium | 30 |
| 3 | full_complexity | hard | 50 |
"""

app = create_app(ClaimWatchEnv, ClaimAction, ClaimObservation, env_name="claims_watch")


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


# from typing import Any, Dict, List, Optional

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field

# from env.claim_env import ClaimWatchEnv
# from env.models import Action, Observation, Reward, RoutingDecision
# from env.tasks import TASKS

# # ── App setup ────────────────────────────────────────────────────────────────



# app = FastAPI(
#     title="ClaimWatch",
#     version="1.0.0",
#     description=DESCRIPTION,
#     docs_url="/docs",
#     redoc_url="/redoc",
#     openapi_tags=[
#         {"name": "Environment", "description": "Core RL environment endpoints (reset / step / state)"},
#         {"name": "Info", "description": "Metadata, health checks, and task listing"},
#     ],
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# _env = ClaimWatchEnv()


# # ── Request / response models ────────────────────────────────────────────────

# class ResetRequest(BaseModel):
#     """Parameters for resetting the environment."""
#     task_id: int = Field(1, description="Task ID (1=easy, 2=medium, 3=hard)", ge=1, le=3)
#     seed: int = Field(42, description="Random seed for reproducibility")
#     n_claims: Optional[int] = Field(None, description="Override number of claims (for testing)", ge=1)

#     model_config = {"json_schema_extra": {"examples": [{"task_id": 1, "seed": 42}]}}


# class StepRequest(BaseModel):
#     """One triage action submitted by the agent."""
#     claim_id: str = Field(..., description="ID of the claim to act on (e.g. 'CLM-42-00003')")
#     decision: RoutingDecision = Field(..., description="Routing decision for this claim")
#     rationale: Optional[str] = Field(None, description="Optional explanation (not graded)")

#     model_config = {"json_schema_extra": {"examples": [
#         {"claim_id": "CLM-42-00003", "decision": "clinical_review", "rationale": "Moderate complexity claim"}
#     ]}}


# class StepResponse(BaseModel):
#     """Result of a single step."""
#     observation: Dict[str, Any] = Field(..., description="Agent-visible observation after the step")
#     reward: Dict[str, Any] = Field(..., description="Decomposed reward breakdown")
#     done: bool = Field(..., description="Whether the episode has ended")
#     info: Dict[str, Any] = Field(default_factory=dict, description="Extra info; contains grader_result when done=True")


# class HealthResponse(BaseModel):
#     """Health check response."""
#     status: str = "ok"
#     env: str = "claimwatch"
#     version: str = "1.0.0"


# class TaskInfo(BaseModel):
#     """Metadata for a single task."""
#     id: int
#     name: str
#     difficulty: str
#     n_claims: int


# # ── Core helpers ─────────────────────────────────────────────────────────────

# def _do_reset(task_id: int = 1, seed: int = 42, n_claims: Optional[int] = None) -> dict:
#     """Shared reset logic for both POST and GET."""
#     try:
#         obs = _env.reset(task_id=task_id, seed=seed, n_claims=n_claims)
#         return obs.model_dump()
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))


# def _do_step(claim_id: str, decision: RoutingDecision, rationale: Optional[str] = None) -> dict:
#     """Shared step logic for both POST and GET."""
#     action = Action(claim_id=claim_id, decision=decision, rationale=rationale)
#     try:
#         obs, reward, done, info = _env.step(action)
#         return {
#             "observation": obs.model_dump(),
#             "reward": reward.model_dump(),
#             "done": done,
#             "info": info,
#         }
#     except RuntimeError as e:
#         raise HTTPException(status_code=400, detail=str(e))


# # ── Endpoints ────────────────────────────────────────────────────────────────

# @app.post(
#     "/reset",
#     tags=["Environment"],
#     summary="Reset the environment (POST)",
#     response_description="Observation after reset",
# )
# def reset_post(body: ResetRequest = ResetRequest()):
#     """Reset the environment and start a new episode.

#     Accepts an empty body `{}` — defaults to task_id=1, seed=42.

#     **Returns:** The initial observation with the claim queue, resource state, etc.
#     """
#     return _do_reset(task_id=body.task_id, seed=body.seed, n_claims=body.n_claims)


# @app.post(
#     "/step",
#     tags=["Environment"],
#     summary="Submit one triage action (POST)",
#     response_model=StepResponse,
#     response_description="Observation, reward, done flag, and info",
# )
# def step_post(body: StepRequest):
#     """Submit one routing decision for a claim.

#     The agent picks a claim from the queue and assigns a routing decision.
#     Returns the new observation, decomposed reward, done flag, and info dict.
#     When `done=True`, `info.grader_result` contains the final score.
#     """
#     return _do_step(claim_id=body.claim_id, decision=body.decision, rationale=body.rationale)


# @app.get(
#     "/state",
#     tags=["Environment"],
#     summary="Get full environment state",
#     response_description="Current environment state dict",
# )
# def get_state():
#     """Return the full internal state of the environment.

#     Includes step count, slot availability, episode flags, etc.
#     Useful for debugging and validation.
#     """
#     return {"state": _env.state()}


# @app.get(
#     "/tasks",
#     tags=["Info"],
#     summary="List available tasks",
#     response_model=List[TaskInfo],
#     response_description="Array of task metadata",
# )
# def get_tasks():
#     """Return metadata for all available tasks (difficulty, claim count)."""
#     return [
#         TaskInfo(
#             id=cfg.task_id,
#             name=cfg.name,
#             difficulty=cfg.difficulty,
#             n_claims=cfg.n_claims,
#         )
#         for cfg in TASKS.values()
#     ]


# @app.get(
#     "/health",
#     tags=["Info"],
#     summary="Health check",
#     response_model=HealthResponse,
# )
# def health():
#     """Simple health check — returns `{"status": "ok"}` when the server is running."""
#     return HealthResponse()


# @app.get("/", tags=["Info"], summary="Root info")
# def root():
#     """Root endpoint with environment metadata and available endpoints.

#     Also links to `/docs` for interactive Swagger UI documentation.
#     """
#     return {
#         "name": "ClaimWatch",
#         "version": "1.0.0",
#         "description": "Insurance claims triage RL environment",
#         "openenv_compliant": True,
#         "docs": "/docs",
#         "endpoints": ["/reset", "/step", "/state", "/tasks", "/health", "/docs", "/redoc"],
#     }

# def main() -> Any:
#     """Return the ASGI app for runners/validators to import as ``server.app:main``.

#     Returns the FastAPI `app` instance.
#     """
#     return app


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(main(), host="0.0.0.0", port=8000)