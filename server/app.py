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

app = create_app(ClaimWatchEnv, ClaimAction, ClaimObservation, env_name="claimwatch")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()