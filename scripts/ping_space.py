"""Ping a Hugging Face Space (or any deployment) to ensure it's up and responds to /reset.

Usage:
  python scripts/ping_space.py --url https://<your-space>.hf.space

Checks:
  - GET /health returns 200
  - POST /reset (with empty JSON) returns 200 and contains `total_claims_in_episode` and `step_number`==0

Retries for a short period to allow slow cold-starting Spaces.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import requests


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def check_health(base_url: str, timeout: int = 5) -> bool:
    try:
        url = join_url(base_url, "/health")
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def check_reset(base_url: str, timeout: int = 10) -> Optional[str]:
    try:
        url = join_url(base_url, "/reset")
        r = requests.post(url, json={}, timeout=timeout)
        if r.status_code != 200:
            return f"reset returned status {r.status_code}"
        data = r.json()
        # Expect core OpenEnv response keys
        if not ("total_claims_in_episode" in data and "step_number" in data):
            return "reset response missing expected keys"
        if int(data.get("step_number", -1)) != 0:
            return f"reset.step_number != 0 (got {data.get('step_number')})"
        return None
    except Exception as e:
        return str(e)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", "-u", help="Base URL of the Space (e.g. https://<user>-<space>.hf.space)")
    p.add_argument("--attempts", type=int, default=12, help="Number attempts (default=12)")
    p.add_argument("--interval", type=int, default=5, help="Seconds between attempts (default=5)")
    args = p.parse_args()

    base_url = args.url or os.getenv("HF_SPACE_URL")
    if not base_url:
        print("ERROR: supply --url or set HF_SPACE_URL environment variable", file=sys.stderr)
        sys.exit(2)

    print(f"Pinging Space at {base_url}")

    for attempt in range(1, args.attempts + 1):
        print(f"Attempt {attempt}/{args.attempts}: checking /health...")
        if check_health(base_url):
            print("  /health OK")
            print("  calling /reset to verify environment responds to reset()")
            err = check_reset(base_url)
            if err is None:
                print("SUCCESS: Space responded to /health and /reset as expected")
                sys.exit(0)
            else:
                print(f"  /reset check failed: {err}")
        else:
            print("  /health not ready yet")

        if attempt < args.attempts:
            time.sleep(args.interval)

    print("FAILED: Space did not become ready or /reset did not respond correctly", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
