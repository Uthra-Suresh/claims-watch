"""
ClaimWatch — FastAPI server (no openenv-core dependency).

Endpoints:
  WS   /ws       — WebSocket session (reset / step / state)
  GET  /health   — healthcheck
  GET  /info     — environment metadata
  GET  /web      — Gradio debug UI
  GET  /         — redirects to /web
"""

import os
import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse

from models import ClaimAction, ClaimObservation, ClaimState
from environment import ClaimWatchEnv, ACTIONS

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ClaimWatch", version="1.0.0")

# ---------------------------------------------------------------------------
# ClaimWatchServer — one instance per WebSocket session
# ---------------------------------------------------------------------------

class ClaimWatchServer:
    """Manages a single episode over a WebSocket connection."""

    def __init__(self):
        self._env: ClaimWatchEnv | None = None

    def reset(self, task: int = 1, seed: int = 42) -> dict:
        self._env = ClaimWatchEnv(task=task, seed=seed)
        obs = self._env.reset()
        return {"observation": obs}

    def step(self, action: str) -> dict:
        assert self._env is not None, "Call reset first"
        assert action in ACTIONS, f"Invalid action '{action}'. Must be one of {ACTIONS}"
        obs, reward, done, info = self._env.step(action)
        result = {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }
        if done:
            result["score"] = self._env.score()
        return result

    def state(self) -> dict:
        if self._env is None:
            return {"task": None, "step": 0, "done": False, "cumulative_reward": 0.0}
        return {
            "task": self._env.task,
            "step": self._env._step_idx,
            "done": self._env._done,
            "cumulative_reward": round(sum(self._env._episode_rewards), 4),
        }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = ClaimWatchServer()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            method = data.get("method")

            if method == "reset":
                result = session.reset(
                    task=int(data.get("task", 1)),
                    seed=int(data.get("seed", 42)),
                )
                await websocket.send_text(json.dumps(result))

            elif method == "step":
                action = data.get("action", "")
                if action not in ACTIONS:
                    await websocket.send_text(json.dumps({
                        "error": f"Invalid action '{action}'. Must be one of {ACTIONS}"
                    }))
                else:
                    result = session.step(action)
                    await websocket.send_text(json.dumps(result))

            elif method == "state":
                await websocket.send_text(json.dumps(session.state()))

            else:
                await websocket.send_text(json.dumps({
                    "error": f"Unknown method '{method}'. Use: reset, step, state"
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "name": "ClaimWatch",
        "version": "1.0.0",
        "description": "Payer-side insurance claims triage RL environment",
        "tasks": [1, 2, 3],
        "actions": ACTIONS,
        "reward_range": [-0.80, 0.50],
        "score_keys": ["routing_accuracy", "reward_score", "composite"],
        "websocket_endpoint": "/ws",
        "web_ui": "/web",
    }


# ---------------------------------------------------------------------------
# Gradio debug UI — mounted at /web
# ---------------------------------------------------------------------------

if os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true":
    import gradio as gr

    _ui_env: ClaimWatchEnv | None = None
    _ui_history: list[dict] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_obs(obs: dict, reward: float | None = None, done: bool = False) -> str:
        if not obs:
            return "No observation — hit Reset to start an episode."
        info = obs.get("episode_info", {})
        lines = [
            f"Claim ID       : {obs.get('claim_id', '')}",
            f"Procedure      : {obs.get('procedure_name', '')} ({obs.get('procedure_code', '')})",
            f"Diagnosis      : {obs.get('diagnosis_name', '')} ({obs.get('diagnosis_code', '')})",
            f"Hospital       : {obs.get('hospital_id', '')}",
            f"Amount         : ${obs.get('claim_amount', 0):.2f}",
            f"Acuity         : {obs.get('patient_acuity', '')}/5",
            f"Policy         : {obs.get('policy_name', '')}",
            f"Covered procs  : {', '.join(obs.get('policy_covered_procedures', []))}",
            f"Always review  : {', '.join(obs.get('policy_always_review', []))}",
            f"Auto-approve ≤ : ${obs.get('policy_auto_approve_threshold', 0):.0f}",
            f"",
            f"Clinical note  :",
            f"{obs.get('clinical_note', '')}",
            f"",
            f"Cumulative reward : {info.get('cumulative_reward', 0.0):.2f}",
            f"Step              : {info.get('step', 0)} / {info.get('total_claims', 0)}",
        ]
        if reward is not None:
            lines.append(f"Last reward       : {reward:+.2f}")
        if done:
            lines.append("")
            lines.append("*** EPISODE COMPLETE — hit Reset to start again ***")
        return "\n".join(lines)

    def _format_history() -> str:
        if not _ui_history:
            return "(no steps yet)"
        return "\n".join(
            f"step {i+1:3d}: {h['action']:20s}  reward={h['reward']:+.2f}  gt={h['ground_truth']}"
            for i, h in enumerate(_ui_history)
        )

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def ui_reset(task: str, seed: float):
        global _ui_env, _ui_history
        _ui_env = ClaimWatchEnv(task=int(task), seed=int(seed))
        obs = _ui_env.reset()
        _ui_history = []
        n = len(_ui_env._queue)
        status = f"Episode started — {n} claims in queue  (task={task}, seed={int(seed)})"
        return _format_obs(obs), _format_history(), status

    def ui_step(action: str):
        global _ui_env, _ui_history

        if _ui_env is None:
            return "No active episode. Hit Reset first.", "(no steps yet)", "Idle"

        if _ui_env._done:
            score = _ui_env.score()
            status = (
                f"Episode already finished — "
                f"routing_accuracy={score['routing_accuracy']:.3f}  "
                f"composite={score['composite']:.3f}  "
                f"total_reward={score['total_reward']:.2f}"
            )
            return _format_obs({}, done=True), _format_history(), status

        obs, reward, done, info = _ui_env.step(action)

        _ui_history.append({
            "action": action,
            "reward": reward,
            "ground_truth": info.get("ground_truth", "?"),
        })

        if done:
            score = _ui_env.score()
            status = (
                f"DONE — "
                f"routing_accuracy={score['routing_accuracy']:.3f}  "
                f"composite={score['composite']:.3f}  "
                f"total_reward={score['total_reward']:.2f}"
            )
        else:
            status = (
                f"step {_ui_env._step_idx} / {len(_ui_env._queue)}   "
                f"cumulative reward: {sum(_ui_env._episode_rewards):.2f}"
            )

        return _format_obs(obs, reward=reward, done=done), _format_history(), status

    # ------------------------------------------------------------------
    # Gradio layout
    # ------------------------------------------------------------------

    with gr.Blocks(title="ClaimWatch Debug UI", theme=gr.themes.Soft()) as _demo:

        gr.Markdown(
            """
            ## ClaimWatch — interactive episode debugger
            Step through claims manually to verify environment logic and reward signals.
            """
        )

        with gr.Row():
            task_input = gr.Dropdown(
                choices=["1", "2", "3"],
                value="1",
                label="Task",
                info="1=easy (50 claims)  2=medium (300)  3=hard (1000 + fraud burst)",
            )
            seed_input = gr.Number(value=42, label="Seed", precision=0)
            reset_btn = gr.Button("Reset episode", variant="primary")

        status_box = gr.Textbox(label="Status", lines=1, interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                obs_box = gr.Textbox(
                    label="Current claim observation",
                    lines=18,
                    interactive=False,
                )
            with gr.Column(scale=2):
                history_box = gr.Textbox(
                    label="Step history  (action | reward | ground truth)",
                    lines=18,
                    interactive=False,
                )

        with gr.Row():
            action_input = gr.Dropdown(
                choices=ACTIONS,
                value="auto_approve",
                label="Action",
            )
            step_btn = gr.Button("Step →", variant="secondary")

        gr.Markdown(
            """
            **Actions:** `auto_approve` · `clinical_review` · `md_review` · `request_info` · `deny`

            **Key rewards:** urgent escalation `+0.5` · routine auto-approve `+0.3` ·
            missed urgent `-0.8` · false denial `-0.6` · correct fraud deny `+0.4`
            """
        )

        reset_btn.click(
            fn=ui_reset,
            inputs=[task_input, seed_input],
            outputs=[obs_box, history_box, status_box],
        )
        step_btn.click(
            fn=ui_step,
            inputs=[action_input],
            outputs=[obs_box, history_box, status_box],
        )

    gr.mount_gradio_app(app, _demo, path="/web")

    @app.get("/")
    async def root_redirect():
        return RedirectResponse(url="/web")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    workers = int(os.getenv("WORKERS", "1"))
    print(f"ClaimWatch server — workers={workers}")
    print("  WebSocket : ws://localhost:8000/ws")
    print("  Health    : http://localhost:8000/health")
    print("  Info      : http://localhost:8000/info")
    if os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true":
        print("  Debug UI  : http://localhost:8000/web")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=workers)
