"""ClaimWatch — Gradio UI.

Layout (top to bottom, no scroll):
  Row A: Title (centered) | Score badge (right)
  Row B: Task dropdown | Reset | Run All | Get State
  Row C: Cumulative State summary (visible, updates on Get State)
  Row D: Status pills (step / claims / slots / policy — live)
  Row E: Current Claim (left)  |  Your Decision (right)
  Row F: Env-state metric cards (bottom strip)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd

from env.claim_env import ClaimWatchEnv
from env.models import Action, Observation, Reward, RoutingDecision
from env.policies import PROCEDURE_RULES
from env.tasks import TASKS

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DECISION_LABELS: Dict[str, str] = {
    "auto_approve":    "✅ Auto Approve",
    "clinical_review": "🏥 Clinical Review",
    "md_review":       "👨‍⚕️ MD Review",
    "flag_fraud":      "🚨 Flag Fraud",
    "request_info":    "📋 Request Info",
    "deny":            "❌ Deny",
}
DECISION_VALUES  = list(DECISION_LABELS.keys())
DECISION_DISPLAY = [DECISION_LABELS[v] for v in DECISION_VALUES]

TASK_CHOICES = [
    "Easy — Task 1 (routine triage, 30 claims)",
    "Medium — Task 2 (multi-hospital, 50 claims)",
    "Hard — Task 3 (full complexity, 100 claims)",
]

CSS = """
.gradio-container { max-width:100% !important; padding:0 10px !important; margin:0 !important; }
footer { display:none !important; }
"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _card(label: str, value: str, colour: str, bg: str = "#f8fafc") -> str:
    """Small stat card with colored top border."""
    return (
        f'<div style="display:inline-flex;flex-direction:column;align-items:center;'
        f'background:{bg};border:1px solid #e2e8f0;border-top:3px solid {colour};'
        f'border-radius:8px;padding:5px 16px;margin:3px 4px;min-width:80px;">'
        f'<span style="font-size:0.6rem;color:#64748b;text-transform:uppercase;'
        f'letter-spacing:.06em;font-weight:700;">{label}</span>'
        f'<span style="font-size:0.95rem;font-weight:800;color:{colour};margin-top:2px;">{value}</span>'
        f'</div>'
    )


def _score_badge(cumulative: float) -> str:
    c  = "#16a34a" if cumulative >= 0 else "#dc2626"
    bg = "#f0fdf4" if cumulative >= 0 else "#fef2f2"
    bd = "#86efac" if cumulative >= 0 else "#fca5a5"
    s  = f"+{cumulative:.3f}" if cumulative >= 0 else f"{cumulative:.3f}"
    return (
        f'<div style="color:{c};background:{bg};border:2px solid {bd};border-radius:10px;'
        f'font-size:1.4rem;font-weight:800;text-align:center;padding:8px 20px;'
        f'white-space:nowrap;display:flex;align-items:center;justify-content:center;">'
        f'🏆 {s}</div>'
    )


def _status_pills(state: Dict) -> str:
    obs: Optional[Observation] = state.get("obs")
    if obs is None:
        return '<div style="color:#94a3b8;padding:6px 0;font-style:italic;">No episode loaded.</div>'
    cfg        = TASKS[state["task_id"]]
    total      = cfg.n_claims
    done_count = total - len(obs.queue)
    pct        = int(done_count / total * 100) if total else 0
    policy     = getattr(state["env"], "_policy_version", "v1")
    pol_c      = "#16a34a" if policy == "v1" else "#d97706"

    pills = [
        ("Step",     str(state["step_count"]),                 "#3b82f6"),
        ("Claims",   f"{done_count}/{total} ({pct}%)",         "#6366f1"),
        ("MD Slots", str(obs.md_slots_remaining),              "#0ea5e9"),
        ("Clinical", str(obs.clinical_slots_remaining),        "#8b5cf6"),
        ("Policy",   policy,                                   pol_c),
    ]
    label = ('<p style="font-size:.65rem;font-weight:700;color:#94a3b8;text-transform:uppercase;'
             'letter-spacing:.06em;margin:2px 0 2px 2px;">Live Status</p>')
    return (
        label
        + '<div style="display:flex;flex-wrap:wrap;gap:0;padding:2px 0;">'
        + "".join(_card(l, v, c) for l, v, c in pills)
        + "</div>"
    )


def _claim_html(snap) -> str:
    if snap is None:
        return (
            '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
            'padding:20px;color:#94a3b8;font-style:italic;text-align:center;">'
            'No pending claims — episode complete or not started.</div>'
        )
    docs     = ", ".join(snap.documentation) if snap.documentation else "<em>none</em>"
    sla_val  = snap.sla_tier if isinstance(snap.sla_tier, str) else snap.sla_tier.value
    sla_icon = {"urgent": "🔴", "critical": "🟡", "routine": "🟢"}.get(sla_val, "⚪")
    sla_rem  = getattr(snap, "sla_remaining_hr", "?")
    sla_colors = {"urgent": "#dc2626", "critical": "#d97706", "routine": "#16a34a"}
    sla_bgs    = {"urgent": "#fee2e2", "critical": "#fef9c3", "routine": "#f0fdf4"}
    sla_c  = sla_colors.get(sla_val, "#475569")
    sla_bg = sla_bgs.get(sla_val, "#f8fafc")
    rows = [
        ("Claim ID",
         f"<strong style='color:#1e293b;font-size:1rem'>{snap.claim_id}</strong>", ""),
        ("Procedure",
         f"<span style='background:#dbeafe;color:#1d4ed8;padding:2px 8px;border-radius:5px;"
         f"font-weight:700;font-size:.8rem;margin-right:6px'>{snap.procedure_code}</span>"
         f"<span style='color:#334155'>{snap.procedure_description}</span>", ""),
        ("Diagnosis",
         f"<span style='background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:5px;"
         f"font-weight:700;font-size:.8rem;margin-right:6px'>{snap.diagnosis_code}</span>"
         f"<span style='color:#334155'>{snap.diagnosis_description}</span>", ""),
        ("Billed",
         f"<strong style='color:#0f172a;'>${snap.billed_amount:,.2f}</strong>", ""),
        ("Docs", docs, ""),
        ("SLA",
         f"<span style='background:{sla_bg};color:{sla_c};padding:3px 10px;border-radius:5px;"
         f"font-weight:700;font-size:.85rem'>{sla_icon} {sla_val.upper()}</span>"
         f"&nbsp;<span style='color:#64748b;font-size:.8rem'>{sla_rem} hr remaining</span>",
         sla_bg),
        ("Hospital",      snap.hospital_id, ""),
        ("Patient Age",   str(snap.patient_age), ""),
        ("Days in Queue", str(snap.days_in_queue), ""),
    ]
    trs = ""
    for k, v, row_bg in rows:
        bg_attr = f'background:{row_bg};' if row_bg else ""
        trs += (
            f'<tr style="{bg_attr}">'
            f'<td style="padding:5px 12px 5px 0;color:#64748b;font-size:0.72rem;'
            f'font-weight:700;text-transform:uppercase;letter-spacing:.04em;'
            f'white-space:nowrap;vertical-align:middle;">{k}</td>'
            f'<td style="padding:5px 0;font-size:0.85rem;color:#1e293b;">{v}</td></tr>'
        )
    return (
        '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
        'box-shadow:0 1px 3px rgba(0,0,0,.07);padding:12px 16px;">'
        f'<table style="border-collapse:collapse;width:100%;">{trs}</table></div>'
    )


def _state_summary_html(state: Dict) -> str:
    """Cumulative state summary displayed above the main area."""
    env: ClaimWatchEnv = state["env"]
    s       = env.state()
    history: List[float] = state.get("history", [])

    # ── Use values directly from env.state() so they always match the score badge ──
    cum_r   = s.get("cumulative_reward", 0.0)
    steps   = s.get("step_number", 0)
    decided = s.get("decided_count", 0)
    pending = s.get("pending_count", 0)
    total   = s.get("total_claims", 0)

    pos     = sum(1 for r in history if r > 0)
    neg     = sum(1 for r in history if r < 0)
    neutral = len(history) - pos - neg

    # Accuracy rate (positive decisions / total decisions)
    acc     = pos / len(history) * 100 if history else 0.0

    # Trend: avg of last 5 steps
    recent  = history[-5:] if history else []
    avg5    = sum(recent) / len(recent) if recent else 0.0
    trend_c = "#16a34a" if avg5 >= 0 else "#dc2626"
    trend_s = f"+{avg5:.3f}" if avg5 >= 0 else f"{avg5:.3f}"

    # Progress bar html
    pct      = int(decided / total * 100) if total else 0
    bar_fill = f'<div style="height:100%;width:{pct}%;background:#6366f1;border-radius:4px;transition:width .4s;"></div>'
    bar      = (
        f'<div style="background:#e2e8f0;border-radius:4px;height:8px;'
        f'width:180px;display:inline-block;vertical-align:middle;margin-left:8px;">'
        f'{bar_fill}</div>'
        f'<span style="font-size:.75rem;color:#6366f1;margin-left:6px;">{pct}%</span>'
    )

    cr_c  = "#16a34a" if cum_r >= 0 else "#dc2626"
    cr_s  = f"+{cum_r:.4f}" if cum_r >= 0 else f"{cum_r:.4f}"

    if not history:
        return (
            '<p style="color:#94a3b8;font-style:italic;font-size:.82rem;padding:2px 8px;margin:0;">'
            'No decisions yet — submit a claim decision first.</p>'
        )

    metrics = [
        ("Total Steps",       str(steps),                                                      "#3b82f6"),
        ("Cumulative Reward",  cr_s,                                                           cr_c),
        ("✓ Correct",         str(pos),                                                        "#16a34a"),
        ("✗ Penalised",       str(neg),                                                        "#dc2626"),
        ("→ Neutral",         str(neutral),                                                    "#64748b"),
        ("Accuracy",          f"{acc:.1f}%",                                                   "#6366f1"),
        ("Avg Reward (L5)",   trend_s,                                                         trend_c),
        ("Decided",           str(decided),                                                    "#6366f1"),
        ("Pending",           str(pending),                                                    "#f59e0b"),
    ]

    cards_html = "".join(_card(l, v, c) for l, v, c in metrics)

    label = ('<p style="font-size:.65rem;font-weight:700;color:#94a3b8;text-transform:uppercase;'
             'letter-spacing:.06em;margin:0 0 4px 2px;">Episode Summary</p>')
    return (
        label
        + '<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;'
        'box-shadow:0 1px 3px rgba(0,0,0,.06);padding:10px 16px;">'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<span style="font-size:.7rem;font-weight:700;color:#64748b;text-transform:uppercase;'
        f'letter-spacing:.06em;">Progress</span>{bar}'
        f'</div>'
        f'<div style="margin-top:6px;">{cards_html}</div>'
        f'</div>'
    )


def _env_strip_html(state: Dict) -> str:
    """Bottom environment stat strip (slots, policy, day/hour — things not in state summary)."""
    env: ClaimWatchEnv = state["env"]
    s      = env.state()
    policy = getattr(env, "_policy_version", "v1")
    pol_c  = "#16a34a" if policy == "v1" else "#d97706"

    cards = [
        ("MD Slots",  str(s.get("md_slots_remaining", 0)),         "#0ea5e9"),
        ("Clinical",  str(s.get("clinical_slots_remaining", 0)),   "#8b5cf6"),
        ("Policy",    policy,                                       pol_c),
        ("Day",       str(s.get("current_day", 0)),                "#475569"),
        ("Hour",      str(s.get("current_hour", 8)),               "#475569"),
    ]
    label = ('<p style="font-size:.65rem;font-weight:700;color:#94a3b8;text-transform:uppercase;'
             'letter-spacing:.06em;margin:4px 0 2px 2px;">Environment</p>')
    return (
        label
        + '<div style="display:flex;flex-wrap:wrap;gap:0;padding:2px 0;">'
        + "".join(_card(l, v, c) for l, v, c in cards)
        + "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Baseline heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_decision(snap, md_slots: int, clinical_slots: int) -> str:
    rule = PROCEDURE_RULES.get(snap.procedure_code)
    if rule is None or not rule.covered:
        return "deny"
    present = set(snap.documentation)
    if rule.required_docs and not all(d in present for d in rule.required_docs):
        return "request_info"
    if rule.max_billed > 0 and snap.billed_amount > rule.max_billed * 1.5:
        return "flag_fraud"
    if rule.auto_approve_if_docs_complete:
        return "auto_approve"
    decision = rule.default_routing.value
    if decision == "md_review" and md_slots <= 0:
        decision = "clinical_review"
    if decision == "clinical_review" and clinical_slots <= 0:
        decision = "request_info"
    return decision


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def _task_id_from_choice(choice: str) -> int:
    if "Task 1" in choice:
        return 1
    if "Task 2" in choice:
        return 2
    return 3


def _empty_state() -> Dict:
    return {
        "env":               ClaimWatchEnv(),
        "obs":               None,
        "task_id":           1,
        "step_count":        0,
        "cumulative_reward": 0.0,
        "history":           [],   # per-step reward floats (mirrors env._reward_history)
        "done":              True,
    }


def _first_pending(state: Dict) -> Optional[Any]:
    obs: Optional[Observation] = state.get("obs")
    return obs.queue[0] if (obs and obs.queue) else None


# ─────────────────────────────────────────────────────────────────────────────
# Output contract  — every main handler returns this 7-tuple (OUTPUTS order):
#   0  state
#   1  score_badge   gr.HTML
#   2  status_pills  gr.HTML
#   3  state_summary gr.HTML
#   4  claim_card    gr.HTML
#   5  env_strip     gr.HTML
#   6  btn_submit    gr.Button
# ─────────────────────────────────────────────────────────────────────────────

def _build(state: Dict, snap):
    """6-tuple: state, score_badge, status_pills, claim_card, env_strip, btn_submit."""
    return (
        state,
        _score_badge(state["cumulative_reward"]),
        _status_pills(state),
        _claim_html(snap if not state["done"] else None),
        _env_strip_html(state),
        gr.update(interactive=not state["done"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Action handlers
# ─────────────────────────────────────────────────────────────────────────────

def do_reset(task_choice: str, state: Dict):
    task_id = _task_id_from_choice(task_choice)
    obs = state["env"].reset(task_id=task_id, seed=42)
    state.update({
        "obs": obs, "task_id": task_id, "step_count": 0,
        "cumulative_reward": 0.0, "history": [], "done": False,
    })
    base = _build(state, _first_pending(state))
    # Insert hidden state_summary at position 3 (OUTPUTS_FULL contract)
    return base[:3] + (gr.update(visible=False, value=""),) + base[3:]


def do_step(decision_label: str, rationale: str, state: Dict):
    if state.get("done", True):
        return _build(state, _first_pending(state))

    snap = _first_pending(state)
    if snap is None:
        state["done"] = True
        return _build(state, None)

    decision_val = (
        DECISION_VALUES[DECISION_DISPLAY.index(decision_label)]
        if decision_label in DECISION_DISPLAY else decision_label
    )
    obs, reward, done, info = state["env"].step(Action(
        claim_id=snap.claim_id,
        decision=RoutingDecision(decision_val),
        rationale=rationale or None,
    ))
    state["obs"]               = obs
    state["step_count"]       += 1
    # Keep cumulative_reward in sync with env so badge matches state summary
    state["cumulative_reward"] = state["env"].state()["cumulative_reward"]
    state["done"]              = done
    state["history"].append(reward.total)

    return _build(state, _first_pending(state))


def do_run_all(state: Dict):
    """Run baseline over all remaining pending claims (never resets)."""
    if state.get("done", True):
        return _build(state, _first_pending(state))

    env: ClaimWatchEnv = state["env"]
    while not state["done"]:
        obs: Observation = state["obs"]
        if not obs.queue:
            break
        snap = obs.queue[0]
        dv = _baseline_decision(snap, obs.md_slots_remaining, obs.clinical_slots_remaining)
        obs, reward, done, info = env.step(Action(
            claim_id=snap.claim_id, decision=RoutingDecision(dv)
        ))
        state["obs"]               = obs
        state["step_count"]       += 1
        state["done"]              = done
        state["history"].append(reward.total)

    # Sync cumulative reward from the env (single source of truth)
    state["cumulative_reward"] = env.state()["cumulative_reward"]
    return _build(state, None)


def do_get_state(state: Dict):
    """Show episode summary panel and refresh all other panels."""
    return (
        state,
        _score_badge(state["cumulative_reward"]),
        _status_pills(state),
        gr.update(visible=True, value=_state_summary_html(state)),
        _claim_html(_first_pending(state) if not state["done"] else None),
        _env_strip_html(state),
        gr.update(interactive=not state["done"]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio layout
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="ClaimWatch") as demo:

    state = gr.State(_empty_state())

    # ── Row A: Title (center) + Score (right) ─────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            gr.HTML("")   # left spacer
        with gr.Column(scale=6):
            gr.HTML(
                '<h1 style="text-align:center;margin:6px 0 2px;font-size:1.45rem;'
                'font-weight:800;color:#1e293b;letter-spacing:-.02em;">'
                '🏥 ClaimWatch &mdash; Insurance Claims Triage</h1>'
            )
        with gr.Column(scale=2):
            score_badge = gr.HTML(_score_badge(0.0))

    # ── Row B: Controls ───────────────────────────────────────────────────────
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=TASK_CHOICES, value=TASK_CHOICES[0],
            label="Task / Difficulty", scale=5,
        )
        btn_reset = gr.Button("🔄 Reset",         variant="secondary", scale=1)
        btn_all   = gr.Button("⚡ Run All Claims", variant="primary",   scale=2)
        btn_state = gr.Button("📊 Get State",      variant="secondary", scale=1)

    # ── Row C: Episode summary — hidden until user clicks Get State ─────────────
    state_summary = gr.HTML("", visible=False)

    # ── Row D: Live status pills (label embedded inside HTML) ─────────────────
    status_pills = gr.HTML(_status_pills(_empty_state()))

    # ── Row E: Current Claim (left) | Your Decision (right) ───────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.HTML(
                '<p style="font-weight:700;font-size:.9rem;color:#334155;'
                'margin:0 0 4px;">📋 Current Claim</p>'
            )
            claim_card = gr.HTML(_claim_html(None))

        with gr.Column(scale=5):
            gr.HTML(
                '<p style="font-weight:700;font-size:.9rem;color:#334155;'
                'margin:0 0 4px;">🎯 Your Decision</p>'
            )
            decision_radio = gr.Radio(
                choices=DECISION_DISPLAY,
                value=DECISION_DISPLAY[0],
                label="",
                interactive=True,
            )
            rationale_box = gr.Textbox(
                label="Rationale (optional)",
                placeholder="Explain your reasoning…",
                lines=2,
            )
            btn_submit = gr.Button("Submit Decision ▶", variant="primary", interactive=False)

    # ── Row F: Env detail strip (label embedded inside HTML) ────────────────────
    env_strip = gr.HTML(_env_strip_html(_empty_state()))

    # ── Wire up ───────────────────────────────────────────────────────────────
    # OUTPUTS_BASE: step/run_all do NOT update state_summary (stays hidden until Get State)
    OUTPUTS_BASE = [state, score_badge, status_pills, claim_card, env_strip, btn_submit]
    # OUTPUTS_FULL: reset (hides summary) and get_state (shows summary)
    OUTPUTS_FULL = [state, score_badge, status_pills, state_summary, claim_card, env_strip, btn_submit]

    btn_reset.click( fn=do_reset,     inputs=[task_dropdown, state],                 outputs=OUTPUTS_FULL)
    btn_submit.click(fn=do_step,      inputs=[decision_radio, rationale_box, state], outputs=OUTPUTS_BASE)
    btn_all.click(   fn=do_run_all,   inputs=[state],                                outputs=OUTPUTS_BASE)
    btn_state.click( fn=do_get_state, inputs=[state],                                outputs=OUTPUTS_FULL)

    demo.load(fn=do_reset, inputs=[task_dropdown, state], outputs=OUTPUTS_FULL)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
