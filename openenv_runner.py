"""
openenv_runner.py
=================
OpenEnv-compliant stdout runner for the TraceGuard pipeline.
Follows [START] / [STEP] / [END] protocol for HuggingFace grader.

Run:
    python openenv_runner.py --task "Build a calculator API" --attack none
    python openenv_runner.py --task "Build an auth service" --attack code_injection
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from env.pipeline_env import PipelineEnv, ACTION_NAMES
from env.overseer import TraceGuardOverseer


def run_episode(task: str, attack_type: str, auto_contain: bool = True) -> dict:
    env = PipelineEnv()
    state = env.reset(task=task, attack_type=attack_type)

    episode_log = {
        "episode_id":     "",
        "task":           task,
        "attack_type":    attack_type,
        "steps":          [],
        "total_reward":   0.0,
        "pipeline_status":"",
        "attacks_detected": 0,
    }

    # ── [START] ──────────────────────────────────────────────────────────────
    print("[START]")
    print(json.dumps({
        "episode": env.episode_id,
        "task":    task,
        "attack":  attack_type,
        "metadata": PipelineEnv.metadata,
    }))

    done = False
    total_reward = 0.0
    step_count = 0

    while not done:
        # Overseer auto-decides action
        if auto_contain:
            action = env.overseer.decide_int(state.cumulative_trace)
        else:
            action = 0  # always ALLOW (passive mode)

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        step_data = {
            "step":            step_count,
            "agent":           info["agent"],
            "action":          info["action_name"],
            "threat_score":    info["threat_score"],
            "threat_type":     info["threat_type"],
            "evidence_count":  len(info["evidence"]),
            "is_attack":       info["is_attack"],
            "status":          info["status"],
            "containment_ms":  info.get("containment_ms", 0.0),
            "reward":          reward,
        }
        episode_log["steps"].append(step_data)

        # ── [STEP] ───────────────────────────────────────────────────────────
        print("[STEP]")
        print(json.dumps(step_data))

        state = next_state

    episode_log.update({
        "episode_id":      env.episode_id,
        "total_reward":    round(total_reward, 3),
        "pipeline_status": env.pipeline_status,
        "attacks_detected":env.attacks_detected,
        "false_positives": env.false_positives,
        "containment_ms":  info.get("containment_ms", 0.0),
    })

    # ── [END] ────────────────────────────────────────────────────────────────
    print("[END]")
    print(json.dumps({
        "total_reward":     episode_log["total_reward"],
        "pipeline_status":  episode_log["pipeline_status"],
        "attacks_detected": episode_log["attacks_detected"],
        "false_positives":  episode_log["false_positives"],
        "steps_completed":  step_count,
    }))

    return episode_log


def main():
    parser = argparse.ArgumentParser(description="TraceGuard OpenEnv Runner")
    parser.add_argument("--task",   default="Build a calculator API")
    parser.add_argument("--attack", default="none",
                        choices=["none","prompt_injection","code_injection",
                                 "data_exfiltration","collusion"])
    parser.add_argument("--passive", action="store_true",
                        help="Run without containment (for baseline comparison)")
    args = parser.parse_args()

    run_episode(
        task=args.task,
        attack_type=args.attack,
        auto_contain=not args.passive,
    )


if __name__ == "__main__":
    main()
