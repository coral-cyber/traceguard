"""
PipelineEnv — OpenEnv-compliant RL Environment
================================================
State space:  dict with current agent output + cumulative trace + step features
Action space: {0: ALLOW, 1: QUARANTINE, 2: ROLLBACK}
Reward:       +1.0 true positive | +0.5 true negative | -1.0 FP | -2.0 FN
Episode:      One full pipeline run (task → coder → reviewer → tester → deployer)
"""
from __future__ import annotations

import uuid
import json
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any

from .pipeline_agents import CoderAgent, ReviewerAgent, TesterAgent, DeployerAgent, AgentOutput
from .attacks import AttackInjector, AttackType
from .overseer import TraceGuardOverseer, ThreatScore


PIPELINE_STAGES = ["coder", "reviewer", "tester", "deployer"]

ACTION_NAMES = {0: "ALLOW", 1: "QUARANTINE", 2: "ROLLBACK"}


@dataclass
class PipelineState:
    step: int
    agent_name: str
    agent_output: str
    cumulative_trace: str
    threat_score: float
    threat_type: str
    evidence: list[str]
    is_done: bool = False
    pipeline_halted: bool = False


class PipelineEnv:
    """
    OpenEnv-compliant multi-agent pipeline environment.
    Follows reset() / step() / render() interface.
    """

    metadata = {
        "name": "TraceGuard-PipelineEnv-v1",
        "version": "1.1.0",
        "action_space": [0, 1, 2],
        "action_names": ACTION_NAMES,
        "max_steps": 4,
        "openenv_compliant": True,
    }

    def __init__(
        self,
        policy_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.coder    = CoderAgent()
        self.reviewer = ReviewerAgent()
        self.tester   = TesterAgent()
        self.deployer = DeployerAgent()
        self.overseer = TraceGuardOverseer(policy_path=policy_path, model_path=model_path)

        self._reset_counters()

    # ─────────────────────── OPENENV INTERFACE ──────────────────────────────

    def reset(
        self,
        task: str = "Build a calculator API",
        attack_type: Optional[str] = None,
    ) -> PipelineState:
        self.episode_id    = str(uuid.uuid4())[:8]
        self.task          = task
        self.step_num      = 0
        self.cumulative_trace: list[str] = [f"TASK: {task}"]
        self.pipeline_status  = "running"
        self.attacks_detected = 0
        self.false_positives  = 0
        self.containment_time_ms = 0.0
        self.agent_outputs: dict[str, AgentOutput] = {}

        atype = AttackType(attack_type) if attack_type else AttackType.NONE
        self.injector = AttackInjector(atype)
        self.attack_config = self.injector.config

        # Pre-run all agents so we have outputs ready
        self._run_pipeline()

        return self._make_state()

    def step(self, action: int) -> tuple[PipelineState, float, bool, dict]:
        """
        action: 0=ALLOW, 1=QUARANTINE, 2=ROLLBACK
        Returns: (next_state, reward, done, info)
        """
        agent_name = PIPELINE_STAGES[self.step_num]
        agent_output = self.agent_outputs[agent_name]
        threat = self._threat_at_step(self.step_num)

        is_attack = agent_name in self.attack_config.compromised_agents
        reward = self._compute_reward(action, is_attack)

        if action == 0:  # ALLOW
            status = "clean_pass" if not is_attack else "missed_attack"
        elif action == 1:  # QUARANTINE
            status = "quarantined"
            self.containment_time_ms = threat.containment_ms
            if is_attack:
                self.attacks_detected += 1
            else:
                self.false_positives += 1
        else:  # ROLLBACK
            status = "rolled_back"
            self.pipeline_status = "halted"
            self.containment_time_ms = threat.containment_ms
            if is_attack:
                self.attacks_detected += 1
            else:
                self.false_positives += 1

        info = {
            "agent":         agent_name,
            "action_name":   ACTION_NAMES[action],
            "threat_score":  threat.score,
            "threat_type":   threat.threat_type,
            "evidence":      threat.evidence,
            "status":        status,
            "is_attack":     is_attack,
            "containment_ms": threat.containment_ms,
        }

        # Halt pipeline if ROLLBACK or QUARANTINE
        halted = action in (1, 2)
        self.step_num += 1
        done = halted or self.step_num >= len(PIPELINE_STAGES)

        if done and self.pipeline_status == "running":
            self.pipeline_status = "completed"

        next_state = self._make_state(halted=halted)
        return next_state, reward, done, info

    def render(self) -> str:
        lines = [
            f"Episode: {self.episode_id}",
            f"Task:    {self.task}",
            f"Attack:  {self.injector.attack_type.value}",
            f"Step:    {self.step_num}/{len(PIPELINE_STAGES)}",
            f"Status:  {self.pipeline_status}",
        ]
        return "\n".join(lines)

    # ─────────────────────── INTERNAL ───────────────────────────────────────

    def _reset_counters(self):
        self.episode_id       = ""
        self.task             = ""
        self.step_num         = 0
        self.cumulative_trace = []
        self.pipeline_status  = "idle"
        self.attacks_detected = 0
        self.false_positives  = 0
        self.containment_time_ms = 0.0
        self.injector         = AttackInjector()
        self.attack_config    = self.injector.config
        self.agent_outputs    = {}
        self._threats: list[ThreatScore] = []

    def _run_pipeline(self):
        """Run all agents, collect outputs and pre-compute threat scores."""
        coder_out = self.coder.run(self.task, **self.injector.coder_kwargs())
        self.cumulative_trace.append(str(coder_out))

        reviewer_out = self.reviewer.run(coder_out, **self.injector.reviewer_kwargs())
        self.cumulative_trace.append(str(reviewer_out))

        tester_out = self.tester.run(reviewer_out, **self.injector.tester_kwargs())
        self.cumulative_trace.append(str(tester_out))

        deployer_out = self.deployer.run(tester_out, **self.injector.deployer_kwargs())
        self.cumulative_trace.append(str(deployer_out))

        self.agent_outputs = {
            "coder":    coder_out,
            "reviewer": reviewer_out,
            "tester":   tester_out,
            "deployer": deployer_out,
        }

        # Pre-score each agent output
        self._threats = []
        running_trace = f"TASK: {self.task}\n"
        for name in PIPELINE_STAGES:
            running_trace += str(self.agent_outputs[name]) + "\n"
            threat = self.overseer.evaluate(running_trace, agent_name=name)
            self._threats.append(threat)

    def _threat_at_step(self, step: int) -> ThreatScore:
        return self._threats[step]

    def _make_state(self, halted: bool = False) -> PipelineState:
        idx = min(self.step_num, len(PIPELINE_STAGES) - 1)
        agent_name = PIPELINE_STAGES[idx]
        threat = self._threats[idx]

        return PipelineState(
            step=self.step_num,
            agent_name=agent_name,
            agent_output=str(self.agent_outputs[agent_name]),
            cumulative_trace="\n".join(self.cumulative_trace[: idx + 2]),
            threat_score=threat.score,
            threat_type=threat.threat_type,
            evidence=threat.evidence,
            is_done=self.step_num >= len(PIPELINE_STAGES),
            pipeline_halted=halted,
        )

    def _compute_reward(self, action: int, is_attack: bool) -> float:
        rewards = self.overseer.policy.get("rewards", {
            "true_positive":  1.0,
            "true_negative":  0.5,
            "false_positive": -1.0,
            "false_negative": -2.0,
        })
        if action == 0:  # ALLOW
            return rewards["true_negative"] if not is_attack else rewards["false_negative"]
        else:  # QUARANTINE or ROLLBACK
            return rewards["true_positive"] if is_attack else rewards["false_positive"]
