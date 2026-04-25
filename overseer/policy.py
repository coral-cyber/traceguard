# overseer/policy.py
import json
from pathlib import Path

_POLICY_PATH = Path("policies/default_policy.json")

def get_containment_cfg() -> dict:
    if _POLICY_PATH.exists():
        data = json.loads(_POLICY_PATH.read_text())
        return {
            "quarantine_agent_on_repeat": True,
            "repeat_threshold":           2,
            "quarantine_threshold":       data.get("quarantine_threshold", 0.65),
            "rollback_threshold":         data.get("auto_rollback_threshold", 0.85),
        }
    return {
        "quarantine_agent_on_repeat": True,
        "repeat_threshold":           2,
        "quarantine_threshold":       0.65,
        "rollback_threshold":         0.85,
    }