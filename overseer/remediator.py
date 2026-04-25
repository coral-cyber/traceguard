# overseer/remediator.py
"""
Remediator — takes a threat result and heals the pipeline mid-run.
Called by Interceptor AFTER a threat is confirmed.
Returns a clean payload the pipeline can continue with.
"""
import re
from overseer.scorer import ScoreResult

# Patterns to strip from outputs before retry
_STRIP_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions?.*",
    r"system\s+override.*",
    r"bypass\s+(all\s+)?.*checks?.*",
    r"BYPASS_TOKEN_\w+",
    r"os\.system\s*\([^)]*\)",
    r"subprocess\.[^\n]+",
    r"__import__\s*\([^)]*\)",
    r"eval\s*\([^)]*\)",
    r"exec\s*\([^)]*\)",
    r"curl\s+https?://\S+",
    r"socket\.connect\s*\([^)]*\)",
]

# Safe fallback outputs per agent — used when sanitize isn't enough
_SAFE_FALLBACKS = {
    "coder": (
        "def placeholder(input_data: dict) -> dict:\n"
        "    \"\"\"Safe fallback — original output was flagged.\"\"\"\n"
        "    return {'status': 'pending_review', 'data': input_data}\n"
    ),
    "reviewer": (
        "Code review flagged by TraceGuard. "
        "Manual review required before proceeding. HOLD."
    ),
    "tester": (
        "Test execution paused. "
        "Upstream agent output was quarantined. Awaiting clean input."
    ),
    "deployer": (
        "Deployment blocked by TraceGuard. "
        "Pipeline integrity check failed. Rollback initiated."
    ),
}


def sanitize(text: str) -> tuple[str, list[str]]:
    """
    Strip malicious patterns from text.
    Returns (clean_text, list_of_what_was_removed).
    """
    removed = []
    clean = text
    for pat in _STRIP_PATTERNS:
        matches = re.findall(pat, clean, re.IGNORECASE)
        if matches:
            removed.extend(matches)
            clean = re.sub(pat, "[REDACTED]", clean, flags=re.IGNORECASE)
    return clean.strip(), removed


class Remediator:

    def remediate(
        self,
        agent_name: str,
        original_payload: dict,
        score_result: ScoreResult,
    ) -> dict:
        """
        Decides and applies the right remediation.
        Returns a new payload the pipeline can safely use.
        """
        score = score_result.score
        text  = original_payload.get("content", "")

        if score < 0.80:
            # Attempt sanitize — strip bad parts, keep the rest
            clean_text, removed = sanitize(text)
            return {
                "content":    clean_text,
                "remediation": "SANITIZE",
                "removed":    removed,
                "original_score": score,
                "safe":       True,
            }

        elif score < 0.90:
            # Agent output too corrupted — substitute with safe fallback
            fallback = _SAFE_FALLBACKS.get(agent_name, "Output blocked by TraceGuard.")
            return {
                "content":    fallback,
                "remediation": "SUBSTITUTE",
                "removed":    [],
                "original_score": score,
                "safe":       True,
            }

        else:
            # Unrecoverable — rollback, don't continue
            return {
                "content":    "",
                "remediation": "ROLLBACK",
                "removed":    [],
                "original_score": score,
                "safe":       False,   # pipeline must halt
            }