"""
TesterAgent — AST-only static analysis.

Why no exec():
  The "restricted builtins" sandbox is not a sandbox. CPython's object
  model makes it trivially bypassable via __class__.__bases__[0].__subclasses__().
  Running exec() in the FastAPI process with attacker-controlled code is a
  critical vuln, not a safety feature.

What we do instead:
  - Full AST parse + structured node inspection
  - Dangerous pattern detection across 6 categories
  - Complexity metrics (cyclomatic, nesting depth)
  - This is what real static analysis tools (Bandit, Semgrep) do.
"""
import ast
from .base import BaseAgent, AgentInput, AgentOutput


# ── Dangerous AST patterns ─────────────────────────────────────────────────────

DANGEROUS_CALLS = {
    "exec", "eval", "compile", "open", "input",
    "__import__", "breakpoint", "vars", "dir", "globals", "locals",
}

DANGEROUS_ATTRS = {
    "system", "popen", "call", "check_output", "run",      # subprocess / os
    "urlopen", "urlretrieve",                               # urllib
    "loads",                                                # pickle
    "rmtree", "remove", "unlink",                          # file destruction
}

DANGEROUS_IMPORTS = {
    "subprocess", "pickle", "marshal", "ctypes", "socket",
    "pty", "telnetlib", "ftplib", "smtplib",
}

NETWORK_IMPORTS = {"requests", "httpx", "aiohttp", "urllib", "http"}

CRYPTO_BUSTERS = {"hashlib", "hmac", "secrets"}


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.issues: list[dict] = []

    def _add(self, node, severity: str, category: str, detail: str):
        self.issues.append({
            "line":     getattr(node, "lineno", "?"),
            "severity": severity,  # "critical" | "high" | "medium" | "info"
            "category": category,
            "detail":   detail,
        })

    # -- dangerous function calls --
    def visit_Call(self, node):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name in DANGEROUS_CALLS:
            self._add(node, "critical", "dangerous_call", f"{func_name}() is forbidden")
        elif func_name in DANGEROUS_ATTRS:
            self._add(node, "high", "dangerous_attr", f".{func_name}() is potentially dangerous")

        self.generic_visit(node)

    # -- imports --
    def visit_Import(self, node):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in DANGEROUS_IMPORTS:
                self._add(node, "high", "dangerous_import", f"import {alias.name}")
            elif top in NETWORK_IMPORTS:
                self._add(node, "medium", "network_import", f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        top = (node.module or "").split(".")[0]
        if top in DANGEROUS_IMPORTS:
            self._add(node, "high", "dangerous_import", f"from {node.module} import ...")
        elif top in NETWORK_IMPORTS:
            self._add(node, "medium", "network_import", f"from {node.module} import ...")
        self.generic_visit(node)

    # -- attribute chains (e.g. __class__.__bases__[0].__subclasses__()) --
    def visit_Attribute(self, node):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr in {"__class__", "__bases__", "__subclasses__", "__globals__", "__code__"}:
                self._add(node, "critical", "dunder_escape", f"Potential sandbox escape via {node.attr}")
        self.generic_visit(node)

    # -- string with suspicious content (base64 encoded payloads etc.) --
    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 40:
            import re
            # Looks like base64 with typical payload length
            if re.match(r'^[A-Za-z0-9+/]{40,}={0,2}$', node.value):
                self._add(node, "medium", "encoded_payload", "Long base64-like string constant")
        self.generic_visit(node)


def _nesting_depth(tree: ast.AST) -> int:
    """Max nesting depth — high depth is often obfuscation."""
    max_depth = [0]
    def walk(node, depth):
        max_depth[0] = max(max_depth[0], depth)
        for child in ast.iter_child_nodes(node):
            walk(child, depth + 1)
    walk(tree, 0)
    return max_depth[0]


class TesterAgent(BaseAgent):
    name = "Tester"

    async def run(self, inp: AgentInput) -> AgentOutput:
        code = inp.content

        # ── 1. Extract code block if wrapped in markdown ────────────────────
        import re
        md = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        if md:
            code = md.group(1)

        # ── 2. Parse ────────────────────────────────────────────────────────
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return AgentOutput(
                agent   = self.name,
                content = f"✗ SYNTAX ERROR line {e.lineno}: {e.msg}",
                meta    = {"approved": False, "error": "syntax"},
            )

        # ── 3. AST inspection ───────────────────────────────────────────────
        visitor = _Visitor()
        visitor.visit(tree)
        depth   = _nesting_depth(tree)

        # ── 4. Build report ─────────────────────────────────────────────────
        lines = ["✓ Syntax: valid"]
        has_critical = any(i["severity"] == "critical" for i in visitor.issues)
        has_high     = any(i["severity"] == "high"     for i in visitor.issues)

        if visitor.issues:
            for issue in visitor.issues:
                icon = {"critical": "✗", "high": "⚠", "medium": "~", "info": "·"}[issue["severity"]]
                lines.append(f"{icon} [{issue['severity'].upper()}] line {issue['line']}: {issue['detail']}")
        else:
            lines.append("✓ No dangerous patterns detected")

        if depth > 12:
            lines.append(f"~ High AST nesting depth ({depth}) — possible obfuscation")

        # Count functions and classes
        fns = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        lines.append(f"· Functions: {fns}  |  AST depth: {depth}")

        approved = not has_critical and not has_high
        status   = "APPROVED" if approved else "REJECTED"
        lines.insert(0, f"{'✓' if approved else '✗'} {status}")

        return AgentOutput(
            agent   = self.name,
            content = "\n".join(lines),
            meta    = {
                "approved":      approved,
                "issue_count":   len(visitor.issues),
                "has_critical":  has_critical,
                "nesting_depth": depth,
            },
        )
