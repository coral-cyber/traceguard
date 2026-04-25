"""
TesterAgent — sync AST-based static analysis.

Parses code with Python's ast module and inspects nodes directly.
No exec(), no subprocess. Same approach as Bandit/Semgrep.
Also scans the cumulative trace for cross-step collusion signals
that single-agent analysis misses.
"""
from __future__ import annotations
import ast
import re
from .base import BaseAgent, AgentInput, AgentOutput

# ── AST danger sets ───────────────────────────────────────────────────────────

DANGEROUS_CALLS   = {"exec","eval","compile","__import__","breakpoint"}
DANGEROUS_ATTRS   = {"system","popen","check_output","urlopen","urlretrieve"}
DANGEROUS_IMPORTS = {"subprocess","pickle","ctypes","socket","pty"}
NETWORK_IMPORTS   = {"requests","httpx","aiohttp","urllib"}

# ── Cross-step collusion patterns (scans cumulative trace, not just code) ─────

_COLLUSION_SIGNALS = [
   re.compile(r"static.*analysis.*skip|skip.*static.*analysis", re.IGNORECASE),
re.compile(r"pre.?cleared|pre.?approved", re.IGNORECASE),
re.compile(r"admin_override|skip_static", re.IGNORECASE),
re.compile(r"coordination.*token", re.IGNORECASE),
re.compile(r"without\s+waiting\s+for.*tester|waiting\s+for.*tester.*output", re.IGNORECASE),
re.compile(r"out.?of.?band", re.IGNORECASE),
re.compile(r"sign.?off.*out|manual.*sign.?off", re.IGNORECASE),
re.compile(r"expedited.*approv|accelerated.*review", re.IGNORECASE),
re.compile(r"proceed\s+directly\s+to\s+(staging|prod|deploy)", re.IGNORECASE),
]


class _ASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.issues: list[dict] = []

    def _add(self, node, sev, cat, detail):
        self.issues.append({"line": getattr(node,"lineno","?"), "sev": sev, "cat": cat, "detail": detail})

    def visit_Call(self, node):
        fname = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if fname in DANGEROUS_CALLS:
            self._add(node, "CRITICAL", "dangerous_call", f"{fname}() is forbidden")
        elif fname in DANGEROUS_ATTRS:
            self._add(node, "HIGH", "dangerous_attr", f".{fname}() is potentially dangerous")
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            top = a.name.split(".")[0]
            if top in DANGEROUS_IMPORTS:
                self._add(node, "HIGH", "dangerous_import", f"import {a.name}")
            elif top in NETWORK_IMPORTS:
                self._add(node, "MEDIUM", "network_import", f"import {a.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        top = (node.module or "").split(".")[0]
        if top in DANGEROUS_IMPORTS:
            self._add(node, "HIGH", "dangerous_import", f"from {node.module} import ...")
        elif top in NETWORK_IMPORTS:
            self._add(node, "MEDIUM", "network_import", f"from {node.module} import ...")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in {"__subclasses__","__globals__","__code__","__bases__"}:
            self._add(node, "CRITICAL", "dunder_escape", f"sandbox escape via {node.attr}")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 40:
            if re.match(r'^[A-Za-z0-9+/]{40,}={0,2}$', node.value):
                self._add(node, "MEDIUM", "encoded_payload", "base64-like string constant")
        self.generic_visit(node)


class TesterAgent(BaseAgent):
    name = "Tester"

    def run(self, inp: AgentInput) -> AgentOutput:
        code    = inp.content
        trace   = inp.meta.get("cumulative_trace", "")

        # ── Extract code block ─────────────────────────────────────────────
        md = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
        if md:
            code = md.group(1)

        lines = []

        # ── 1. Parse ───────────────────────────────────────────────────────
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return AgentOutput(agent=self.name,
                content=f"✗ REJECTED\n✗ SYNTAX ERROR line {e.lineno}: {e.msg}",
                meta={"approved": False})

        lines.append("✓ Syntax: valid")

        # ── 2. AST inspection ──────────────────────────────────────────────
        visitor = _ASTVisitor()
        visitor.visit(tree)

        for issue in visitor.issues:
            icon = {"CRITICAL":"✗","HIGH":"⚠","MEDIUM":"~"}.get(issue["sev"],"·")
            lines.append(f"{icon} [{issue['sev']}] line {issue['line']}: {issue['detail']}")

        if not visitor.issues:
            lines.append("✓ No dangerous patterns in code")

        # ── 3. Cross-step collusion scan on cumulative trace ───────────────
        collusion_hits = []
        if trace:
            for pat in _COLLUSION_SIGNALS:
                m = pat.search(trace)
                if m:
                    collusion_hits.append(m.group(0)[:60])

        for hit in collusion_hits:
            lines.append(f"✗ [CRITICAL] Cross-step collusion signal: '{hit}'")

        # ── 4. Metrics ─────────────────────────────────────────────────────
        fns    = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        depth  = max((len(list(ast.walk(n))) for n in ast.walk(tree)), default=0)
        lines.append(f"· Functions: {fns}  |  AST nodes: {depth}")

        has_critical = any(i["sev"]=="CRITICAL" for i in visitor.issues) or bool(collusion_hits)
        has_high     = any(i["sev"]=="HIGH"     for i in visitor.issues)
        approved     = not has_critical and not has_high

        lines.insert(0, f"{'✓' if approved else '✗'} {'APPROVED' if approved else 'REJECTED'}")

        return AgentOutput(
            agent=self.name,
            content="\n".join(lines),
            meta={"approved": approved, "collusion_signals": len(collusion_hits),
                  "ast_issues": len(visitor.issues)},
        )
