"""Stage 8 — Agent tools.

The agent reaches core/ and verify/ only through these functions. That is
the point: a bounded blast radius. If a prompt injection in a user file
gets the model to do something, what it can reach is still these six
read-only operations, not the filesystem and not arbitrary code.

Every tool is a pure function over data Graft already has. None of them
executes user code, none of them writes files, and none of them can change
a verdict. If a tool could set PRODUCTION_READY, the layering would be a
lie.

The agent may call these to gather context and to propose things. The
numbers it narrates over still come from verify/ and bench/, not from the
model.
"""

import ast
from typing import Any

from graft.core import coverage as coverage_module
from graft.verify import cases as case_module


def inspect_changes(source: str) -> dict:
    """Run the converter and return the change log as structured data.

    Takes source text rather than a MigrationResult object: a model can
    only pass JSON scalars, so an object parameter would either fail schema
    generation or arrive as a useless empty dict. The tool does the
    conversion itself so the change log is always the real one.

    Returns counts and entries rather than prose so the model cannot
    accidentally paraphrase a line number.
    """
    from graft.core import converter

    result = converter.convert(source)
    return {
        "backend": result.backend,
        "total": len(result.changes),
        "verdict": str(result.coverage.verdict),
        "entries": [
            {
                "symbol": change.get("symbol", "?"),
                "line": change.get("line"),
                "reason": change.get("reason") or change.get("note") or "",
            }
            for change in result.changes
        ],
    }


def describe_coverage(source: str) -> dict:
    """Run the real scanner and return its findings.

    The agent gets the same severities and verdict the report uses. It
    cannot soften them, because it never computes them.
    """
    assessment = coverage_module.assess(source)
    return {
        "verdict": assessment.verdict,
        "summary": assessment.summary(),
        "counts": assessment.counts,
        "blockers": [finding.to_dict() for finding in assessment.blockers],
        "reviews": [finding.to_dict() for finding in assessment.reviews],
        "infos": [finding.to_dict() for finding in assessment.infos],
    }


def known_hip_equivalents(source: str) -> list[dict]:
    """Which CUDA symbols in this file have a mapping in Graft's table.

    Read from hipify_backend.CUDA_TO_HIP rather than from a curated list
    maintained separately, so this cannot drift from what conversion
    actually does. A symbol absent from the table is reported as unknown,
    which is the honest answer and the one that stops the model inventing
    an equivalent for something with none.
    """
    from graft.core import hipify_backend

    found: list[dict] = []
    for cuda, hip in hipify_backend.CUDA_TO_HIP.items():
        pattern = f"\\b{re_escape(cuda)}\\b"
        line = _first_line(source, pattern)
        if line is None:
            continue
        found.append({"cuda": cuda, "hip": hip, "line": line})

    present = {entry["cuda"] for entry in found}
    for finding in coverage_module.scan(source):
        # Report the construct as it appears in the source, not the rule's
        # human label. "warp shuffle intrinsic" is prose; the model needs
        # __shfl_xor_sync so it can search for it and write a replacement.
        matched = finding.matched_text or finding.symbol
        if any(matched in cuda for cuda in present):
            continue
        found.append(
            {
                "cuda": matched,
                "hip": None,
                "line": finding.line,
                "category": finding.category,
                "severity": str(finding.severity),
                "note": finding.reason,
            }
        )
    return sorted(found, key=lambda entry: (entry["line"] or 0, entry["cuda"]))


def re_escape(text: str) -> str:
    import re

    return re.escape(text)


def _first_line(source: str, pattern: str) -> int | None:
    import re

    match = re.search(pattern, source)
    return source.count("\n", 0, match.start()) + 1 if match else None


def existing_case_coverage() -> dict:
    """What the deterministic harness already tests.

    Returned so the prompt can tell the model not to propose cases that
    exist, which is the most common failure mode of a case-generating
    prompt.
    """
    distributions = sorted(case_module.DISTRIBUTIONS)
    shapes = [tuple(shape) for shape in case_module.SHAPES]
    generated = case_module.generate()
    return {
        "case_count": len(generated),
        "distributions": distributions,
        "shapes": [list(shape) for shape in shapes],
        "dtype": str(generated[0]["input"].dtype) if generated else None,
    }


def suggest_test_shapes(operation: str) -> list[dict]:
    """Alignment and vector-width candidates not already in SHAPES.

    These are the shapes where a hand-ported kernel most often breaks on
    AMD: vector loads assume 16-byte alignment, and odd row counts change
    the tail-handling path entirely. Proposed deterministically, so the
    candidate list does not depend on a model call.
    """
    covered = {tuple(shape) for shape in case_module.SHAPES}
    candidates = [
        ((3, 17), "unaligned row length; defeats vectorised loads"),
        ((2, 4095), "one less than a page boundary"),
        ((7, 8193), "large odd width, stresses the tail path"),
        ((1, 65537), "very wide single row"),
        ((129, 129), "odd square, neither dimension vectorisable"),
    ]
    return [
        {"shape": list(shape), "why": why, "operation": operation}
        for shape, why in candidates
        if shape not in covered
    ]


def python_entry_points(source: str) -> dict:
    """What the ported file exposes, read from its AST.

    Used to tell the model which function it is allowed to talk about.
    Parsing rather than regexing, because a regex cannot tell a function
    named in a comment from one that exists.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "functions": []}

    functions = [
        {"name": node.name, "line": node.lineno}
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        assignments[target.id] = "<not a literal>"

    return {
        "functions": functions,
        "operation": assignments.get("OPERATION"),
        "imports": sorted(
            {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
        ),
    }


class ToolBudgetExceeded(RuntimeError):
    """Raised when a request exceeds its tool-call allowance.

    A plain exception on purpose: the agent treats a failing tool as a tool
    result and stops calling, and agent._run converts anything raised into
    report text rather than letting it escape.
    """


class _Budget:
    """Counts tool calls so one runaway request cannot loop forever.

    Agno's Agent has no tool-call cap of its own, so the bound is enforced
    here instead. Counted per build_tools() call, which is once per agent
    construction, so the limit applies per narrative request.
    """

    def __init__(self, limit: int | None):
        self.limit = limit
        self.spent = 0

    def spend(self, name: str) -> None:
        if self.limit is None:
            return
        self.spent += 1
        if self.spent > self.limit:
            raise ToolBudgetExceeded(
                f"tool call limit reached ({self.limit}); refusing {name}"
            )


def _with_budget(fn, budget: _Budget):
    """Wrap a tool so it consumes from the budget.

    The signature is copied explicitly because Agno builds the tool schema
    by inspecting parameters, and a bare *args/**kwargs wrapper would
    produce an empty-args schema.
    """
    import functools
    import inspect

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        budget.spend(fn.__name__)
        return fn(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(fn)
    return wrapper


# Agno tool wrappers. Each returns JSON-serialisable data and nothing
# else. Named explicitly because Agno derives the tool schema from the
# name and docstring, so the wording here is part of the contract with
# the model.

TOOL_FUNCTIONS = (
    inspect_changes,
    describe_coverage,
    known_hip_equivalents,
    existing_case_coverage,
    suggest_test_shapes,
    python_entry_points,
)

TOOLS: list[Any] = []


def _register(max_calls: int | None = None) -> None:
    from agno.tools.function import Function as AgnoFunction

    budget = _Budget(max_calls)
    TOOLS.extend(
        AgnoFunction.from_callable(_with_budget(fn, budget)) for fn in TOOL_FUNCTIONS
    )


def build_tools(max_calls: int | None = None) -> list[Any]:
    """Tool list for the agent. Empty if agno is unavailable.

    ``max_calls`` caps how many tools one request may invoke. It is
    ignored on the second and later calls, because Agno registers a tool
    once and Graft builds one agent per request.

    Returning an empty list rather than raising keeps the agent optional:
    a missing optional dependency must not take down the product.
    """
    if TOOLS:
        return list(TOOLS)
    try:
        _register(max_calls)
    except ImportError:
        return []
    return list(TOOLS)
