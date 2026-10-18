"""Stage 8 — Agent context helpers.

These pure functions are the only way the narrative layer reaches core/ and
verify/. That is the point: a bounded blast radius. Even if a prompt
injection in a user file got the model to request context, what it can reach
is still these read-only computations, not the filesystem and not arbitrary
code.

Every helper is a pure function over data Graft already has. None of them
executes user code, none of them writes files, and none of them can change a
verdict. If one could set PRODUCTION_READY, the layering would be a lie.

The numbers the model narrates over still come from verify/ and bench/, not
from the model.
"""

import ast

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


TOOL_FUNCTIONS = (
    inspect_changes,
    describe_coverage,
    known_hip_equivalents,
    existing_case_coverage,
    suggest_test_shapes,
    python_entry_points,
)
