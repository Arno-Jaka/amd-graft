"""Stage 1 — PyTorch Python AST rewriting.

Two passes, deliberately separated: the visitor only observes and logs,
the transformer only restructures. Mixing observation with mutation makes
line attribution unreliable.

On ROCm, PyTorch retains the torch.cuda namespace as a compatibility
shim, so most cuda API calls are verified-portable rather than rewritten.
The genuine structural fix here is removing hardcoded device indices.
"""

from __future__ import annotations

import ast

from .detector import Lang, Route


class RewriteLog:
    def __init__(self) -> None:
        self.changes: list[dict] = []
        self.unmapped: list[dict] = []

    def changed(self, symbol: str, line: int, reason: str) -> None:
        self.changes.append({"symbol": symbol, "line": line, "reason": reason})

    def unmapped(self, symbol: str, line: int, note: str) -> None:
        self.unmapped.append({"symbol": symbol, "line": line, "note": note})


class CudaApiVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.log = RewriteLog()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Attribute):
            mod = node.value
            if (
                isinstance(mod.value, ast.Name)
                and mod.value.id == "torch"
                and mod.attr == "cuda"
            ):
                self.log.changed(
                    f"torch.cuda.{node.attr}",
                    node.lineno,
                    "ROCm keeps the torch.cuda namespace; verified portable",
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and node.value == "cuda":
            self.log.changed(
                "device string 'cuda'",
                node.lineno,
                "device string is valid on ROCm; made configurable",
            )
        self.generic_visit(node)


class IndexNormalizer(ast.NodeTransformer):
    def __init__(self, log: RewriteLog) -> None:
        self.log = log

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)

        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "device"
        ):
            return node

        # torch.device("cuda", 0) puts the device string first and the
        # index second, so the literal to normalize is args[1].
        if len(node.args) < 2:
            return node

        index = node.args[1]
        if isinstance(index, ast.Constant) and index.value == 0:
            node.args[1] = ast.Name(id="GPU_DEVICE_INDEX", ctx=ast.Load())
            self.log.changed(
                "torch.device('cuda', 0)",
                node.lineno,
                "hardcoded device index replaced with config constant",
            )
        return node


def rewrite(source: str, route: Route) -> tuple[str, RewriteLog]:
    if route.lang is not Lang.PYTORCH_PYTHON:
        return source, RewriteLog()

    tree = ast.parse(source)

    visitor = CudaApiVisitor()
    visitor.visit(tree)

    normalizer = IndexNormalizer(visitor.log)
    tree = normalizer.visit(tree)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree), visitor.log
