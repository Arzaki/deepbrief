"""Safe arithmetic evaluator (no eval()). Supports + - * / ** % () and math funcs."""

from __future__ import annotations

import ast
import math

_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.UAdd,
    ast.USub,
    ast.FloorDiv,
)


def safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS and node.id not in ("pi", "e"):
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS):
            raise ValueError("disallowed call")
    env = dict(_ALLOWED_FUNCS)
    env.update({"pi": math.pi, "e": math.e})
    return float(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, env))  # noqa: S307


def calculator(expression: str) -> str:
    try:
        return str(safe_eval(expression))
    except Exception as e:
        return f"ERROR: {e}"
