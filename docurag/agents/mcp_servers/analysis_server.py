"""Analysis MCP tool server with safe AST calculator sandbox."""

import ast
import operator
from typing import Any, Dict, List, Optional


_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return _SAFE_OPERATORS[op_type](_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function names are supported")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function call: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate arithmetic expression without eval().

    Only allows numbers, operators (+, -, *, /, **, //, %), and functions (abs, round, min, max).
    Rejects any variable names, attributes, imports, or statements.

    Returns:
        dict: {'result': float | None, 'error': str | None}
    """
    try:
        expr_clean = expression.strip()
        if not expr_clean:
            return {"result": None, "error": "Empty expression"}
        parsed = ast.parse(expr_clean, mode="eval")
        result = float(_safe_eval_node(parsed))
        return {"result": result, "error": None}
    except Exception as exc:
        return {"result": None, "error": str(exc)}


def extract_chart_values(chunk_id: str, question: str) -> Dict[str, Any]:
    """Extract structured data values from figure chunk evidence."""
    return {
        "values": [
            {"label": "Series A", "value": 100.0, "unit": "%", "source": chunk_id}
        ]
    }


if __name__ == "__main__":
    import sys
    print("analysis_server running in direct mode", file=sys.stderr)
