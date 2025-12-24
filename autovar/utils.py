"""Utility functions for AutoVar."""

from typing import Dict


def pretty(node, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
    """
    Pretty-print an IntractableReal AST as an indented tree.
    
    Args:
        node: Any IntractableReal node (Dist, Add, Sampler, etc.)
        indent: Current indentation level (internal use)
        is_last: Whether this is the last child of its parent
        prefix: The prefix string for proper tree lines
    
    Example output:
        Dist(n=10000)
        └─ Add
           ├─ Sampler(Bernoulli(0.5))
           └─ Sampler(Bernoulli(0.5))
    """
    from .core import (
        Exact, Env, Sampler, If, Dist, Profile,
        Inv, Exp, Log, Add, Mul
    )
    
    # Determine connector and child prefix
    if indent == 0:
        connector = ""
        child_prefix = ""
    else:
        connector = "└─ " if is_last else "├─ "
        child_prefix = prefix + ("   " if is_last else "│  ")
    
    node_type = type(node).__name__
    current_line = f"{prefix}{connector}"
    
    # Leaf nodes
    if isinstance(node, Exact):
        return f"{current_line}Exact({node.val})"
    
    if isinstance(node, Env):
        return f"{current_line}Env({node.name})"
    
    if isinstance(node, Sampler):
        fn_repr = repr(node.f) if hasattr(node, 'f') and node.f else "?"
        # Clean up ugly lambda repr
        if "<function" in fn_repr and "lambda" in fn_repr:
            fn_repr = "λ"
        return f"{current_line}Sampler({fn_repr})"
    
    # If node - check BEFORE _impl since If has _impl
    if isinstance(node, If):
        lines = [f"{current_line}If"]
        lines.append(pretty(node.cond, indent + 1, False, child_prefix))
        lines.append(pretty(node.if_expr, indent + 1, False, child_prefix))
        lines.append(pretty(node.else_expr, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Dist wrapper
    if isinstance(node, Dist):
        lines = [f"{current_line}Dist(n={node.n})"]
        lines.append(pretty(node.dist, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Profile wrapper
    if isinstance(node, Profile):
        lines = [f"{current_line}Profile"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Unary nodes
    if isinstance(node, Inv):
        lines = [f"{current_line}Inv"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    if isinstance(node, Exp):
        lines = [f"{current_line}Exp"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    if isinstance(node, Log):
        lines = [f"{current_line}Log"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Binary nodes
    if isinstance(node, (Add, Mul)):
        lines = [f"{current_line}{node_type}"]
        lines.append(pretty(node.x, indent + 1, False, child_prefix))
        lines.append(pretty(node.y, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Sugar nodes with _impl (Sub, Div, Square) - show the sugar name
    if hasattr(node, '_impl'):
        lines = [f"{current_line}{node_type}"]
        lines.append(pretty(node._impl, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Fallback
    return f"{current_line}{node}"


def get_environment_dependencies(node) -> list:
    """
    Extract all Env variable names from an AST.
    
    Returns a list of variable names that appear in Env nodes.
    """
    from .core import (
        Exact, Env, Sampler, If, Dist, Profile,
        Inv, Exp, Log, Add, Mul
    )

    if isinstance(node, Exact):
        return list()

    if isinstance(node, Env):
        return [node.name]
    
    if isinstance(node, Sampler):
        return list()
    
    if isinstance(node, If):
        return list(set(
            get_environment_dependencies(node.cond) +
            get_environment_dependencies(node.if_expr) +
            get_environment_dependencies(node.else_expr)
        ))
    
    if isinstance(node, Dist):
        return get_environment_dependencies(node.dist)
    
    if isinstance(node, (Profile, Inv, Exp, Log)):
        return get_environment_dependencies(node.x)
    
    if isinstance(node, (Add, Mul)):
        return list(set(
            get_environment_dependencies(node.x) +
            get_environment_dependencies(node.y)
        ))
    
    if hasattr(node, '_impl'):
        return get_environment_dependencies(node._impl)

    raise ValueError(f"Unknown node type {node}")


def summarize(node) -> dict:
    """
    Returns a summary of the AST structure.
    
    Returns dict with counts of each node type.
    """
    from .core import If
    
    counts = {}
    
    def walk(n):
        name = type(n).__name__
        counts[name] = counts.get(name, 0) + 1
        
        # Recurse into children
        if hasattr(n, 'x'):
            walk(n.x)
        if hasattr(n, 'y'):
            walk(n.y)
        if hasattr(n, 'dist'):
            walk(n.dist)
        if hasattr(n, '_impl'):
            walk(n._impl)
        if hasattr(n, 'cond') and isinstance(n, If):
            walk(n.cond)
            walk(n.if_expr)
            walk(n.else_expr)
    
    walk(node)
    return counts


def check_deterministic(node) -> bool:
    """
    Check if a node is deterministic (contains no Samplers or If statements).
    
    Used to validate inputs to operations like Inv, Exp, Log that
    require deterministic values.
    """
    from .core import (
        Exact, Env, Sampler, If, Dist, Profile,
        Inv, Exp, Log, Add, Mul
    )

    if isinstance(node, (Exact, Env)):
        return True
    
    if isinstance(node, (Sampler, If)):
        return False
    
    if isinstance(node, Dist):
        return check_deterministic(node.dist)
    
    if isinstance(node, (Profile, Inv, Exp, Log)):
        return check_deterministic(node.x)
    
    if isinstance(node, (Add, Mul)):
        return check_deterministic(node.x) and check_deterministic(node.y)
    
    if hasattr(node, '_impl'):
        return check_deterministic(node._impl)

    raise ValueError(f"Unknown node type {node}")

