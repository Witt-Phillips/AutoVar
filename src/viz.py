""" Utils for visualizing programs. """

from typing import Callable


class NamedCallable:
    """Wraps a callable with a display name for cleaner repr."""
    def __init__(self, fn: Callable, name: str):
        self.fn = fn
        self.name = name
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
    
    def __repr__(self):
        return self.name


def pretty(node, indent: int = 0, compact: bool = False) -> str:
    """
    Pretty-print an IntractableReal AST as an indented tree.
    
    Args:
        node: Any IntractableReal node (Dist, Add, Sampler, etc.)
        indent: Current indentation level (internal use)
        compact: If True, tries to keep simple nodes on one line
    
    Example output:
        Dist(n=10000)
        └─ Add
           ├─ Add
           │  ├─ Sampler(Bernoulli(0.5))
           │  └─ Sampler(Bernoulli(0.5))
           └─ Sampler(Bernoulli(0.5))
    """
    from . import dsl  # Import here to avoid circular imports
    
    prefix = "   " * indent
    connector = "└─ " if indent > 0 else ""
    
    node_type = type(node).__name__
    
    # Leaf nodes
    if isinstance(node, dsl.Exact):
        return f"{prefix}{connector}Exact({node.val})"
    
    if isinstance(node, dsl.Sampler):
        fn_repr = repr(node.f) if hasattr(node, 'f') and node.f else "?"
        # Clean up ugly lambda repr
        if "<function" in fn_repr and "lambda" in fn_repr:
            fn_repr = "λ"
        return f"{prefix}{connector}Sampler({fn_repr})"
    
    # Unary nodes
    if isinstance(node, dsl.Inv):
        child = pretty(node.x, indent + 1).lstrip()
        return f"{prefix}{connector}Inv\n{pretty(node.x, indent + 1)}"
    
    if isinstance(node, dsl.Exp):
        return f"{prefix}{connector}Exp\n{pretty(node.x, indent + 1)}"
    
    if isinstance(node, dsl.Log):
        return f"{prefix}{connector}Log\n{pretty(node.x, indent + 1)}"
    
    if isinstance(node, dsl.Profile):
        return f"{prefix}{connector}Profile\n{pretty(node.x, indent + 1)}"
    
    # Binary nodes
    if isinstance(node, (dsl.Add, dsl.Mul)):
        lines = [f"{prefix}{connector}{node_type}"]
        lines.append(pretty(node.x, indent + 1).replace("└─", "├─", 1) if hasattr(node, 'y') else pretty(node.x, indent + 1))
        lines.append(pretty(node.y, indent + 1))
        return "\n".join(lines)
    
    # Dist wrapper
    if isinstance(node, dsl.Dist):
        return f"{prefix}{connector}Dist(n={node.n})\n{pretty(node.dist, indent + 1)}"
    
    # Sugar nodes with _impl
    if hasattr(node, '_impl'):
        return f"{prefix}{connector}{node_type}\n{pretty(node._impl, indent + 1)}"
    
    # If node
    if isinstance(node, dsl.If):
        lines = [f"{prefix}{connector}If"]
        lines.append(f"{'   ' * (indent + 1)}├─ cond: {pretty(node.cond, 0).strip()}")
        lines.append(f"{'   ' * (indent + 1)}├─ then: {pretty(node.if_expr, 0).strip()}")
        lines.append(f"{'   ' * (indent + 1)}└─ else: {pretty(node.else_expr, 0).strip()}")
        return "\n".join(lines)
    
    # Fallback
    return f"{prefix}{connector}{node}"


def summarize(node) -> dict:
    """
    Returns a summary of the AST structure.
    
    Returns dict with counts of each node type.
    """
    from . import dsl
    
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
        if hasattr(n, 'cond') and isinstance(n, dsl.If):
            walk(n.cond)
            walk(n.if_expr)
            walk(n.else_expr)
    
    walk(node)
    return counts
