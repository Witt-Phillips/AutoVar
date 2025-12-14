""" Utils for visualizing programs. """
""" This code was generated using AI. """


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
    from . import dsl  # Import here to avoid circular imports
    
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
    if isinstance(node, dsl.Exact):
        return f"{current_line}Exact({node.val})"
    
    if isinstance(node, dsl.Sampler):
        fn_repr = repr(node.f) if hasattr(node, 'f') and node.f else "?"
        # Clean up ugly lambda repr
        if "<function" in fn_repr and "lambda" in fn_repr:
            fn_repr = "λ"
        return f"{current_line}Sampler({fn_repr})"
    
    # If node - check BEFORE _impl since If has _impl
    if isinstance(node, dsl.If):
        lines = [f"{current_line}If"]
        lines.append(pretty(node.cond, indent + 1, False, child_prefix))
        lines.append(pretty(node.if_expr, indent + 1, False, child_prefix))
        lines.append(pretty(node.else_expr, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Dist wrapper
    if isinstance(node, dsl.Dist):
        lines = [f"{current_line}Dist(n={node.n})"]
        lines.append(pretty(node.dist, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Profile wrapper
    if isinstance(node, dsl.Profile):
        lines = [f"{current_line}Profile"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Unary nodes
    if isinstance(node, dsl.Inv):
        lines = [f"{current_line}Inv"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    if isinstance(node, dsl.Exp):
        lines = [f"{current_line}Exp"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    if isinstance(node, dsl.Log):
        lines = [f"{current_line}Log"]
        lines.append(pretty(node.x, indent + 1, True, child_prefix))
        return "\n".join(lines)
    
    # Binary nodes
    if isinstance(node, (dsl.Add, dsl.Mul)):
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
