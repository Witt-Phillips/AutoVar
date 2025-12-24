def minimize_variance(
    program,
    base_env = {},
    iterations: int = 1000,
    lr: float = 1e-2,
    lr_decay: float = 0.99,
):
    variance = program.variance(env=base_env, adaptive=True)
    variance_of_variance = variance.variance(env=base_env)

    bias_vars = get_environment_dependencies(variance_of_variance)
    bias_env = {bias_var: 0.5 for bias_var in bias_vars}

    low_bound, high_bound = 0.02, 0.98

    for _ in range(1, iterations + 1):
        _, grads = variance_of_variance.estimate_with_grad(dict(base_env, **bias_env))

        for v in bias_vars:
            g = float(grads.get(v, 0.0))

            new_val = bias_env[v] - lr * g

            bias_env[v] = (new_val if new_val < high_bound else high_bound) if new_val > low_bound else low_bound

        lr *= lr_decay

    return bias_env


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
    
    if isinstance(node, dsl.Env):
        return f"{current_line}Env({node.name})"
    
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

def get_environment_dependencies(node):

    from . import dsl

    if isinstance(node, dsl.Exact):
        return list()

    if isinstance(node, dsl.Env):
        return [node.name]
    
    if isinstance(node, dsl.Sampler):
        return list()
    
    if isinstance(node, dsl.If):
        return list(set(get_environment_dependencies(node.cond) +
                        get_environment_dependencies(node.if_expr) +
                        get_environment_dependencies(node.else_expr)))
    
    if isinstance(node, dsl.Dist):
        return get_environment_dependencies(node.dist)
    
    if isinstance(node, (dsl.Profile, dsl.Inv, dsl.Exp, dsl.Log)):
        return get_environment_dependencies(node.x)
    
    if isinstance(node, (dsl.Add, dsl.Mul)):
        return list(set(get_environment_dependencies(node.x) +
                        get_environment_dependencies(node.y)))
    
    if hasattr(node, '_impl'):
        return get_environment_dependencies(node._impl)

    raise ValueError(f"Unknown node type {node}")


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

def check_deterministic(node) -> bool:
    from . import dsl

    if isinstance(node, (dsl.Exact, dsl.Env)):
        return True
    
    if isinstance(node, (dsl.Sampler, dsl.If)):
        return False
    
    if isinstance(node, dsl.Dist):
        return check_deterministic(node.dist)
    
    if isinstance(node, (dsl.Profile, dsl.Inv, dsl.Exp, dsl.Log)):
        return check_deterministic(node.x)
    
    if isinstance(node, (dsl.Add, dsl.Mul)):
        return check_deterministic(node.x) and check_deterministic(node.y)
    
    if hasattr(node, '_impl'):
        return check_deterministic(node._impl)

    raise ValueError(f"Unknown node type {node}")