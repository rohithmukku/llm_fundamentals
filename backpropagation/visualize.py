from graphviz import Digraph


def trace(root):
    """Walk the graph from `root` and collect all nodes and directed edges."""
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root, format='svg', rankdir='LR'):
    """
    Render the computation graph ending at `root`.
    rankdir: 'LR' (left-to-right) or 'TB' (top-to-bottom).
    Returns a graphviz.Digraph; call .render('filename') to write a file,
    or just display it (it shows inline in a Jupyter notebook).
    """
    assert rankdir in ('LR', 'TB')
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

    for n in nodes:
        uid = str(id(n))
        # optional label support: if you later add an `n.label`, it shows up here
        lbl = getattr(n, 'label', '')
        prefix = (lbl + ' | ') if lbl else ''
        dot.node(
            name=uid,
            label="{ %sdata %.4f | grad %.4f }" % (prefix, n.data, n.grad),
            shape='record',
        )
        # if this value was produced by an op, draw a little op-node feeding it
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for child, parent in edges:
        # edge points from a child into the op-node that consumed it
        dot.edge(str(id(child)), str(id(parent)) + parent._op)

    return dot