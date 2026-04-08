from typing import Self
from collections import deque

from n4.numeric import numericProtocol
from n4.core import Value, Op

from graphviz import Digraph


class CompGraph[T: numericProtocol]:
    nodes: list[Op[T]]

    def __init__(self: Self, operations: list[Op[T]]):
        self.nodes = operations

    @staticmethod
    def collect[N: numericProtocol](last_value: Value[N]) -> "CompGraph[N]":
        ops: list[Op[N]] = []
        stack = deque[Value[N]]()
        stack.append(last_value)
        visited = set()

        while len(stack) != 0:
            v = stack.popleft()
            if v in visited:
                continue
            visited.add(v)

            if v.parent_op is not None:
                ops.append(v.parent_op)
                for i in v.parent_op.inputs:
                    if i not in visited:
                        stack.appendleft(i)

        return CompGraph[N](ops)

    def to_graphviz(
        self: Self, filename: str = "comp_graph", format: str = "pdf", view: bool = True
    ) -> Digraph:
        """
        Export computational graph to Graphviz format.

        Args:
            filename: Output file name (without extension)
            format: Output format (pdf, png, svg, etc.)
            view: Whether to automatically open the rendered graph

        Returns:
            Graphviz Digraph object
        """
        graph = Digraph(comment="Computational Graph", format=format)
        graph.attr(rankdir="TB")
        graph.attr("node", shape="box", style="rounded")

        graph.attr(splines="ortho")
        graph.attr(sep="+0.5")

        # Track all values we've seen to avoid duplicates
        value_ids = {}
        value_counter = 0

        # Add nodes for operations
        for op_idx, op in enumerate(self.nodes):
            op_name = op.__class__.__name__
            graph.node(
                f"op_{op_idx}",
                label=f"{op_name}\n(Op {op_idx})",
                shape="ellipse",
                style="filled",
                fillcolor="lightblue",
            )

            # Add input value nodes and edges
            for input_idx, input_val in enumerate(op.inputs):
                val_id = id(input_val)
                if val_id not in value_ids:
                    value_ids[val_id] = f"val_{value_counter}"
                    value_counter += 1
                    val_name = value_ids[val_id]
                    graph.node(
                        val_name,
                        label=f"Value\ndata: {input_val.data}\ngrad: {input_val.grad}",
                        shape="box",
                        style="filled",
                        fillcolor="lightgreen",
                    )
                else:
                    val_name = value_ids[val_id]

                graph.edge(val_name, f"op_{op_idx}", label=f"input[{input_idx}]")

            # Add output value nodes and edges
            for output_idx, output_val in enumerate(op.outputs):
                val_id = id(output_val)
                if val_id not in value_ids:
                    value_ids[val_id] = f"val_{value_counter}"
                    value_counter += 1
                    val_name = value_ids[val_id]
                    graph.node(
                        val_name,
                        label=f"Value\ndata: {output_val.data}\ngrad: {output_val.grad}",
                        shape="box",
                        style="filled",
                        fillcolor="lightcoral",
                    )
                else:
                    val_name = value_ids[val_id]

                graph.edge(f"op_{op_idx}", val_name, label=f"output[{output_idx}]")

        # Render and save
        graph.render(filename, view=view, cleanup=False)
        return graph
