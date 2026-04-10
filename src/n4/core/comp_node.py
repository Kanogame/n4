from typing import Self, Any
from collections import deque

from n4.numeric import NumericProtocol
from n4.core import Value, Op


class CompGraph[T: NumericProtocol]:
    """
    Вычислительный граф

    Узлы = операции
    Связи = op.inputs/op.outputs = Value
    """

    nodes: list[Op[T]]

    def __init__(self: Self, operations: list[Op[T]]):
        self.nodes = operations

    @staticmethod
    def collect[N: NumericProtocol](last_value: Value[N]) -> "CompGraph[N]":
        """
        Factory метод сбора графа

        Сборка вычислительного графа путем простого bfs
        """

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

    def export_graphviz(self: Self) -> Any:
        from graphviz import Digraph

        """
        Метод сбора graphviz DiGraph

        Формирует граф по логике:
        OP остаются узлами
        Value становятся узлами с другим форматированием

        Возвращает Digraph - но в typehint указан Any, так как это опциональная зависимость 
        """
        graph = Digraph(comment="N4 Computational graph dump")
        graph.attr(rankdir="TB")
        graph.attr("node", shape="box", style="rounded")

        graph.attr(splines="ortho")
        graph.attr(sep="+0.5")

        # Храним все прошедние значения для исключения повторов и циклов
        value_ids = {}

        # Нумеруем узлы для более информативного графа
        value_counter = 0

        def form_value_node(input_val: Value[T]):
            graph.node(
                val_name,
                label=f"Value\ndata: {input_val.data}\ngrad: {input_val.grad}",
                shape="box",
                style="filled",
                fillcolor="lightgreen",
            )

        def form_op_node(op_val: Op[T], op_idx: int):
            op_name = op_val.__class__.__name__
            graph.node(
                f"op_{op_idx}",
                label=f"{op_name}\n(Op {op_idx})",
                shape="ellipse",
                style="filled",
                fillcolor="lightblue",
            )

        for op_idx, op in enumerate(self.nodes):
            # Добавляем операцию
            form_op_node(op, op_idx)

            # добавляем все входы
            for input_idx, input_val in enumerate(op.inputs):
                val_id = id(input_val)
                val_name = value_ids[val_id]
                if val_id not in value_ids:
                    value_ids[val_id] = f"val_{value_counter}"
                    value_counter += 1
                    form_value_node(input_val)

                graph.edge(val_name, f"op_{op_idx}", label=f"input[{input_idx}]")

            # добавляем все выходы
            for output_idx, output_val in enumerate(op.outputs):
                val_id = id(output_val)
                val_name = value_ids[val_id]
                if val_id not in value_ids:
                    value_ids[val_id] = f"val_{value_counter}"
                    value_counter += 1
                    form_value_node(output_val)

                graph.edge(f"op_{op_idx}", val_name, label=f"output[{output_idx}]")

        return graph
