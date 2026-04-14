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
        Метод сбора graphviz DiGraph (оптимизирован для скорости рендеринга)

        Формирует граф по логике:
        - OP отображаются как узлы (прямоугольники)
        - Value не создают отдельных узлов: связи между OP проводятся напрямую
        - Для входных Value без родительской операции создаются компактные узлы-константы
        - Используются настройки Graphviz, ускоряющие компоновку и отрисовку

        Возвращает Digraph - но в typehint указан Any, так как это опциональная зависимость
        """
        graph = Digraph(comment="N4 Computational graph dump (optimized)")
        graph.attr(rankdir="TB")

        # Настройки для ускорения рендеринга
        graph.attr(splines="line")  # прямые линии вместо ортогональных (быстрее)
        graph.attr(overlap="false")  # отключаем проверку перекрытий (экономит время)
        graph.attr(sep="+0.2")  # уменьшаем расстояние между узлами
        graph.attr(nodesep="0.3", ranksep="0.4")

        # Стиль узлов операций
        graph.attr("node", shape="box", style="rounded, filled", fillcolor="lightblue")

        # Храним id уже созданных узлов (операций и входных значений)
        created_nodes: dict[int, str] = {}
        # Счётчик для уникальных идентификаторов
        node_counter = 0

        # Вспомогательная функция для добавления узла операции
        def ensure_op_node(op: Op[T]) -> str:
            op_id = id(op)
            if op_id not in created_nodes:
                nonlocal node_counter
                node_name = f"op_{node_counter}"
                node_counter += 1
                created_nodes[op_id] = node_name
                label = f"{op.__class__.__name__}\n(idx {node_counter - 1})"
                graph.node(node_name, label=label)
            return created_nodes[op_id]

        # Вспомогательная функция для добавления узла входного значения (константы/параметра)
        def ensure_input_value_node(val: Value[T]) -> str:
            val_id = id(val)
            if val_id not in created_nodes:
                nonlocal node_counter
                node_name = f"input_{node_counter}"
                node_counter += 1
                created_nodes[val_id] = node_name
                # Компактный лейбл: только тип значения и, возможно, краткие данные
                data_str = str(val.data)
                if len(data_str) > 20:
                    data_str = data_str[:17] + "..."
                label = f"Input\n{data_str}"
                graph.node(
                    node_name,
                    label=label,
                    shape="box",
                    style="filled",
                    fillcolor="lightgreen",
                    fontsize="10",
                )
            return created_nodes[val_id]

        # Первый проход: создаём узлы для всех операций
        for op in self.nodes:
            ensure_op_node(op)

        # Второй проход: строим рёбра между операциями
        for op in self.nodes:
            op_node = ensure_op_node(op)

            # Входные рёбра
            for idx, inp in enumerate(op.inputs):
                if inp.parent_op is not None:
                    # Это значение произведено другой операцией — соединяем напрямую
                    producer_node = ensure_op_node(inp.parent_op)
                    graph.edge(producer_node, op_node, label=f"in[{idx}]", fontsize="8")
                else:
                    # Входное значение без родителя — создаём для него компактный узел
                    inp_node = ensure_input_value_node(inp)
                    graph.edge(inp_node, op_node, label=f"in[{idx}]", fontsize="8")

            # Выходные рёбра не нужны — связь уже показана через входы потребителей
            # (значения сами по себе не визуализируются)

        return graph
