from function import Function

class Value[T]:
    """Класс отражающий одно значение тензона, ожидает в качестве типа self некий класс, поддерживающий базовые операции"""

    def __init__(self, data: T):
        self.data = data
        self._grad: int = 0

        self._backward = lambda: None
        self.parent_ops: list[Function] = []

    def set_grad(self, new_grad: int):
        self._grad = new_grad
    
    def get_grad(self) -> int:
        return self._grad

    def backward(self):
        topo = []
        visited = set()

        def build(v: Value):
            if v not in visited:
                visited.add(v)
                for f in v.parents:
                    for i in f.inputs:
                        # TODO: remove recursion, remove inplace function
                        build(i)
                topo.append(v)

        build(self)

        self.grad = 1

        for v in reversed(topo):
            for f in v.parents:
                f.backward()

        