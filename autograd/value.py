class Value:
    """Класс отражающий одно значение тензона, ожидает в качестве типа self некий класс, поддерживающий базовые операции"""

    def __init__(self, data):
        self.data = data
        self.grad: int = 0

        self._backward = lambda: None

    def copy(self) -> Value:
        # TODO: add overloaded init with complete deep copy
        return Value(self.data)