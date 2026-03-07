from typing import Iterable

from n4.core import Value
from n4.numeric import NumericProtocol
from .optimizer import Optimizer


class SGD[T: NumericProtocol](Optimizer[T]):
    """
    Простая стохастическая градиентная оптимизация (SGD).

    Аргументы:
        params: Iterable[Value[T]] -- список/итерация параметров для обновления
        lr: float -- скорость обучения

    Поведение:
        - Метод `step()` обновляет `param.data = param.data - lr * param.grad` для всех параметров
        - Метод `zero_grad()` обнуляет градиенты у всех параметров

    Ограничения: все переданные параметры должны быть одного бекенда (NumericProtocol).
    """

    lr: float
    _backend: type[T]

    def __init__(self, params: Iterable[Value[T]], lr: float = 1e-3) -> None:
        super().__init__(list(params))

        if len(self.params) == 0:
            raise ValueError("SGD requires at least one parameter")

        self.lr = lr

        # Ensure all parameters share backend
        self._backend = self.params[0].get_backend()
        for p in self.params:
            if p.get_backend() is not self._backend:
                raise ValueError("All parameters must share the same numeric backend")

    def step(self) -> None:
        """Apply one optimization step: p.data = p.data - lr * p.grad"""

        lr_val = self._backend.from_float(self.lr)

        for p in self.params:
            p.data = p.data - (p.grad * lr_val)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
