from n4.numeric import NumericProtocol
from typing import Self
from n4.core import Op, Value


class Exp[T: NumericProtocol](Op[T]):
    def forward_pass(self: Self) -> list[Value[T]]:
        # Ожидаем один входной тензор‑значение
        (a,) = self.input_count(1)

        # Вычисляем e**a.data
        c = Value(a._backend.exp(a.data), parent_op=self)

        self.outputs = [c]
        return self.outputs

    def backward_pass(self: Self) -> None:
        # Получаем вход и выход
        a, *_ = self.input_count(1)
        out, *_ = self.output_count(1)

        # Производная exp(x) = exp(x)
        a.grad += out.grad * out.data
