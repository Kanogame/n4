from typing import Iterable

from n4.core import Value
from n4.numeric import NumericProtocol
from .optimizer import Optimizer


class Adam[T: NumericProtocol](Optimizer[T]):
    """
    Оптимизатор Adam

    Аргументы:
        params: Iterable[Value[T]] -- список параметров для обновления
        lr: float -- скорость обучения (default 1e-3)
        beta1: float -- decay rate for 1st moment (default 0.9)
        beta2: float -- decay rate for 2nd moment (default 0.999)
        eps: float -- small constant for stability (default 1e-8)

    Поведение:
        - Поддерживает экспоненциальные скользящие средние градиентов и их квадратов
        - Использует bias correction для первых шагов
        - Адаптивная скорость обучения для каждого параметра
    """

    lr: float
    beta1: float
    beta2: float
    eps: float
    _backend: type[T]
    _m: dict[int, T]  # First moment (mean of gradients)
    _v: dict[int, T]  # Second moment (mean of squared gradients)
    _t: int  # Timestep counter

    def __init__(
        self,
        params: Iterable[Value[T]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(list(params))

        if len(self.params) == 0:
            raise ValueError("Adam requires at least one parameter")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Ensure all parameters share backend
        self._backend = self.params[0].get_backend()
        for p in self.params:
            if p.get_backend() is not self._backend:
                raise ValueError("All parameters must share the same numeric backend")

        # Initialize moment estimates
        self._m = {}
        self._v = {}
        for i in range(len(self.params)):
            self._m[i] = self._backend.from_float(0.0)
            self._v[i] = self._backend.from_float(0.0)

        self._t = 0

    def step(self) -> None:
        """Apply one Adam optimization step"""
        self._t += 1

        lr_val = self._backend.from_float(self.lr)
        beta1_val = self._backend.from_float(self.beta1)
        beta2_val = self._backend.from_float(self.beta2)
        eps_val = self._backend.from_float(self.eps)
        one_val = self._backend.from_float(1.0)
        t_float_val = self._backend.from_float(float(self._t))

        # Bias correction terms
        bias_correction1 = one_val - beta1_val**t_float_val
        bias_correction2 = one_val - beta2_val**t_float_val

        for i, p in enumerate(self.params):
            g = p.grad

            # Update first moment: m = beta1 * m + (1 - beta1) * grad
            self._m[i] = beta1_val * self._m[i] + (one_val - beta1_val) * g

            # Update second moment: v = beta2 * v + (1 - beta2) * grad^2
            self._v[i] = beta2_val * self._v[i] + (one_val - beta2_val) * (g * g)

            # Bias-corrected estimates
            m_hat = self._m[i] / bias_correction1
            v_hat = self._v[i] / bias_correction2

            # Update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
            # Note: requires sqrt() method on backend
            p.data = p.data - (lr_val * m_hat) / (v_hat.sqrt() + eps_val)
