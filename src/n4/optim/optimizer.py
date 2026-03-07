from abc import ABC, abstractmethod
from n4.core import Value
from n4.numeric import NumericProtocol
from typing import Optional, Self


class Optimizer[T: NumericProtocol](ABC):
    """Abstract optimizer base class.

    Subclasses must implement `step()` and `zero_grad()`.
    """

    params: list[Value[T]]

    def __init__(self: Self, params: Optional[list[Value[T]]] = None) -> None:
        self.params = [] if params is None else list(params)

    @abstractmethod
    def step(self: Self) -> None: ...

    @abstractmethod
    def zero_grad(self: Self) -> None: ...
