from typing import Self
from abc import ABC, abstractmethod

from n4.tensor import Tensor
from n4.core import Value
from n4.numeric import NumericProtocol


class Loss[T: NumericProtocol](ABC):
    """
    Абстрактный класс фукнций потерь.

    Вызывается с (pred, target) и всегда возращает скалярный Value
    """

    @abstractmethod
    def __call__(self: Self, pred: Tensor[T], target: Tensor[T]) -> Value[T]: ...
