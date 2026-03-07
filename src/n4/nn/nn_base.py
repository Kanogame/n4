from n4.op import NonOp
from abc import ABC, abstractmethod
from typing import Self, Optional
from n4.core import Value, Op
from n4.numeric import NumericProtocol


class NnBase[T: NumericProtocol](ABC):
    """
    Базовый класс для всех элементов нейросети.
    Класс задан абстрактным, так как не имеет смысла сам по себе

    Позволяет обнулять все градиенты
    """

    # Бекенд вычислений
    _backend: type[T]

    # Касательно типов
    # Вот как это делает нормальный человек (Go):
    # - Типы не теряются при компиляции
    # - типы inferred сквозь все уровни
    # - Тип определяет реализацию интерфейса
    # -> Реализация интерфейса очевидна без explicit передачи типа, да и она не нужна
    # Вот как это делает больной на голову шизофреник (Python+MyPy):
    # - Типов не существует по определению
    # - Generic в Stdlib но всем насрать, тип T будет проверен и inferred только MyPy, Python даже читать не будет
    # - Тип пусть и определят реализацию, но он теряется при компиляции, поэтому невозно узнать какой бекенд используется после
    # -> Реализация интерфейса в generic возможна или через dummy (Value, Tensor) или через explicit передачу типа
    # 
    # dummy это совсем кошмар, поэтому берем передачу типа. Как альтернатива - глобальный тип заданный в runtime
    def __init__(self: Self, backend: type[T]):
        self._backend = backend

    def zero_grad(self: Self) -> None:
        for v in self.parameters():
            v.zero_grad()

    @abstractmethod
    def parameters(self: Self) -> list[Value[T]]: ...

    @staticmethod
    def resolve_activation(activation: Optional[type[Op[T]]]) -> type[Op[T]]:
        return NonOp if activation is None else activation
