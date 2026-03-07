from n4.tensor import Tensor
from n4.core import Op, Value
from typing import Optional
from n4.numeric import NumericProtocol
from .layer import Layer


class ConvLayer[T: NumericProtocol](Layer[T]):
    """2d Сверточный слой на нейронах"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        backend: type[T],
        stride: int = 1,
        padding: int = 0,
        activation: Optional[type[Op[T]]] = None,
    ) -> None:
        """
        Для сверточной функции

        Аргументы:
            in_channels: кол-во входных каналов (слоев) - на первом, кол-во цветов изображения
            out_channels: кол-во выходных каналов (слоев) - сколько будет выходных слоев
            activation: функция артивации
        """

        super().__init__(backend)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # each filter is a flattened kernel
        filter_len = in_channels * kernel_size * kernel_size

        # Веса - out_channels - кол-во фильтров, filter_len - строка одного фильтра
        # Т.е. обучается out_channels фильтров
        self.weights = Tensor.random_uniform(
            (out_channels, filter_len),
            backend=self._backend,
            low=-1.0,
            high=1.0,
        )
        self.bias = Tensor.zeros((out_channels,), backend=self._backend)

        self.activation = self.resolve_activation(activation)

    def _pad(self, x: Tensor[T]) -> Tensor[T]:
        """
        Заполнить тензор нулями согластно паддингу

        Берется 3хмерный тензор x, и к его размерности по h и w добавляется паддинг с 2х сторон.
        Тензор заполняется нулями, а потом данные из первого тензона переностся в созданный
        """
        if self.padding == 0:
            return x

        c, h, w = x.shape
        ph = h + 2 * self.padding
        pw = w + 2 * self.padding
        padded: Tensor[T] = Tensor.zeros((c, ph, pw), backend=self._backend)

        for ci in range(c):
            for i in range(h):
                for j in range(w):
                    padded[ci, i + self.padding, j + self.padding] = x[ci, i, j]
        return padded

    def _im2col(self, x: Tensor[T]) -> Tensor[T]:
        """
        Превращает 3д тензор (C, H, W) в 2d матрицу

        Фактически - это сбор входов фильтров для свертки
        """
        c, h, w = x.shape

        # Размер ядра свертки
        k = self.kernel_size
        # Шаг
        s = self.stride

        # Сколько раз нужно пройтись ядром по h
        out_h = (h - k) // s + 1

        # Сколько раз нужно пройтись ядром по w
        out_w = (w - k) // s + 1

        # Кол-во патчей - проходов ядра по изображению
        num_patches = out_h * out_w

        # Размер фильтра
        filter_len = c * k * k

        # Кол-во патчей * размер фильтра -> вывод светрки
        patches = Tensor.zeros((num_patches * filter_len,), self._backend)

        p = 0
        # 2 for - проход по всем возможным позициям ядра
        for i in range(0, h - k + 1, s):
            for j in range(0, w - k + 1, s):
                idx = 0
                # Перебор всех элемнтов ядра
                for ci in range(c):
                    for ki in range(k):
                        for kj in range(k):
                            patches[p * filter_len + idx] = x[ci, i + ki, j + kj]
                            idx += 1
                p += 1

        # Тензор Кол-во патчей x размер фильтра
        # каждая строка - копии всех значений из ядра (длина filter_len)
        # таких строк - patches, так как именно столько раз возможно пройтись по x ядром
        return patches.reshape((num_patches, filter_len))

    def forward_pass(self, x: Tensor[T]) -> Tensor[T]:
        """Свертка

        x должен быть размеров (in_channels, H, W)
        """
        if x.ndim != 3:
            raise ValueError(f"ConvLayer expects 3D input (C, H, W), got {x.ndim}D")
        if x.shape[0] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.shape[0]}")

        # Собираем Тензор Кол-во патчей x размер фильтра
        cols = self._im2col(self._pad(x))

        # Перемножение: Кол-во патчей x filter_len
        # на транспонированные веса - обученные фильтры: filter_len x out_channels
        # Получаем тензор размера: (N, out_channels)
        # Т.е. каждый фильтр получает на вход все возможные патчи, давая сумму - все это строка тензора
        # Считается для каждого фильтра (т.е. для out_channels) - столбцы
        out_mat = (cols @ self.weights.Transposed + self.bias).apply_activation(
            self.activation
        )

        # Вернуть размер обратно
        h_in, w_in = x.shape[1], x.shape[2]
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        return out_mat.Transposed.reshape((self.out_channels, h_out, w_out))

    def parameters(self) -> list[Value[T]]:
        return self.weights._data + self.bias._data
