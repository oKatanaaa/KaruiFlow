import numpy as np

from .core import PyTensor


def tensor(data: np.ndarray, requires_grad=False):
    return PyTensor(data, requires_grad)
