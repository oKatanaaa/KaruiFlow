import numpy as np

from .core import *


def tensor(data: np.ndarray, requires_grad=False):
    return Tensor(data, requires_grad)
