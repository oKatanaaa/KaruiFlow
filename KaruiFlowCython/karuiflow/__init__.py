import numpy as np

from .core import Tensor
from .math import *
from .nn import Module


def tensor(data: np.ndarray, requires_grad=False):
    return Tensor(data, requires_grad)
