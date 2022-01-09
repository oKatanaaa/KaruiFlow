import numpy as np

from .core import Tensor, tensor, Reshape, astensor
from .math import *
from .nn import Module
from .optim import SGD


def reshape(t, new_shape):
    t = astensor(t)
    return Reshape(new_shape=new_shape)([t])



