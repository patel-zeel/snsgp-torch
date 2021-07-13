# Imports to create a new kernel
import lab as B
import lab.tensorflow
import lab.torch
import lab.jax
import lab.autograd
import lab.numpy

import tensorflow as tf
from algebra.util import identical
from matrix import Dense
from plum import dispatch

from mlkernels import Kernel, pairwise, elwise
from stheno import GP


class NSEQ(Kernel):
    """Exponentiated quadratic kernel with a length scale.

    Args:
        scale (scalar): Length scale of the kernel.
    """

    def __init__(self, scale1, scale2):
        self.scale1 = scale1[:, None, :]
        self.scale2 = scale2[None, :, :]

    def _compute(self, dist2):
        # This computes the kernel given squared distances. We use `B` to provide a
        # backend-agnostic implementation.
        ls2 = self.scale1**2 + self.scale2**2
        prefix = tf.reduce_prod((2*self.scale1*self.scale2/ls2)**0.5, axis=2)
        return prefix * B.exp(-B.sum(dist2/ls2, axis=2))

    def render(self, formatter):
        # This method determines how the kernel is displayed.
        return "NSEQ"

    @property
    def _stationary(self):
        # This method can be defined to return `True` to indicate that the kernel is
        # stationary. By default, kernels are assumed to not be stationary.
        return False

    @dispatch
    def __eq__(self, other: "NSEQ"):
        # If `other` is also a `EQWithLengthScale`, then this method checks whether
        # `self` and `other` can be treated as identical for the purpose of
        # algebraic simplifications. In this case, `self` and `other` are identical
        # for the purpose of algebraic simplification if `self.scale` and
        # `other.scale` are. We use `algebra.util.identical` to check this condition.
        return identical(self.scale1, other.scale1)

# It remains to implement pairwise and element-wise computation of the kernel.


@pairwise.dispatch
def pairwise(k: NSEQ, x: B.Numeric, y: B.Numeric):
    dist2 = (x[:, None, :]-y[None, :, :])**2
    return Dense(k._compute(dist2))

# @elwise.dispatch
# def elwise(k: EQWithLengthScale, x: B.Numeric, y: B.Numeric):
#     return k._compute(B.ew_dists2(x, y))
