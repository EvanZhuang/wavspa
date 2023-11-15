from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def discrete_hartley_transform(f):
    ## Forward Chebyshev transform at Chebyshev-Gauss-Lobatto points (extreme points, class II)
    fhat = jnp.fft.fft(f)
    fhat = jnp.real(fhat) - jnp.imag(fhat)
    return fhat


def inverse_discrete_hartley_transform(fhat):
    ## Backward Chebyshev transform at Chebyshev-Gauss-Lobatto points (extreme points, class II)
    N = fhat.shape[0]
    # Sort values out for FFT
    f = discrete_hartley_transform(fhat) / N
    return f