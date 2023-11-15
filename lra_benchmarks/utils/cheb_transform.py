from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def discrete_chebyshev_transform(f):
    ## Forward Chebyshev transform at Chebyshev-Gauss-Lobatto points (extreme points, class II)
    N = f.shape[0]-1
    # Create Even function
    fbar = jnp.hstack([f[::-1], f[1:N]])
    # Transform
    fhat = jnp.fft.ifft(fbar)
    fhat = jnp.hstack([fhat[0], 2*fhat[1:N], fhat[N]])
    return jnp.real(fhat)


def inverse_discrete_chebyshev_transform(fhat):
    ## Backward Chebyshev transform at Chebyshev-Gauss-Lobatto points (extreme points, class II)
    N = fhat.shape[0]
    # Sort values out for FFT
    fhat = jnp.hstack([fhat[0], jnp.hstack([fhat[1:N-1], fhat[N-1]*2, fhat[-2:0:-1] ])*0.5 ])
    f = jnp.fft.fft(fhat)
    f = f[N-1::-1]
    return jnp.real(f)