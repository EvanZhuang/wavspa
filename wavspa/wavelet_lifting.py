"""Wavelet Lifting Transforms."""

# -*- coding: utf-8 -*-

# Created on Nov 14, 2023
# Copyright (c) 2023 Yufan Zhuang
#

from typing import List, Optional, Tuple, Union
import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", False)

@partial(jax.jit, static_argnames=['level'])
def liftdec(data: jnp.ndarray, level: Optional[int] = 1) -> List[jnp.ndarray]:
    ## Lazy Wavelet
    ## forward process
    A = _fwt_pad(data)
    result_lst = []
    for l in range(level):
        dec_lo = A[:, ::2, :]

        id1 = list(range(dec_lo.shape[1]))
        id2 = list(range(1, dec_lo.shape[1])) + [0]
        id3 = [dec_lo.shape[1]] + list(range(0, dec_lo.shape[1]-1)) 

        lazy_wavelet = A[:, 1::2, :] - 1/2 * (dec_lo[:, id1, :] + dec_lo[:, id2, :])
        dec_lifted = dec_lo + 1/4 * (lazy_wavelet[:, id3, :] + lazy_wavelet[:, id1, :])
        result_lst.append(lazy_wavelet)
        A = _fwt_pad(dec_lifted)
    result_lst.append(dec_lifted)
    result_lst.reverse()
    return result_lst


@jax.jit
def liftrec(coeffs: List[jnp.ndarray]) -> jnp.ndarray:
    ## Backward process
    dec_lifted = coeffs[0]
    for lazy_wavelet in coeffs[1:]:
        lazy_wavelet = _bk_pad(lazy_wavelet, dec_lifted)
        id1 = list(range(dec_lifted.shape[1]))
        id2 = list(range(1, dec_lifted.shape[1])) + [0]
        id3 = [dec_lifted.shape[1]] + list(range(0, dec_lifted.shape[1]-1)) 
        
        rec_lo = dec_lifted - 1/4 * (lazy_wavelet[:, id3, :] + lazy_wavelet[:, id1, :])
        rec_high = lazy_wavelet + 1/2 * (rec_lo[:, id1, :] + rec_lo[:, id2, :])

        B = jnp.zeros(shape=(rec_lo.shape[0], rec_lo.shape[1] + rec_high.shape[1], rec_lo.shape[2]))
        B = B.at[:, ::2, :].set(rec_lo)
        B = B.at[:, 1::2, :].set(rec_high)
        dec_lifted = B
    return dec_lifted


@partial(jax.jit, static_argnames=['level'])
def liftdec_learn(data: jnp.ndarray, wavelet_est: jnp.ndarray, wavelet_pred: jnp.ndarray,
                  level: Optional[int] = 1) -> List[jnp.ndarray]:
    ## Lazy Wavelet
    ## forward process
    A = _fwt_pad(data)
    result_lst = []
    for l in range(level):
        dec_loe = A[:, ::2, :]
        dec_loo = A[:, 1::2, :]

        id1 = list(range(dec_loe.shape[1]))
        id2 = list(range(1, dec_loe.shape[1])) + [0]
        id3 = [dec_loe.shape[1]] + list(range(0, dec_loe.shape[1]-1)) 
        fft_shape = dec_loe.shape[1]

        estimation = nn.silu(fft_subroutine(dec_loe, wavelet_est))        
        lazy_wavelet = dec_loo - 1/2 * estimation
        
        prediction = nn.silu(fft_subroutine(lazy_wavelet, wavelet_pred)) 
        dec_lifted = dec_loe + 1/4 * prediction
        
        result_lst.append(lazy_wavelet)
        A = _fwt_pad(dec_lifted)
    result_lst.append(dec_lifted)
    result_lst.reverse()
    return result_lst


@jax.jit
def liftrec_learn(coeffs: List[jnp.ndarray], wavelet_est: jnp.ndarray, wavelet_pred: jnp.ndarray) -> jnp.ndarray:
    ## Backward process
    dec_lifted = coeffs[0]
    for lazy_wavelet in coeffs[1:]:
        dec_lifted = dec_lifted[:,:lazy_wavelet.shape[1], :]
        prediction = nn.silu(fft_subroutine(lazy_wavelet, wavelet_pred))
        #lazy_wavelet = _bk_pad(lazy_wavelet, dec_lifted)
        rec_lo = dec_lifted - 1/4 * prediction
        estimation = nn.silu(fft_subroutine(rec_lo, wavelet_est))
        rec_high = lazy_wavelet + 1/2 * estimation
        
        B = jnp.zeros(shape=(rec_lo.shape[0], rec_lo.shape[1] + rec_high.shape[1], rec_lo.shape[2]))
        B = B.at[:, ::2, :].set(rec_lo)
        B = B.at[:, 1::2, :].set(rec_high)
        dec_lifted = B
    return dec_lifted

def fft_subroutine(A, filt):
    fft_shape = A.shape[-2]
    A_fft = jnp.fft.rfft(A, n=fft_shape, axis=-2)
    filt_fft = jnp.fft.rfft(filt, n=fft_shape, axis=-2)
    filt_fft = jnp.conjugate(filt_fft)
    return jnp.fft.irfft(A_fft * filt_fft, n=fft_shape, axis=-2)


def _fwt_pad(data: jnp.ndarray) -> jnp.ndarray:
    """Pad an input to ensure our fwts are invertible.

    Args:
        data (jnp.array): The input array.

    Returns:
        jnp.array: A padded version of the input data array.
    """
    padr = 0
    padl = 0

    # pad to even singal length.
    if data.shape[1] % 2 != 0:
        padr += 1
        
    data = jnp.pad(data, ((0, 0), (padl, padr), (0, 0)))
    return data


def _bk_pad(data: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Pad an input to ensure our fwts are invertible.

    Args:
        data (jnp.array): The input array.

    Returns:
        jnp.array: A padded version of the input data array.
    """
    padr = reference.shape[1] - data.shape[1]
    padl = 0
        
    data = jnp.pad(data, ((0, 0), (padl, padr), (0, 0)))
    return data