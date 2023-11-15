"""Convolution based learnable fast wavelet transforms."""

# -*- coding: utf-8 -*-

# Created on Nov 14, 2023
# Copyright (c) 2023 Yufan Zhuang
#
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.config import config
import pywt
from functools import partial

from .utils import Wavelet

config.update("jax_enable_x64", False)


@partial(jax.jit, static_argnames=['level'])
def wavedec(
    data: jnp.ndarray,
    wavelet: jnp.ndarray,
    level: Optional[int] = 1,
) -> List[jnp.ndarray]:
    """Compute the one dimensional analysis wavelet transform of the last dimension.

    Args:
        data (jnp.array): Input data array of shape [batch, channels, time]
        wavelet (Wavelet): The named tuple containing the wavelet filter arrays.
        level (int): Max scale level to be used, of none as many levels as possible are
                     used. Defaults to None.
        mode: The padding used to extend the input signal. Choose reflect, symmetric or zero.
            Defaults to reflect.

    Returns:
        list: List containing the wavelet coefficients.
            The coefficients are in pywt order:
            [cA_n, cD_n, cD_n-1, …, cD2, cD1].
            A denotes approximation and D detail coefficients.

    Examples:
        >>> import pywt
        >>> import jaxwt as jwt
        >>> import jax.numpy as jnp
        >>> # generate an input of even length.
        >>> data = jnp.array([0., 1., 2., 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> jwt.wavedec(data, pywt.Wavelet('haar'),
                        mode='reflect', level=2)
    """
    mode = "reflect"
    dec_lo, dec_hi, _, _ = CQF_expansion_flip(wavelet)
    filt_len = dec_lo.shape[0]
    filt = jnp.stack([dec_lo, dec_hi], 0)    

    result_lst = []
    res_lo = data
    for _ in range(level):        
        ## FFT Routine
        res_ll = _fwt_pad(jnp.squeeze(res_lo), len(wavelet), mode=mode)
        dec_filt = jnp.squeeze(filt)
        fft_shape = res_ll.shape[-2]
        
        A = jnp.fft.rfft(res_ll, n=fft_shape, axis=-2)
        B = jnp.fft.rfft(dec_filt, n=fft_shape, axis=-2)
        B = jnp.conjugate(B)
        
        A = jnp.expand_dims(A, axis=1)
        ret = jnp.fft.irfft(A * B, n=fft_shape, axis=-2)
        
        # Strides
        res = ret[:,:,::2,:]
        
        res_lo, res_hi = jnp.split(res, 2, 1)
        result_lst.append(res_hi)
    result_lst.append(res_lo)
    result_lst.reverse()
    return result_lst


@jax.jit
def waverec(coeffs: List[jnp.ndarray], wavelet: Wavelet) -> jnp.ndarray:
    """Reconstruct the original signal in one dimension.

    Args:
        coeffs (list): Wavelet coefficients, typically produced by the wavedec function.
        wavelet (Wavelet): The named tuple containing the wavelet filters used to evaluate
                              the decomposition.

    Returns:
        jnp.array: Reconstruction of the original data.

    Examples:
        >>> import pywt
        >>> import jaxwt as jwt
        >>> import jax.numpy as jnp
        >>> # generate an input of even length.
        >>> data = jnp.array([0., 1., 2., 3, 4, 5, 5, 4, 3, 2, 1, 0])
        >>> transformed = jwt.wavedec(data, pywt.Wavelet('haar'),
                          mode='reflect', level=2)
        >>> jwt.waverec(transformed, pywt.Wavelet('haar'))
    """
    # lax's transpose conv requires filter flips in contrast to pytorch.
    # Edit on FFT VERION No flip
    _, _, rec_lo, rec_hi = CQF_expansion(wavelet)
    filt_len = rec_lo.shape[0]
    filt = jnp.stack([rec_lo, rec_hi], 0)

    res_lo = coeffs[0]
    for c_pos, res_hi in enumerate(coeffs[1:]):
        # print('shapes', res_lo.shape, res_hi.shape)
        res_lo = jnp.concatenate([res_lo, res_hi], 1)
        
        ## FFT ROUTINE
        res_ll = jnp.squeeze(res_lo)
        rec_filt = jnp.squeeze(filt)
        
        res_ll_dialated = jnp.zeros((res_ll.shape[0], res_ll.shape[1], res_ll.shape[2]*2, res_ll.shape[3]))
        res_ll = res_ll_dialated.at[:,:,::2,:].set(res_ll)
        
        fft_shape = res_ll.shape[-2]
        A = jnp.fft.rfft(res_ll, n=fft_shape, axis=-2)
        B = jnp.fft.rfft(rec_filt, n=fft_shape, axis=-2)
        
        ret = jnp.fft.irfft(A * B, axis=-2)
        res_lo = jnp.sum(ret, axis=1)

        res_lo = _fwt_unpad(res_lo, filt_len, c_pos, coeffs)
        res_lo = jnp.expand_dims(res_lo, axis=1)
    return jnp.squeeze(res_lo)


def _fwt_unpad(
    res_lo: jnp.ndarray, filt_len: int, c_pos: int, coeffs: List[jnp.ndarray]
) -> jnp.ndarray:
    padr = 0
    padl = 0
    if filt_len > 2:
        padr += (2 * filt_len - 3) // 2
        padl += (2 * filt_len - 3) // 2
    if c_pos < len(coeffs) - 2:
        pred_len = res_lo.shape[-2] - (padl + padr)
        nex_len = coeffs[c_pos + 2].shape[-2]
        if nex_len != pred_len:
            padl += 1
            pred_len = res_lo.shape[-2] - padl
            # assert nex_len == pred_len, 'padding error, please open an issue on github '
    if padl == 0:
        res_lo = res_lo[:, padr:, :]
    else:
        res_lo = res_lo[:, padr:-padl, :]
    return res_lo


def _fwt_pad(data: jnp.ndarray, filt_len: int, mode: str = "reflect") -> jnp.ndarray:
    """Pad an input to ensure our fwts are invertible.

    Args:
        data (jnp.array): The input array.
        filt_len (int): The length of the wavelet filters
        mode (str): How to pad. Defaults to "reflect".

    Returns:
        jnp.array: A padded version of the input data array.
    """
    # pad to we see all filter positions and pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    if mode == "zero":
        # translate pywt to numpy.
        mode = "constant"

    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data.shape[-2] % 2 != 0:
        padr += 1
    
    data = jnp.pad(data, ((0, 0), (padl, padr), (0, 0)), mode)
    return data


@jax.jit
def CQF_expansion(dec_lo: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = dec_lo.shape[0]
    a = jnp.empty((N,1),int)
    a = a.at[::2].set(1)
    a = a.at[1::2].set(-1)
    
    dec_hi = -1 * a * jnp.flip(dec_lo)
    rec_lo = jnp.flip(dec_lo)
    rec_hi = a * dec_lo
    return dec_lo, dec_hi, rec_lo, rec_hi


@jax.jit
def CQF_expansion_flip(dec_lo: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = dec_lo.shape[0]
    a = jnp.empty((N,1),int)
    a = a.at[::2].set(1)
    a = a.at[1::2].set(-1)
    
    dec_hi = -1 * a * jnp.flip(dec_lo)
    rec_lo = jnp.flip(dec_lo)
    rec_hi = a * dec_lo
    return jnp.flip(dec_lo), jnp.flip(dec_hi), jnp.flip(rec_lo), jnp.flip(rec_hi)