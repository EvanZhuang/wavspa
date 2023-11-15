"""Two dimensional convolution based learnable fast wavelet transforms."""
#
# Created on Nov 14, 2023
# Copyright (c) 2023 Yufan Zhuang
#
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pywt
from jax.config import config
from functools import partial


from .conv_fwt import _get_filter_arrays
from .utils import Wavelet

config.update("jax_enable_x64", False)

@partial(jax.jit, static_argnames=['level'])
def wavedec2(
    data: jnp.ndarray,
    wavelet: jnp.ndarray,
    level: Optional[int] = None
) -> List[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]]:
    """Compute the two dimensional wavelet analysis transform on the last two dimensions of the input data array.

    Args:
        data (jnp.array): Jax array containing the data to be transformed. Assumed shape:
                         [batch size, hight, width].
        wavelet (Wavelet): A namedtouple containing the filters for the transformation.
        level (int): The max level to be used, if not set as many levels as possible
                               will be used. Defaults to None.
        mode (str): The desired padding mode. Choose reflect, symmetric or zero.
            Defaults to reflect.

    Returns:
        list: The wavelet coefficients in a nested list.
            The coefficients are in pywt order. That is:
            [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)].
            A denotes approximation, H horizontal, V vertical
            and D diagonal coefficients.

    Examples:
        >>> import pywt, scipy.misc
        >>> import jaxwt as jwt
        >>> import jax.numpy as jnp
        >>> face = jnp.transpose(scipy.misc.face(), [2, 0, 1]).astype(jnp.float64)
        >>> jwt.wavedec2(face, pywt.Wavelet("haar"), level=2, mode="reflect")
    """
    data = jnp.expand_dims(data, 1)
    mode = "reflect"
    
    if level is None:
        level = pywt.dwtn_max_level(
            [data.shape[-1], data.shape[-2]], pywt.Wavelet("MyWavelet", wavelet)
        )
        
    wavelen = wavelet.shape[-1]
    dec_lo, dec_hi, _, _ = _get_custom_filter_arrays(wavelet)
    dec_filt = construct_2d_filt(lo=dec_lo, hi=dec_hi)

    result_lst: List[
        Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    ] = []
    res_ll = data
    for _ in range(level):
        res_ll = _fwt_pad2d(res_ll, wavelen, mode=mode)
        
        ## FFT Routine
        res_ll = jnp.squeeze(_fwt_pad2d(res_ll, len(wavelet), mode=mode))
        dec_filt = jnp.squeeze(dec_filt)
        
        fft_shape = (res_ll.shape[1], res_ll.shape[2])
        
        a = jnp.fft.rfftn(res_ll, s=fft_shape, axes=[1,2])
        b = jnp.fft.rfftn(dec_filt, s=fft_shape, axes=[1,2])
        
        a = jnp.expand_dims(a, axis=1)
        ret = jnp.fft.irfftn(a * b, fft_shape)
        
        # Strides
        res = ret[:,:,::2,::2]
        
        res_ll, res_lh, res_hl, res_hh = jnp.split(res, 4, 1)
        result_lst.append([res_lh.squeeze(1), res_hl.squeeze(1), res_hh.squeeze(1)])
    result_lst.append(res_ll.squeeze(1))
    result_lst.reverse()
    return result_lst


@jax.jit
def waverec2(
    coeffs: List[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]],
    wavelet: jnp.ndarray,
) -> jnp.ndarray:
    """Compute a two dimensional synthesis wavelet transfrom.

       Use it to reconstruct the original input image from the wavelet coefficients.

    Args:
        coeffs (list): The input coefficients, typically the output of wavedec2.
        wavelet (Wavelet): The named tuple contining the filters used to compute the analysis transform.

    Returns:
        jnp.array: Reconstruction of the original input data array of shape [batch, height, width].

    Raises:
        ValueError: If `coeffs` is not in the shape as it is returned from `wavedec2`.

    Example:
        >>> import pywt, scipy.misc
        >>> import jaxwt as jwt
        >>> import jax.numpy as jnp
        >>> face = jnp.transpose(scipy.misc.face(), [2, 0, 1]).astype(jnp.float64)
        >>> transformed = jwt.wavedec2(face, pywt.Wavelet("haar"), level=2, mode="reflect")
        >>> jwt.waverec2(transformed, pywt.Wavelet("haar"))


    """
    if not isinstance(coeffs[0], jnp.ndarray):
        raise ValueError(
            "First element of coeffs must be the approximation coefficient tensor."
        )
        
    _, _, rec_lo, rec_hi = _get_custom_filter_arrays(wavelet)
    filt_len = rec_lo.shape[-1]
    rec_filt = construct_2d_filt(lo=rec_lo, hi=rec_hi)
    rec_filt = jnp.transpose(rec_filt, [1, 0, 2, 3])
    res_ll = jnp.expand_dims(coeffs[0], 1)
    for c_pos, res_lh_hl_hh in enumerate(coeffs[1:]):
        
    
        res_ll = jnp.concatenate(
            [
                res_ll,
                jnp.expand_dims(res_lh_hl_hh[0], 1),
                jnp.expand_dims(res_lh_hl_hh[1], 1),
                jnp.expand_dims(res_lh_hl_hh[2], 1),
            ],
            1,
        )
        
        ## FFT ROUTINE
        res_ll = jnp.squeeze(res_ll)
        rec_filt = jnp.squeeze(rec_filt)
        
        res_ll_dialated = jnp.zeros((res_ll.shape[0], res_ll.shape[1], res_ll.shape[2]*2, res_ll.shape[3]*2))
        res_ll = res_ll_dialated.at[:,:,::2,::2].set(res_ll)
        
        fft_shape = (res_ll.shape[-2], res_ll.shape[-1])
        a = jnp.fft.rfftn(res_ll, axes=[-2,-1])
        b = jnp.fft.rfftn(rec_filt, s=fft_shape, axes=[-2,-1])
        
        ret = jnp.fft.irfftn(a * b, s=fft_shape, axes=[-2,-1])
        res_ll = jnp.sum(ret, axis=1)
        
        # remove the padding
        padl = (2 * filt_len - 3) // 2
        padr = (2 * filt_len - 3) // 2
        padt = (2 * filt_len - 3) // 2
        padb = (2 * filt_len - 3) // 2
        if c_pos < len(coeffs) - 2:
            pred_len = res_ll.shape[-1] - (padl + padr)
            next_len = coeffs[c_pos + 2][0].shape[-1]
            pred_len2 = res_ll.shape[-2] - (padt + padb)
            next_len2 = coeffs[c_pos + 2][0].shape[-2]
            if next_len != pred_len:
                padr += 1
                pred_len = res_ll.shape[-1] - (padl + padr)
                assert (
                    next_len == pred_len
                ), "padding error, please open an issue on github "
            if next_len2 != pred_len2:
                padb += 1
                pred_len2 = res_ll.shape[-2] - (padt + padb)
                assert (
                    next_len2 == pred_len2
                ), "padding error, please open an issue on github "
        # print('padding', padt, padb, padl, padr)
        if padt > 0:
            res_ll = res_ll[..., padt:, :]
        if padb > 0:
            res_ll = res_ll[..., :-padb, :]
        if padl > 0:
            res_ll = res_ll[..., padl:]
        if padr > 0:
            res_ll = res_ll[..., :-padr]
    return res_ll


@jax.jit
def CQF_expansion(dec_lo: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    N = dec_lo.shape[0]
    a = jnp.empty((N,),int)
    a = a.at[::2].set(1)
    a = a.at[1::2].set(-1)
    
    #a[::2] = 1
    #a[1::2] = -1

    dec_hi = -1 * a * jnp.flip(dec_lo)
    rec_lo = jnp.flip(dec_lo)
    rec_hi = a * dec_lo
    return dec_lo, dec_hi, rec_lo, rec_hi


@jax.jit
def _get_custom_filter_arrays(
    wavelet: jnp.ndarray, dtype: jnp.dtype = jnp.float32  # type: ignore
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract the filter coefficients from an input wavelet object.

    Args:
        wavelet (Wavelet): input wavelet dec_lo parameters
        flip (bool): If true flip the input coefficients.
        dtype: The desired precision. Defaults to jnp.float64 .

    Returns:
        tuple: The dec_lo, dec_hi, rec_lo and rec_hi
            filter coefficients as jax arrays.
    """

    def create_array(filter: Union[List[float], jnp.ndarray]) -> jnp.ndarray:
        
        return jnp.expand_dims(filter, 0)

    dec_lo, dec_hi, rec_lo, rec_hi = CQF_expansion(wavelet)
    return dec_lo, dec_hi, rec_lo, rec_hi


@jax.jit
def construct_2d_filt(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Construct 2d filters from 1d inputs using outer products.

    Args:
        lo (jnp.array): 1d lowpass input filter of size [1, length].
        hi (jnp.array): 1d highpass input filter of size [1, length].

    Returns
        jnp.array: 2d filter arrays of shape [4, 1, length, length].
    """
    ll = jnp.outer(lo, lo)
    lh = jnp.outer(hi, lo)
    hl = jnp.outer(lo, hi)
    hh = jnp.outer(hi, hi)
    filt = jnp.stack([ll, lh, hl, hh], 0)
    filt = jnp.expand_dims(filt, 1)
    return filt


def _fwt_pad2d(data: jnp.ndarray, filt_len: int, mode: str = "reflect") -> jnp.ndarray:
    padr = 0
    padl = 0
    padt = 0
    padb = 0
    if filt_len > 2:
        # we pad half of the total requried padding on each side.
        padr += (2 * filt_len - 3) // 2
        padl += (2 * filt_len - 3) // 2
        padt += (2 * filt_len - 3) // 2
        padb += (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data.shape[-1] % 2 != 0:
        padr += 1
    if data.shape[-2] % 2 != 0:
        padb += 1

    data = jnp.pad(data, ((0, 0), (0, 0), (padt, padb), (padl, padr)), mode)
    return data
