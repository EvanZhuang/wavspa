# -*- coding: utf-8 -*-

"""Differentiable and gpu enabled fast wavelet transforms in JAX."""
from .continuous_transform import cwt
from .conv_fwt import wavedec, waverec
from .conv_fwt_learn import wavedec as wavedec_learn, waverec as waverec_learn

from .conv_fwt_2d import wavedec2, waverec2
from .conv_fwt_2d_learn import wavedec2 as wavedec2_learn, waverec2 as waverec2_learn
from .packets import WaveletPacket, WaveletPacket2D

from .wavelet_lifting import liftdec, liftrec
