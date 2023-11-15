# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom Attention core modules for Flax."""

from flax import linen as nn
from flax.linen.attention import dot_product_attention
from jax import lax
import jax.numpy as jnp
from typing import Callable, Any, Optional
from flax.linen.linear import PrecisionLike


class LinformerAttention(nn.Module):
    """Linformer Architecture."""
    
    num_heads: int
    qkv_features: int
    out_features: int = None
    dtype: Any = jnp.float32
    broadcast_dropout: bool=True
    dropout_rate: float=0.
    precision: PrecisionLike = None
    kernel_init: Callable = nn.linear.default_kernel_init
    bias_init: Callable = nn.initializers.zeros
    bias: bool = True
    low_rank_features: int = 16
    max_len: int = 1000
        
    @nn.compact        
    def __call__(self,
              inputs_q,
              inputs_kv=None, deterministic=False):
        """Applies Linformer's low-rank attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        This can be used for encoder-decoder attention by specifying both `inputs_q`
        and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
        setting `inputs_kv` to None.

        Args:
          inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
          inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
            or None for self-attention, inn which case key/values will be derived
            from inputs_q.
          num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
            should be divisible by the number of heads.
          dtype: the dtype of the computation (default: float32)
          qkv_features: dimension of the key, query, and value.
          out_features: dimension of the last projection
          attention_axis: axes over which the attention is applied ( 'None' means
            attention over all axes, but batch, heads, and features).
          causal_mask: boolean specifying whether to apply a causal mask on the
            attention weights. If True, the output at timestep `t` will not depend
            on inputs at timesteps strictly greater than `t`.
          padding_mask: boolean specifying query tokens that are pad token.
          key_padding_mask: boolean specifying key-value tokens that are pad token.
          segmentation: segment indices for packed inputs_q data.
          key_segmentation: segment indices for packed inputs_kv data.
          cache: an instance of `flax.nn.attention.Cache` used for efficient
            autoregressive decoding.
          broadcast_dropout: bool: use a broadcasted dropout along batch dims.
          dropout_rng: JAX PRNGKey: to be used for dropout
          dropout_rate: dropout rate
          deterministic: bool, deterministic or not (to apply dropout)
          precision: numerical precision of the computation see `jax.lax.Precision`
            for details.
          kernel_init: initializer for the kernel of the Dense layers.
          bias_init: initializer for the bias of the Dense layers.
          bias: bool: whether pointwise QKVO dense transforms use bias.
          low_rank_features: int: how many low-rank projected features.
          max_len: int maximum sequence length.

        Returns:
          output of shape `[bs, dim1, dim2, ..., dimN, features]`.
        """

        assert inputs_q.ndim == 3

        if inputs_kv is None:
            inputs_kv = inputs_q

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]

        assert self.qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = self.qkv_features // self.num_heads

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [bs, dims..., n_heads, n_features_per_head]

        query, key, value = (nn.DenseGeneral(axis=-1, features=(self.num_heads, head_dim),
                                             kernel_init=self.kernel_init,
                                             bias_init=self.bias_init,
                                             use_bias=self.bias, dtype=self.dtype, name='query')(inputs_q),
                             nn.DenseGeneral(axis=-1, features=(self.num_heads, head_dim),
                                             kernel_init=self.kernel_init,
                                             bias_init=self.bias_init,
                                             use_bias=self.bias, dtype=self.dtype, name='key')(inputs_kv),
                             nn.DenseGeneral(axis=-1, features=(self.num_heads, head_dim),
                                             kernel_init=self.kernel_init,
                                             bias_init=self.bias_init,
                                             use_bias=self.bias, dtype=self.dtype, name='value')(inputs_kv))

        def low_rank_projection(inputs, kernel, precision):
            """low rank projection."""
            input_dim = inputs.shape[1]
            # this kernel/parameter relies on sequence length
            kernel = kernel[:input_dim, :]
            inputs = inputs.transpose((0, 3, 2, 1))
            y = lax.dot_general(inputs, kernel,
                                (((inputs.ndim - 1,), (0,)), ((), ())),
                                precision=precision)
            y = y.transpose((0, 3, 2, 1))
            return y

        # Shared Kernel for low-rank length dimension projections.
        low_rank_kernel = self.param('lr_kernel', self.kernel_init, 
                                     (self.max_len, self.low_rank_features), self.dtype)

        key = low_rank_projection(key, low_rank_kernel, self.precision)
        value = low_rank_projection(value, low_rank_kernel, self.precision)

        # apply regular dot product attention
        x = dot_product_attention(
            query,
            key,
            value,
            dtype=self.dtype,
            bias=None,
            precision=self.precision,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=True)
        
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # back to the original inputs dimensions
        out = nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.bias,
            dtype=self.dtype,
            name='out')(x)

        return out

#LinformerSelfAttention = LinformerAttention.partial(inputs_kv=None)
