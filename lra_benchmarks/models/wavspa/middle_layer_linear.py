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
"""Custom Attention modules for Linear Transformer."""

from flax import linen as nn
import jax
import jax.numpy as jnp
from typing import Callable, Any, Optional



def elu_feature_map(x):
    return nn.elu(x) + 1


def linear_attention(query,
                     key,
                     value,
                     feature_map=elu_feature_map,
                     eps=1e-6):
    """Computes linear attention given query, key, and value.


    Args:
      query: queries for calculating attention with shape of `[batch_size, len,
        num_heads, mem_channels]`.
      key: keys for calculating attention with shape of `[batch_size, len,
        num_heads, mem_channels]`.
      value: values to be used in attention with shape of `[batch_size, len,
        num_heads, value_channels]`.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      feature_map: function, to map query and key to a new feature space.
      eps: float, used to avoid division by zero.

    Returns:
      Output of shape `[bs, length, num_heads, value_channels]`.
    """
    assert key.ndim == query.ndim == value.ndim == 4
    assert key.shape[:-1] == value.shape[:-1]
    assert (query.shape[0:1] == key.shape[0:1] and
            query.shape[-1] == key.shape[-1])

    query_mapped = feature_map(query)
    key_mapped = feature_map(key)
    kv = jnp.einsum('nshd,nshm->nhmd', key_mapped, value)

    z = 1 / (
        jnp.einsum('nlhd,nhd->nlh', query_mapped, jnp.sum(key_mapped, axis=1)) +
        eps)
    y = jnp.einsum('nlhd,nhmd,nlh->nlhm', query_mapped, kv, z)

    return y


class LinearAttention(nn.Module):
    """Linear Attention Architecture."""
    num_heads: int
    qkv_features: int
    out_features: int = None
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    bias: bool = True
        
    @nn.compact
    def __call__(self,
              inputs_q,
              inputs_kv=None,
              deterministic=False):
        """Applies linear attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies linear attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
          inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]` or
            None for self-attention, inn which case key/values will be derived from
            inputs_q.
          num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
            should be divisible by the number of heads.
          dtype: the dtype of the computation (default: float32)
          qkv_features: dimension of the key, query, and value.
          out_features: dimension of the last projection
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

        Returns:
          output of shape `[bs, dim1, dim2, ..., dimN, features]`.
        """

        assert inputs_q.ndim == 3

        if inputs_kv is None:
            inputs_kv = inputs_q

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]

        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

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
        # apply regular dot product attention
        x = linear_attention(query, key, value)

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
