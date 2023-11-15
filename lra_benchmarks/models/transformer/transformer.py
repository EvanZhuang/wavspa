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
"""Transformer model."""
from flax import linen as nn
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from typing import Callable, Any, Optional


class TransformerBlock(nn.Module):
    """Transformer layer (https://openreview.net/forum?id=H1e5GJBtDr)."""
    qkv_dim: int
    mlp_dim: int
    num_heads: int
    dtype: jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    decode: bool = False
        
    @nn.compact
    def __call__(self,
              inputs,
              inputs_segmentation=None,
              padding_mask=None,
              deterministic=False):
        """Applies TransformerBlock module.

        Args:
          inputs: input data
          qkv_dim: dimension of the query/key/value
          mlp_dim: dimension of the mlp on top of attention block
          num_heads: number of heads
          dtype: the dtype of the computation (default: float32).
          inputs_segmentation: input segmentation info for packed examples.
          causal_mask: bool, mask future or not
          padding_mask: bool, mask padding tokens
          dropout_rate: dropout rate
          attention_dropout_rate: dropout rate for attention weights
          deterministic: bool, deterministic or not (to apply dropout)
          cache: flax autoregressive cache for fast decoding.

        Returns:
          output after transformer block.

        """

        # Attention block.
        assert inputs.ndim == 3
        
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            dtype=self.dtype,
            qkv_features=self.qkv_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6),
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=self.attention_dropout_rate,
            decode=self.decode)(x, mask=padding_mask, deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = common_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate)(y, deterministic=deterministic)

        return x + y


class TransformerEncoder(nn.Module):
    """Transformer Model Encoder."""
    vocab_size: int
    use_bfloat16: bool = False
    emb_dim: int = 512
    num_heads: int = 8
    dtype: Any = jnp.float32
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 512
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    learn_pos_emb: bool = False
    classifier: bool = False
    classifier_pool: str = 'CLS'
    num_classes: int = 10
    tied_weights: bool = False
    shared_embedding: Optional[Callable] = None
    
    @nn.compact
    def __call__(self,
              inputs,
              inputs_positions=None,
              inputs_segmentation=None,
              train:bool=False):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          vocab_size: size of the vocabulary
          inputs_positions: input subsequence positions for packed examples.
          inputs_segmentation: input segmentation info for packed examples.
          shared_embedding: a shared embedding layer to use.
          use_bfloat16: bool: whether use bfloat16.
          emb_dim: dimension of embedding
          num_heads: number of heads
          dtype: the dtype of the computation (default: float32)
          num_layers: number of layers
          qkv_dim: dimension of the query/key/value
          mlp_dim: dimension of the mlp on top of attention block
          max_len: maximum length.
          train: if it is training,
          dropout_rate: dropout rate
          attention_dropout_rate: dropout rate for attention weights
          learn_pos_emb: boolean, if learn the positional embedding or use the
            sinusoidal positional embedding.
          classifier: boolean, for classification mode (output N-class logits)
          classifier_pool: str, supports "MEAN", "MAX" pooling.
          num_classes: int, number of classification classes.
          tied_weights: bool, to tie weights or not.

        Returns:
          output of a transformer encoder or logits if classifier_mode is true.
        """
        assert inputs.ndim == 2  # (batch, len)

        # Padding Masks
        src_padding_mask = (inputs > 0)[..., None]
        encoder_mask = nn.make_attention_mask(inputs > 0, inputs > 0, dtype=self.dtype)
        if inputs_segmentation is not None:
            encoder_mask = nn.combine_masks(encoder_mask,
                nn.make_attention_mask(
                    inputs_segmentation,
                    inputs_segmentation,
                    jnp.equal,
                    dtype=config.dtype))

        # Input Embedding
        if self.shared_embedding is None:
            input_embed = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.emb_dim,
                embedding_init=nn.initializers.normal(stddev=1.0))
        else:
            input_embed = self.shared_embedding

        x = inputs.astype('int32')
        x = input_embed(x)
        max_len = self.max_len
        if self.classifier and self.classifier_pool == 'CLS':
            cls = self.param('cls', nn.initializers.zeros, (1, 1, self.emb_dim))
            cls = jnp.tile(cls, [x.shape[0], 1, 1])
            x = jnp.concatenate([cls, x], axis=1)
            max_len += 1
            encoder_mask = jnp.concatenate([encoder_mask[:, :, :1, :], encoder_mask], axis=-2)
            encoder_mask = jnp.concatenate([encoder_mask[:, :, :, :1], encoder_mask], axis=-1)
            src_padding_mask = jnp.concatenate(
                [src_padding_mask[:, :1], src_padding_mask], axis=1)
            
        pe_init = nn.initializers.normal(
            stddev=0.02) if self.learn_pos_emb else None
        x = common_layers.AddPositionEmbs(
            posemb_init=pe_init,
            max_len=max_len,
            name='posembed_input')(x, inputs_positions=inputs_positions)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        if self.use_bfloat16:
            x = x.astype(jnp.bfloat16)
            dtype = jnp.bfloat16
        else:
            dtype = jnp.float32

        # Input Encoder
        if self.tied_weights:
            encoder = TransformerBlock.shared(
                qkv_dim=self.qkv_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dtype=self.dtype,
                padding_mask=src_padding_mask,
                inputs_segmentation=inputs_segmentation,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name='encoderblock')(padding_mask=src_padding_mask, inputs_segmentation=inputs_segmentation, deterministic=not train)
            for _ in range(self.num_layers):
                x = encoder(x)
        else:
            for lyr in range(self.num_layers):
                x = TransformerBlock(
                    qkv_dim=self.qkv_dim,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    dropout_rate=self.dropout_rate,
                    attention_dropout_rate=self.attention_dropout_rate,
                    name=f'encoderblock_{lyr}')(x, inputs_segmentation=inputs_segmentation, 
                                                padding_mask=encoder_mask, deterministic=not train)

        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

        if self.classifier:
            encoded = common_layers.classifier_head(
                encoded, self.num_classes, self.mlp_dim, pooling_mode=self.classifier_pool)
        return encoded


class TransformerDualEncoder(nn.Module):
    """Transformer Model for Matching (dual encoding) tasks."""
    vocab_size: int
    use_bfloat16: bool = False
    emb_dim: int = 512
    num_heads: int = 8
    dtype: Any = jnp.float32
    num_layers: int = 6
    qkv_dim: int = 512
    mlp_dim: int = 2048
    max_len: int = 512
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    learn_pos_emb: bool = False
    classifier: bool = False
    classifier_pool: str = 'CLS'
    num_classes: int = 10
    tied_weights: bool = False
    shared_embedding: Optional[Callable] = None
    interaction: str = None
        
    @nn.compact
    def __call__(self,
              inputs1,
              inputs2,
              inputs1_positions=None,
              inputs2_positions=None,
              inputs1_segmentation=None,
              inputs2_segmentation=None,
              train:bool=False):
        """Applies Transformer model on text similarity.

        A deliberate choice to distinguish this from NLI because
        we may want to do different things to the model later. Dual Encoding
        mode enforces that we do not do cross attention between pairs.

        Args:
          inputs1: input data.
          inputs2: target data.
          vocab_size: size of the input vocabulary.
          inputs1_positions: input subsequence positions for packed examples.
          inputs2_positions: target subsequence positions for packed examples.
          inputs1_segmentation: input segmentation info for packed examples.
          inputs2_segmentation: target segmentation info for packed examples.
          use_bfloat16: bool: whether use bfloat16.
          emb_dim: dimension of embedding.
          num_heads: number of heads.
          num_layers: number of layers.
          qkv_dim: dimension of the query/key/value.
          mlp_dim: dimension of the mlp on top of attention block.
          max_len: maximum length.
          train: whether it is training.
          dropout_rate: dropout rate.
          attention_dropout_rate: dropout rate for attention weights.
          classifier: boolean, to use classifier.
          classifier_pool: str, supports "MEAN", "MAX" pooling.
          num_classes: int, number of classification classes.
          interaction: str, supports "NLI"

        Returns:
          output of a transformer decoder.
        """

        encoder = TransformerEncoder(
            vocab_size=self.vocab_size,
            use_bfloat16=self.use_bfloat16,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            max_len=self.max_len,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            name='encoder')
        inputs1_encoded = encoder(
            inputs=inputs1,
            inputs_positions=inputs1_positions,
            inputs_segmentation=inputs1_segmentation,
            train=train)
        inputs2_encoded = encoder(
            inputs=inputs2,
            inputs_positions=inputs2_positions,
            inputs_segmentation=inputs2_segmentation,
            train=train)

        encoded = common_layers.classifier_head_dual(
            inputs1_encoded,
            inputs2_encoded,
            self.num_classes,
            self.mlp_dim,
            pooling_mode=self.classifier_pool,
            interaction=self.interaction)

        return encoded
