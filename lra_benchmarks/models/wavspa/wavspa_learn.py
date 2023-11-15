"""Transformer model."""
import numpy as np
from flax import linen as nn
import jax
import jax.numpy as jnp
from lra_benchmarks.models.layers import common_layers
from typing import Callable, Any, Optional
from lra_benchmarks.utils.fft_convolve import fftconvolve
import jax.scipy.signal as jsignal
import pywt
import wavspa
from jax.experimental import sparse
from wavspa.wavelet_lifting import liftdec, liftrec, liftdec_learn, liftrec_learn


from lra_benchmarks.models.wavspa.middle_layer_linear import LinearAttention
from lra_benchmarks.models.wavspa.middle_layer_linformer import LinformerAttention
from lra_benchmarks.models.wavspa.middle_layer_performer import PerformerAttn
from lra_benchmarks.models.wavspa.middle_layer_longformer import LongformerAttention


def sinwave(N, decay):
    dt = 1/N*np.pi * (np.sqrt(N))
    x = np.linspace(1, N*dt*np.pi, num=N)
    y = np.cos(x-1) / x ** decay
    h_0 = np.sqrt(2) * y / np.sum(y)
    h_0.shape = (-1, 1)
    return h_0


def daubcqf(N):
    """
    Computes the Daubechies' scaling and wavelet filters (normalized to sqrt(2)).
    """
    assert N%2==0, 'No Daubechies filter exists for odd length'

    K = int(N/2)
    a = 1
    p = 1
    q = 1

    h_0 = np.array([1.0, 1.0])
    for j in range(1, K):
        a = -a * 0.25 * (j + K - 1)/j
        h_0 = np.hstack((0, h_0)) + np.hstack((h_0, 0))
        p = np.hstack((0, -p)) + np.hstack((p, 0))
        p = np.hstack((0, -p)) + np.hstack((p, 0))
        q = np.hstack((0, q, 0)) + a * p

    q = np.sort(np.roots(q))
    qt = q[:K-1]

    h_0 = np.convolve(h_0, np.real(np.poly(qt)))
    h_0 = np.sqrt(2) * h_0 / np.sum(h_0)

    h_0.shape = (-1, 1)
    h_0 = np.flip(h_0, axis=0);
    
    assert np.abs(np.sum(h_0 ** 2))-1 < 1e-4, 'Numerically unstable for this value of "N".'
    return h_0


@jax.jit
def parametrized_wavelet(thetas, S, S_inv):
    L = thetas.shape[0]
    wavelets = []
    
    #for dim in range(thetas.shape[1]):
    C = jnp.eye(N=L*2)[0, :]
    for theta in thetas[:]:
        A = jnp.array([[jnp.sin(theta), jnp.cos(theta)], [jnp.cos(theta), -jnp.sin(theta)]])
        R = sparse.BCOO.fromdense(jax.scipy.linalg.block_diag(*[A for _ in range(L)]), nse=4*L)
        C = C @ R @ S
        #C = jnp.matmul(C, S)
    C = jnp.matmul(C, S_inv)
    #wavelet = jnp.expand_dims(C, axis=-1)
    return C


def ortho_init(key, shape, dtype):
    def init(key, shape, dtype):
        thetas = nn.initializers.uniform(2*jnp.pi)(key, shape=(int(shape[0]/2),), dtype=dtype)
        return jnp.repeat(parametrized_wavelet(thetas), repeats=shape[1], axis=-1)
    return init(key, shape, dtype)


def sin_init(key, shape, dtype):
    def init(key, shape, dtype):
        wav_vec = jnp.asarray(sinwave(N=shape[0]))
        return jnp.repeat(wav_vec, repeats=shape[1], axis=-1)
    return init(key, shape, dtype)


def eye_init():
    def init(key, shape, dtype):
        return jnp.eye(N=shape[0], M=shape[1], dtype=dtype)
    return init


def db_init(key, shape, dtype):
    def init(key, shape, dtype):
        if shape[0] / 2 <= 20:
            return jnp.repeat(jnp.expand_dims(jnp.asarray(pywt.Wavelet('db{}'.format(int(shape[0] / 2))).filter_bank[0]), axis=-1), repeats=shape[1], axis=-1)
        else:
            #invoke debauchies calculation
            wav_vec = jnp.asarray(daubcqf(N=shape[0]))
            return jnp.repeat(wav_vec, repeats=shape[1], axis=-1)
    return init(key, shape, dtype)

class WavspaBlock(nn.Module):
    qkv_dim: int
    mlp_dim: int
    num_heads: int
    L: int
    max_len: int
    nb_features: int 
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    level: int = 2
    wlen: int = 32
    wav_dim: int = None
    poly_dim: int = 1
    wavelet: str = 'db2'
    model_type: str = 'transformer'
    
    def setup(self):
        ## db initialization
        assert self.wlen % 2 == 0, "incompatible"        
        self.eps = 1e-4
        if "lift" in self.wavelet:
            self.adawave_est = self.param('adawave_est', nn.initializers.normal(stddev=0.02), (self.wlen, self.wav_dim), self.dtype)
            self.adawave_pred = self.param('adawave_pred', nn.initializers.normal(stddev=0.02), (self.wlen, self.wav_dim), self.dtype)
        elif "ortho" in self.wavelet:
            L = int(self.wlen / 2)
            S = jnp.zeros(shape=[2*L, 2*L], dtype=int)
            i = jnp.asarray(range(2*L))
            j = jnp.asarray(range(1, 2*L+1)) % (2*L)
            S = S.at[i, j].set(1)
            self.S = sparse.BCOO.fromdense(S, nse=2*L)
            self.S_inv = jnp.linalg.inv(S)
            self.thetas = self.param('thetas', nn.initializers.uniform(2*jnp.pi), (L, self.wav_dim), self.dtype)
        elif "db" in self.wavelet:
            self.adawave = self.param('adawave', db_init, (self.wlen, self.wav_dim), self.dtype)
        elif "sin" in self.wavelet:
            self.adawave = self.param('adawave', sin_init, (self.wlen, self.wav_dim), self.dtype)
        else:
            # default to daubechie wave, non trainable
            self.adawave = db_init(key=None, shape=(self.wlen, self.wav_dim), dtype=self.dtype)
        
    
                             
    @nn.compact
    def __call__(self,
              inputs,
              inputs_segmentation=None,
              padding_mask=None,
              deterministic=False):
        
        # Attention block.
        assert inputs.ndim == 3
        
        inputs = jnp.where(padding_mask, inputs, 0)
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        
        # Multi Channel Expansion
        if "ortho" in self.wavelet:
            wavelet = jax.vmap(parametrized_wavelet, in_axes=(1, None, None), out_axes=1)(self.thetas, self.S, self.S_inv)
        elif "lift" not in self.wavelet:
            wavelet = self.adawave
        
        if "adalift" in self.wavelet:
            z = liftdec_learn(x, self.adawave_est, self.adawave_pred, level=self.level)
        elif "lift" in self.wavelet:
            z = wavspa.liftdec(x, level=self.level)
        else:
            z = wavspa.wavedec_learn(x, wavelet, level=self.level)
        
        for level in range(len(z)):
            conv_flag = False
            if z[level].shape[1] == 1:
                z[level] = jnp.squeeze(z[level], axis=1)
                conv_flag = True
            if self.model_type == 'transformer':
                z[level] = nn.SelfAttention(num_heads=self.num_heads,
                                            dtype=self.dtype,
                                            qkv_features=self.qkv_dim,
                                            kernel_init=nn.initializers.xavier_uniform(),
                                            bias_init=nn.initializers.normal(stddev=1e-6),
                                            use_bias=False,
                                            broadcast_dropout=False,
                                            dropout_rate=self.attention_dropout_rate,
                                            decode=False)(z[level], deterministic=deterministic)
            elif self.model_type == 'performer':
                z[level] = PerformerAttn(num_heads=self.num_heads, 
                                         qkv_dim=self.qkv_dim,
                                         lax_scan_unroll=16,
                                         nb_features=self.nb_features,
                                         dropout_rate=self.attention_dropout_rate,
                                         qkv_normalizarion=True)(z[level], deterministic=deterministic)
            elif self.model_type == 'linformer':
                z[level] = LinformerAttention(num_heads=self.num_heads, 
                                              qkv_features=self.qkv_dim, 
                                              max_len=self.max_len, 
                                              dropout_rate=self.attention_dropout_rate,
                                              low_rank_features=128)(z[level], deterministic=deterministic)
            elif self.model_type == 'linear_attention':
                z[level] = LinearAttention(num_heads=self.num_heads, qkv_features=self.qkv_dim)(z[level])
                z[level] = nn.Dropout(rate=self.dropout_rate)(z[level], deterministic=deterministic)
            elif self.model_type == 'longformer':
                z[level] = LongformerAttention(num_heads=self.num_heads, qkv_features=self.qkv_dim, sliding_window_size=512, 
                                           broadcast_dropout=False, bias=False, dropout_rate=self.attention_dropout_rate, 
                                           dtype=self.dtype)(z[level], deterministic=deterministic)
            else:
                raise NotImplementedError
            if conv_flag:
                z[level] = jnp.expand_dims(z[level], axis=1)
        if "adalift" in self.wavelet:
            z = liftrec_learn(z, self.adawave_est, self.adawave_pred)[:,:inputs.shape[1],:]
        elif "lift" in self.wavelet:
            z = wavspa.liftrec(z)[:,:inputs.shape[1],:]
        else:
            z = wavspa.waverec_learn(z, wavelet)[:,:inputs.shape[1],:]

        x = z + inputs
        y = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # MLP block.
        y = common_layers.MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate)(y, deterministic=deterministic)
        return x + y


class WavspaEncoder(nn.Module):
    """Transformer Model Encoder."""
    vocab_size: int
    nb_features: int = 256
    use_bfloat16: bool = False
    emb_dim: int = 512
    num_heads: int = 8
    poly_dim: int = 1
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
        
    wavelet: str = 'db2'
    level: int = 2
    wlen: int = 32
    h: int = 16
    model_type: str = 'transformer'
               
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
        for lyr in range(self.num_layers):
            x = WavspaBlock(
                wav_dim=self.emb_dim,
                qkv_dim=self.qkv_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                poly_dim = self.poly_dim,
                wavelet = self.wavelet,
                level = self.level,
                model_type=self.model_type,
                wlen = self.wlen,
                nb_features = self.nb_features,
                max_len = max_len,
                L = max_len,
                dtype=self.dtype,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                name=f'encoderblock_{lyr}')(x, inputs_segmentation=inputs_segmentation, 
                                            padding_mask=src_padding_mask, deterministic=not train)

        encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

        if self.classifier:
            encoded = common_layers.classifier_head(
                encoded, self.num_classes, self.mlp_dim, pooling_mode=self.classifier_pool)
        return encoded
    
    
class WavspaDualEncoder(nn.Module):
    """WavSpa Model Dual Encoder."""
    vocab_size: int
    use_bfloat16: bool = False
    emb_dim: int = 512
    num_heads: int = 8
    poly_dim: int = 1
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
    
    wavelet: str = 'db2'
    level: int = 2
    wlen: int = 32
        
    model_type: str = 'transformer'
        
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

        encoder = WavspaEncoder(
            vocab_size=self.vocab_size,
            use_bfloat16=self.use_bfloat16,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            poly_dim=self.poly_dim,
            num_layers=self.num_layers,
            qkv_dim=self.qkv_dim,
            mlp_dim=self.mlp_dim,
            max_len=self.max_len,
            wavelet=self.wavelet,
            level=self.level,
            wlen=self.wlen,
            model_type=self.model_type,
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

        
    
