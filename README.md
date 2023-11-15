## WavSpA: Wavelet Space Attention for Enhancing Transformer's Long Sequence Learning

Welcome to the official repository of the [WavSpA](https://arxiv.org/abs/2210.01989) paper. This innovative work introduces adaptive wavelet transform techniques coupled with Transformer models to excel at processing long sequences. The implementation is crafted using [JAX](https://github.com/google/jax) alongside [Flax](https://github.com/google/flax) for robustness and efficiency.

### Installation

Setup your environment to run the models using the provided `requirements.txt`:

```bash
$ pip install -r requirements.txt
$ pip install wavspa
```
Note: This codebase supports JAX version 0.3.13.

#### Our experiments

We demonstrate substantial gains in performance with WavSpA across various attention-based architectures. The framework offers three parametrization options:

* Adaptive Wavelet (AdaWavSpA)
* Orthogonal Adaptive Wavelet (OrthoWavSpA)
* Wavelet Lifting (LiftWavSpA)

Performance metrics are as follows:

Models | ListOps | Text | Retrieval | Image | Pathfinder | Avg | Avg (w/o r)
--- | --- | --- | --- | --- | --- | --- | ---
**Transformer** | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | 54.39 | 53.62
AdaWavSpA | 55.40 | 81.60 | 79.27 | 55.58 | 81.12 | 70.59 | 68.43
OrthoWavSpA | 45.95 | 81.63 | 71.52 | 49.29 | 81.13 | 65.90 | 64.50
LiftWavSpA | 42.95 | 75.63 | 56.45 | 42.48 | 81.73 | 59.85 | 60.70
--- | --- | --- | --- | --- | --- | --- | ---
**Longformer** | 35.63 | 62.85 | 56.89 | 42.22 | 69.71 | 53.46 | 52.60
AdaWavSpA | 49.30 | 79.73 | 58.57 | 50.84 | 79.48 | 63.66 | 64.93
OrthoWavSpA | 39.45 | 78.41 | 79.93 | 49.93 | 79.47 | 54.96 | 54.96
LiftWavSpA | 39.40 | 78.00 | 53.27 | 40.95 | 75.80 | 57.48 | 58.54
--- | --- | --- | --- | --- | --- | --- | ---
**Linformer** | 35.70 | 53.94 | 52.27 | 38.47 | 66.44 | 49.36 | 48.64
AdaWavSpA | 37.15 | 54.75 | 61.09 | 34.93 | 65.66 | 50.72 | 48.12
OrthoWavSpA | 38.05 | 56.93 | 60.25 | 39.45 | 65.35 | 52.01 | 49.95
LiftWavSpA | 37.30 | 54.43 | 70.73 | 34.66 | 63.49 | 52.12 | 47.47
--- | --- | --- | --- | --- | --- | --- | ---
**Linear Att.** | 16.13 | 65.90 | 53.09 | 42.32 | 75.91 | 50.67 | 50.06
AdaWavSpA | 38.90 | 76.82 | 71.38 | 54.81 | 79.68 | 64.32 | 62.55
OrthoWavSpA | 39.55 | 79.45 | 69.65 | 49.93 | 78.09 | 55.86 | 55.86
LiftWavSpA | 38.35 | 73.39 | 54.06 | 44.39 | 74.46 | 56.93 | 57.65
--- | --- | --- | --- | --- | --- | --- | ---
**Performer** | 18.01 | 65.40 | 53.82 | 42.77 | 77.05 | 51.41 | 50.81
AdaWavSpA | 46.05 | 80.93 | 71.16 | 52.06 | 77.17 | 65.47 | 64.05
OrthoWavSpA | 39.80 | 79.10 | 57.67 | 48.78 | 78.09 | 60.69 | 61.44
LiftWavSpA | 39.85 | 75.96 | 52.75 | 39.97 | 76.20 | 56.95 | 58.00

## Example Usage

For implementation details, see lra_benchmarks/models/wavspa/wavspa_learn.py. The wavelet initialization and transformation processes are crucial:
```
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
```

Then for forward and backward wavelet transform:
```
z = wavspa.wavedec_learn(x, wavelet, level=self.level)
for level in range(len(z)):
  z[level] = nn.SelfAttention(num_heads=self.num_heads,
                              dtype=self.dtype,
                              qkv_features=self.qkv_dim,
                              kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=nn.initializers.normal(stddev=1e-6),
                              use_bias=False,
                              broadcast_dropout=False,
                              dropout_rate=self.attention_dropout_rate,
                              decode=False)(z[level], deterministic=deterministic)
z = wavspa.waverec_learn(z, wavelet)[:,:inputs.shape[1],:]

```

## Datasets

Access and instructions for LRA, D2A, and CodeXGlue datasets:

* [Long Range Arena](https://github.com/google-research/long-range-arena)
* [D2A IBM official release](https://developer.ibm.com/exchanges/data/all/d2a/)
* [CodeXGlue](https://github.com/microsoft/CodeXGLUE)

To execute a task, use the train_best.py script with the appropriate configurations:

```
PYTHONPATH="$(pwd)":"$PYTHONPATH" python lra_benchmarks/listops/train_best.py \
      --config=lra_benchmarks/listops/configs/wavspa-exp0.py \
      --model_dir=/tmp/listops \
      --task_name=basic \
      --data_dir=$HOME/lra_data/listops/
```

## Citation

If you find out work useful, please cite our paper at:

```
@inproceedings{
zhuang2023wavspa,
title={WavSpA: Wavelet Space Attention for Boosting Transformers' Long Sequence Learning Ability},
author={Yufan Zhuang and Zihan Wang and Fangbo Tao and Jingbo Shang},
booktitle={UniReps:  the First Workshop on Unifying Representations in Neural Models at NeurIPS 2023},
year={2023},
url={https://openreview.net/forum?id=yC6b3hqyf8}
}
```