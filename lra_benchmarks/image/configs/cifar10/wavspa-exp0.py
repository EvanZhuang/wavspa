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

"""Configuration and hyperparameter sweeps."""

from lra_benchmarks.image.configs.cifar10 import base_cifar10_config


def get_config():
    """Get the hyperparameter configuration."""
    config = base_cifar10_config.get_config()
    config.model_type = "wavspa_learn"
    config.poly_dim = 1
    
    config.batch_size = 400
    config.warmup = 10000
    config.num_train_steps = 200000
    config.eval_frequency = 500
    
    
    config.model.num_layers = 1
    config.model.emb_dim = 128
    config.model.qkv_dim = 64
    config.model.mlp_dim = 128
    config.model.num_heads = 8
    
    config.model.classifier_pool = "CLS"
    config.model.level = 1
    config.model.wlen = 40
    config.model.wavelet = 'db2'
    config.model.nb_features = 512
    return config


def get_hyper(hyper):
    return hyper.product([])
