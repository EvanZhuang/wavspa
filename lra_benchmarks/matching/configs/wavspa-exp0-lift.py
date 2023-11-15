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

from lra_benchmarks.matching.configs import base_match_config
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = base_match_config.get_config()
    config.model_type = "wavspa_learn_dual"
    config.poly_dim = 1
    
    config.batch_size = 64
    config.warmup = 2500
    config.num_train_steps = 50000
    config.eval_frequency = 500
    
    config.emb_dim = 256
    config.num_heads = 1
    config.num_layers = 6
    config.qkv_dim = 128 
    config.mlp_dim = 256
    
    config.model = ml_collections.ConfigDict()
    config.model.level = 1
    config.model.wlen = 16
    config.model.wavelet = 'life'
    #config.num_heads = 4
    #config.num_layers = 4
    #config.warmup = 800
    #config.learning_rate = 0.5
    return config


def get_hyper(hyper):
    return hyper.product([])
