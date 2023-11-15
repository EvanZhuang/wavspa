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

from lra_benchmarks.listops.configs import base_listops_config
import ml_collections

def get_config():
    """Get the default hyperparameter configuration."""
    config = base_listops_config.get_config()
    config.model_type = "wavspa_learn"
    config.pooling_mode = "MEAN"
    
    config.batch_size = 800
    config.warmup = 10000
    config.num_train_steps = 40000
    config.eval_frequency = 100
    
    config.emb_dim = 128
    config.num_heads = 1
    config.num_layers = 8
    config.qkv_dim = 64
    config.mlp_dim = 128
    
    
    config.model = ml_collections.ConfigDict()
    config.model.level = 1
    config.model.wlen = 16
    config.model.wavelet = 'db2'
    return config


def get_hyper(hyper):
    return hyper.product([])