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

from lra_benchmarks.text_classification.configs import base_tc_config
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = base_tc_config.get_config()
    config.model_type = "transformer"
    
    config.batch_size = 4
    config.warmup = 6250
    config.num_train_steps = 2000
    config.eval_frequency = 500
    
    config.max_length = 4096
    config.emb_dim = 256
    config.num_heads = 1
    config.num_layers = 6
    config.qkv_dim = 256
    config.mlp_dim = 1024

    config.model = ml_collections.ConfigDict()
    return config


def get_hyper(hyper):
    return hyper.product([])
