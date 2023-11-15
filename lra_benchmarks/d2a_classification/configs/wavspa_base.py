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
    config.model_type = "hippoV"
    
    config.model = ml_collections.ConfigDict()
    config.model.level = 1
    config.model.wavelet = 'db2'
    
    #config.max_length = 3000
    config.model.num_layers = 8
    config.model.num_heads = 8
    config.model.emb_dim = 512
    config.model.dropout_rate = 0.5
    config.attention_dropout_rate = 0.5
    config.model.qkv_dim = 512
    config.model.mlp_dim = 768
    #config.classifier_pool = "CLS"
    #config.attention_dropout_rate = 0.6
    #config.dropout_rate = 0.3
    return config


def get_hyper(hyper):
    return hyper.product([])
