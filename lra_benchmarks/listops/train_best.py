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
"""Main training script for the listops task."""
import functools
import itertools
import json
import os
import time

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import linen as nn

from flax.metrics import tensorboard
from flax.training import checkpoints, train_state
from flax.training import common_utils
import optax
import jax
from jax import random
import jax.nn
import jax.numpy as jnp
from lra_benchmarks.listops import input_pipeline
from lra_benchmarks.utils import train_utils
from ml_collections import config_flags
import numpy as np
import tensorflow.compat.v2 as tf
import copy

from jax.config import config; config.update("jax_enable_x64", False)


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string(
    'task_name',
    default='basic',
    help='Name of the task used for load training/test data.')
flags.DEFINE_string(
    'data_dir', default=None, help='Directory containing datasets.')
flags.DEFINE_bool(
    'test_only', default=False, help='Run the evaluation on the test data.')
flags.DEFINE_string(
    'override_wavelet_basis', default=None, help='Alternative Wavelet Basis')
flags.DEFINE_integer(
    'override_decomposition_level', default=None, help='Alternative Decomposition Level')
flags.DEFINE_integer(
    'override_seed', default=None, help='Alternative Seed')


def create_model(flax_module, model_kwargs):
    """Creates and initializes the model."""

    #@functools.partial(jax.jit, backend='cpu')
    def _create_model():
        #import pdb
        # pdb.set_trace()
        #model_static_kwargs = copy.deepcopy(model_kwargs)
        #model_static_kwargs.update({"train": False})
        module = flax_module(**model_kwargs)
        #module = functools.partial(flax_module, **model_kwargs)
        #module = flax_module.__setattr__(**model_kwargs)
        # with nn.stochastic(key):
        return module

    return _create_model()


def compute_metrics(logits, labels, weights):
    """Compute summary metrics."""
    loss, weight_sum = train_utils.compute_weighted_cross_entropy(
        logits, labels, num_classes=10, weights=weights)
    acc, _ = train_utils.compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'denominator': weight_sum,
    }
    metrics = jax.lax.psum(metrics, 'batch')
    return metrics


def train_step(
    state,
    batch,
    model,
    learning_rate_fn,
    label_smoothing=0.0,
    dropout_rng=None):
    """Perform a single training step."""

    # X_position and X_segmentation are needed only when using "packed examples"
    # where multiple sequences are packed into the same example with this
    # metadata.
    # if such features are not present they are ignored and the example is treated
    # like a normal, unpacked sequence example.
    
    train_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in train_keys]

    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """loss function used for training."""
        #model = train_utils.get_model(model_type, create_model, model_kwargs)
        logits = model.apply({'params': params}, 
                             inputs,
                             train=True,
                             rngs={'dropout': dropout_rng})

        (loss, weight_sum) = train_utils.compute_weighted_cross_entropy(logits, targets, num_classes=10, weights=None)
        mean_loss = loss / weight_sum
        return (mean_loss, logits)

    step = state.step
    lr = learning_rate_fn(step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    ((_, logits), grads) = grad_fn(state.params)
    grads = jax.lax.pmean(grads, 'batch')
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, targets, None)
    metrics['learning_rate'] = lr

    return (new_state, metrics)


def eval_step(params, batch, model):
    """Calculate evaluation metrics on a batch."""
    eval_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in eval_keys]
    #model = train_utils.get_model(model_type, create_model, model_kwargs)
    logits = model.apply({"params": params}, inputs, train=False)
    return compute_metrics(logits, targets, None)


'''
def train_step(optimizer, batch, learning_rate_fn, dropout_rng=None):
    """Perform a single training step."""
    train_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in train_keys]

    # We handle PRNG splitting inside the top pmap, rather
    # than handling it outside in the training loop - doing the
    # latter can add some stalls to the devices.
    dropout_rng, new_dropout_rng = random.split(dropout_rng)

    def loss_fn(model):
        """Loss function used for training."""
        with nn.stochastic(dropout_rng):
            logits = model(inputs, train=True)
        loss, weight_sum = train_utils.compute_weighted_cross_entropy(
            logits, targets, num_classes=10, weights=None)
        mean_loss = loss / weight_sum
        return mean_loss, logits

    step = optimizer.state.step
    lr = learning_rate_fn(step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(optimizer.target)
    grad = jax.lax.pmean(grad, 'batch')
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    metrics = compute_metrics(logits, targets, None)
    metrics['learning_rate'] = lr

    return new_optimizer, metrics, new_dropout_rng


def eval_step(model, batch):
    eval_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in eval_keys]
    logits = model(inputs, train=False)
    return compute_metrics(logits, targets, None)
'''

def tohost(x):
    """Collect batches from all devices to host and flatten batch dimensions."""
    n_device, n_batch, *remaining_dims = x.shape
    return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tf.enable_v2_behavior()

    config = FLAGS.config
    logging.info('===========Config Dict============')
    logging.info(config)
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    num_train_steps = config.num_train_steps
    num_eval_steps = config.num_eval_steps
    eval_freq = config.eval_frequency
    random_seed = config.random_seed
    model_type = config.model_type
    model_kwargs = (config.model_kwargs.to_dict() if 'model_kwargs' in config else {})
    
    if FLAGS.override_seed is not None:
        random_seed = FLAGS.override_seed

    if jax.process_index() == 0:
        summary_writer = tensorboard.SummaryWriter(
            os.path.join(FLAGS.model_dir, 'summary'))

    if batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')

    train_ds, eval_ds, test_ds, encoder = input_pipeline.get_datasets(
        n_devices=jax.local_device_count(),
        task_name=FLAGS.task_name,
        data_dir=FLAGS.data_dir,
        batch_size=batch_size,
        max_length=config.max_length)

    vocab_size = encoder.vocab_size
    train_ds = train_ds.repeat()
    train_iter = iter(train_ds)
    max_length = config.max_length
    input_shape = (batch_size, max_length)

    model_kwargs.update({
        'vocab_size': vocab_size,
        'emb_dim': config.emb_dim,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'qkv_dim': config.qkv_dim,
        'mlp_dim': config.mlp_dim,
        'max_len': config.max_length,
        'classifier': True,
        'num_classes': 10
    })
    model_kwargs.update(config.model)
    
    if FLAGS.override_wavelet_basis is not None:
        model_kwargs['wavelet'] = FLAGS.override_wavelet_basis
    if FLAGS.override_decomposition_level is not None:
        model_kwargs['level'] = FLAGS.override_decomposition_level
    logging.info(model_kwargs)

    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, jax.local_device_count())

    model = train_utils.get_model(model_type, create_model, model_kwargs)
    
    initial_variables = jax.jit(model.init)(init_rng, 
                                            jnp.ones(input_shape, jnp.float32))

    learning_rate_fn = train_utils.create_learning_rate_scheduler(
        base_learning_rate=learning_rate)

    optimizer = optax.adamw(
      learning_rate_fn, b1=0.9, b2=0.98, eps=1e-9,
      weight_decay=FLAGS.config.weight_decay
    )
    
    state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=initial_variables["params"],
      tx=optimizer
      )
    # We access model params only from optimizer below.
    del initial_variables
    
    start_step = 0
    if config.restore_checkpoints or FLAGS.test_only:
        # Restore unreplicated optimizer + model state from last checkpoint.
        state = checkpoints.restore_checkpoint(FLAGS.model_dir, state)
        # Grab last step.
        start_step = int(state.step)

    # Replicate state.
    state = jax_utils.replicate(state)
    
    p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn),
      axis_name="batch",
      donate_argnums=(0,))  # pytype: disable=wrong-arg-types
    
    p_eval_step = jax.pmap(
        functools.partial(eval_step, model=model),
        axis_name="batch")


    # p_pred_step = jax.pmap(predict_step, axis_name='batch')

    def run_eval(params, eval_ds, num_eval_steps=-1):
        eval_metrics = []
        eval_iter = iter(eval_ds)
        if num_eval_steps == -1:
            num_iter = itertools.count()
        else:
            num_iter = range(num_eval_steps)
        for _, eval_batch in zip(num_iter, eval_iter):
            # pylint: disable=protected-access
            eval_batch = common_utils.shard(
                jax.tree_map(lambda x: x._numpy(), eval_batch))
            # pylint: enable=protected-access
            metrics = p_eval_step(params, eval_batch)
            eval_metrics.append(metrics)
        eval_metrics = common_utils.get_metrics(eval_metrics)
        eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
        eval_denominator = eval_metrics_sums.pop('denominator')
        eval_summary = jax.tree_map(
            lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
            eval_metrics_sums)
        # Calculate (clipped) perplexity after averaging log-perplexities:
        eval_summary['perplexity'] = jnp.clip(
            jnp.exp(eval_summary['loss']), a_max=1.0e4)
        return eval_summary

    if FLAGS.test_only:
        with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, 'results.json'),
                               'w') as f:
            test_summary = run_eval(state.params, test_ds)
            json.dump(jax.tree_map(lambda x: x.tolist(), test_summary), f)
        return

    metrics_all = []
    eval_cur = 0
    tick = time.time()
    patience = int(0.1 * num_train_steps)
    patience_ctr = 0
    min_step = 5000
    for step, batch in zip(range(start_step, num_train_steps), train_iter):
        #rng, dropout_rngs = jax.random.split(rng)
        batch = common_utils.shard(jax.tree_map(
            lambda x: x._numpy(), batch))  # pylint: disable=protected-access
        state, metrics = p_train_step(state, batch, dropout_rng=dropout_rngs)
        
        #optimizer, metrics, dropout_rngs = p_train_step(
        #    optimizer, batch, dropout_rng=dropout_rngs)
        metrics_all.append(metrics)
        logging.info('train in step: %d, best eval metric: %.4f', step, eval_cur)
        patience_ctr += 1

        # Periodic metric handling.
        if step % eval_freq == 0 and step > 0:
            metrics_all = common_utils.get_metrics(metrics_all)
            lr = metrics_all.pop('learning_rate').mean()
            metrics_sums = jax.tree_map(jnp.sum, metrics_all)
            denominator = metrics_sums.pop('denominator')
            summary = jax.tree_map(
                lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
            summary['learning_rate'] = lr
            # Calculate (clipped) perplexity after averaging log-perplexities:
            summary['perplexity'] = jnp.clip(
                jnp.exp(summary['loss']), a_max=1.0e4)
            logging.info('train in step: %d, loss: %.4f',
                         step, summary['loss'])
            if jax.process_index() == 0:
                tock = time.time()
                steps_per_sec = eval_freq / (tock - tick)
                tick = tock
                summary_writer.scalar('steps per second', steps_per_sec, step)
                for key, val in summary.items():
                    summary_writer.scalar(f'train_{key}', val, step)
                summary_writer.flush()
            # Reset metric accumulation for next evaluation cycle.
            metrics_all = []

            # Eval Metrics
            eval_summary = run_eval(state.params, eval_ds, num_eval_steps)
            logging.info('eval in step: %d, loss: %.4f, acc: %.4f',
                         step, eval_summary['loss'], eval_summary['accuracy'])
            if jax.process_index() == 0:
                for key, val in eval_summary.items():
                    summary_writer.scalar(f'eval_{key}', val, step)
                summary_writer.flush()
            
            # Save the best
            if eval_summary['accuracy'] > eval_cur:
                patience_ctr = 0
                eval_cur = eval_summary['accuracy']
                
                # Save a Checkpoint
                if jax.process_index() == 0:
                    # Save unreplicated optimizer + model state.
                    checkpoints.save_checkpoint(FLAGS.model_dir,
                                                jax_utils.unreplicate(state), step, overwrite=True)
        if patience_ctr > patience and step > min_step:
            ## Patience Reached
            break
                
                
    #dump test results
    if jax.process_index() == 0 and config.save_checkpoints:
        # Save unreplicated optimizer + model state.
        checkpoints.save_checkpoint(FLAGS.model_dir,
                                    jax_utils.unreplicate(state), 
                                    step, overwrite=True, 
                                    prefix="final_checkpoint_")
        
    with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, 'results.json'), 'w') as f:
        state = checkpoints.restore_checkpoint(FLAGS.model_dir, target=jax_utils.unreplicate(state))
        # Replicate state.
        state = jax_utils.replicate(state)
        test_summary = run_eval(state.params, test_ds)
        json.dump(jax.tree_map(lambda x: x.tolist(), test_summary), f)
        logging.info('test in step: %d, loss: %.4f, acc: %.4f',
                         step, test_summary['loss'], test_summary['accuracy'])


if __name__ == '__main__':
    app.run(main)
