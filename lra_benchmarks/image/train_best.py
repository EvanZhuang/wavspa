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

"""Main training script for the image classification task."""

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
from lra_benchmarks.image import task_registry
from lra_benchmarks.utils import train_utils

from ml_collections import config_flags
import tensorflow.compat.v2 as tf

from jax.config import config; config.update("jax_enable_x64", False)


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string('task_name', default='mnist', help='Name of the task')
flags.DEFINE_bool(
    'eval_only', default=False, help='Run the evaluation on the test data.')
flags.DEFINE_string(
    'override_wavelet_basis', default=None, help='Alternative Wavelet Basis')
flags.DEFINE_integer(
    'override_decomposition_level', default=None, help='Alternative Decomposition Level')
flags.DEFINE_integer(
    'override_seed', default=None, help='Alternative Seed')


def create_model(flax_module, model_kwargs):
    """Creates and initializes the model."""

    def _create_model():
        module = flax_module(**model_kwargs)
        return module

    return _create_model()


def compute_metrics(logits, labels, num_classes, weights):
    """Compute summary metrics."""
    loss, weight_sum = train_utils.compute_weighted_cross_entropy(
        logits, labels, num_classes, weights=weights)
    acc, _ = train_utils.compute_weighted_accuracy(logits, labels, weights)
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'denominator': weight_sum,
    }
    metrics = jax.lax.psum(metrics, 'batch')
    return metrics


def train_step(state,
               batch,
               model,
               learning_rate_fn,
               num_classes,
               flatten_input=True,
               grad_clip_norm=None,
               dropout_rng=None):
    """Perform a single training step."""
    train_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in train_keys]
    if flatten_input:
        inputs = inputs.reshape(inputs.shape[0], -1)

    # We handle PRNG splitting inside the top pmap, rather
    # than handling it outside in the training loop - doing the
    # latter can add some stalls to the devices.
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        """Loss function used for training."""
        logits = model.apply({'params': params}, 
                             inputs,
                             train=True,
                             rngs={'dropout': dropout_rng})
        loss, weight_sum = train_utils.compute_weighted_cross_entropy(
            logits, targets, num_classes=num_classes, weights=None)
        mean_loss = loss / weight_sum
        return (mean_loss, logits)
    
    step = state.step
    lr = learning_rate_fn(step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    ((_, logits), grads) = grad_fn(state.params)
    grads = jax.lax.pmean(grads, 'batch')
    if grad_clip_norm:
        # Optionally resize the global gradient to a maximum norm.
        gradients, _ = jax.tree_flatten(grads)
        g_l2 = jnp.sqrt(sum([jnp.vdot(p, p) for p in gradients]))
        g_factor = jnp.minimum(1.0, grad_clip_norm / g_l2)
        grads = jax.tree_map(lambda p: g_factor * p, grads)
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, targets, num_classes, None)
    metrics['learning_rate'] = lr
    return (new_state, metrics)


def eval_step(model, state, batch, num_classes, flatten_input=True):
    eval_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in eval_keys]
    if flatten_input:
        inputs = inputs.reshape(inputs.shape[0], -1)
    logits = model.apply({"params": state.params}, inputs, train=False)
    return compute_metrics(logits, targets, num_classes, weights=None)


def test(state, p_eval_step, step, test_ds, summary_writer,
         model_dir):
    """Test the flax module in optimizer on test_ds.

    Args:
      optimizer: flax optimizer (contains flax module).
      state: model state, e.g. batch statistics.
      p_eval_step: fn; Pmapped evaluation step function.
      step: int; Number of training steps passed so far.
      test_ds: tf.dataset; Test dataset.
      summary_writer: tensorflow summary writer.
      model_dir: model directory.
    """
    # Test Metrics
    test_metrics = []
    test_iter = iter(test_ds)
    for _, test_batch in zip(itertools.repeat(1), test_iter):
        # pylint: disable=protected-access
        test_batch = common_utils.shard(
            jax.tree_map(lambda x: x._numpy(), test_batch))
        # pylint: enable=protected-access
        metrics = p_eval_step(state=state, batch=test_batch)
        test_metrics.append(metrics)
    test_metrics = common_utils.get_metrics(test_metrics)
    test_metrics_sums = jax.tree_map(jnp.sum, test_metrics)
    test_denominator = test_metrics_sums.pop('denominator')
    test_summary = jax.tree_map(
        lambda x: x / test_denominator,  # pylint: disable=cell-var-from-loop
        test_metrics_sums)
    logging.info('test in step: %d, loss: %.4f, acc: %.4f', step,
                 test_summary['loss'], test_summary['accuracy'])
    if jax.process_index() == 0:
        for key, val in test_summary.items():
            summary_writer.scalar(f'test_{key}', val, step)
        summary_writer.flush()
    with tf.io.gfile.GFile(os.path.join(model_dir, 'results.json'), 'w') as f:
        json.dump(jax.tree_map(lambda x: x.tolist(), test_summary), f)


def train_loop(config, dropout_rngs, eval_ds, eval_freq, num_eval_steps,
               num_train_steps, state, p_eval_step, p_train_step,
               start_step, train_iter, summary_writer):
    """Training loop.

    Args:
      config: experiment config.
      dropout_rngs: float array; Jax PRNG key.
      eval_ds: tf.dataset; Evaluation dataset.
      eval_freq: int; Evaluation frequency;
      num_eval_steps: int; Number of evaluation steps.
      num_train_steps: int; Number of training steps.
      optimizer: flax optimizer.
      state: model state, e.g. batch statistics.
      p_eval_step: fn; Pmapped evaluation step function.
      p_train_step: fn; Pmapped train step function.
      start_step: int; global training step.
      train_iter: iter(tf.dataset); Training data iterator.
      summary_writer: tensorflow summary writer.

    Returns:
      optimizer, global training step
    """
    metrics_all = []
    tick = time.time()
    patience = int(0.1 * num_train_steps)
    patience_ctr = 0
    min_step = 50000
    logging.info('Starting training')
    logging.info('====================')

    step = 0
    eval_cur = 0
    for step, batch in zip(range(start_step, num_train_steps), train_iter):
        batch = common_utils.shard(jax.tree_map(
            lambda x: x._numpy(), batch))  # pylint: disable=protected-access
        state, metrics = p_train_step(state=state, batch=batch, dropout_rng=dropout_rngs)
        
        metrics_all.append(metrics)
        logging.info('train in step: %d, best eval metric: %.4f', step, eval_cur)
        patience_ctr += 1
        '''
        # Save a Checkpoint
        if ((step % config.checkpoint_freq == 0 and step > 0) or
                step == num_train_steps - 1):
            if jax.process_index() == 0 and config.save_checkpoints:
                # Save unreplicated optimizer + model state.
                checkpoints.save_checkpoint(
                    FLAGS.model_dir,
                    jax_utils.unreplicate(state),
                    step)
        '''
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
            logging.info('train in step: %d, loss: %.4f, acc: %.4f', step,
                         summary['loss'], summary['accuracy'])
            if jax.process_index() == 0:
                tock = time.time()
                steps_per_sec = eval_freq / (tock - tick)
                tick = tock
                summary_writer.scalar('examples_per_second',
                                      steps_per_sec * config.batch_size, step)
                for key, val in summary.items():
                    summary_writer.scalar(f'train_{key}', val, step)
                summary_writer.flush()
            # Reset metric accumulation for next evaluation cycle.
            metrics_all = []

            # Eval Metrics
            eval_metrics = []
            eval_iter = iter(eval_ds)
            if num_eval_steps == -1:
                num_iter = itertools.repeat(1)
            else:
                num_iter = range(num_eval_steps)
            for _, eval_batch in zip(num_iter, eval_iter):
                # pylint: disable=protected-access
                eval_batch = common_utils.shard(
                    jax.tree_map(lambda x: x._numpy(), eval_batch))
                # pylint: enable=protected-access
                metrics = p_eval_step(state=state, batch=eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
            eval_denominator = eval_metrics_sums.pop('denominator')
            eval_summary = jax.tree_map(
                lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
                eval_metrics_sums)
            logging.info('eval in step: %d, loss: %.4f, acc: %.4f', step,
                         eval_summary['loss'], eval_summary['accuracy'])
            if jax.process_index() == 0:
                for key, val in eval_summary.items():
                    summary_writer.scalar(f'val_{key}', val, step)
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
    #dump final checkpoint
    if jax.process_index() == 0 and config.save_checkpoints:
        # Save unreplicated optimizer + model state.
        checkpoints.save_checkpoint(FLAGS.model_dir,
                                    jax_utils.unreplicate(state), 
                                    step, overwrite=True, 
                                    prefix="final_checkpoint_")
    return state, step


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
    
    if FLAGS.override_seed is not None:
        random_seed = FLAGS.override_seed

    if jax.process_index() == 0:
        summary_writer = tensorboard.SummaryWriter(
            os.path.join(FLAGS.model_dir, 'summary'))
    else:
        summary_writer = None

    if batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')

    logging.info('Training on %s', FLAGS.task_name)

    if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
        normalize = True
    else:  # transformer-based models
        normalize = False
    (train_ds, eval_ds, test_ds, num_classes, vocab_size,
     input_shape) = task_registry.TASK_DATA_DICT[FLAGS.task_name](
         n_devices=jax.local_device_count(),
         batch_size=batch_size,
         normalize=normalize)
    train_iter = iter(train_ds)
    model_kwargs = {}
    flatten_input = True

    if model_type in ['wideresnet', 'resnet', 'simple_cnn']:
        model_kwargs.update({
            'num_classes': num_classes,
        })
        flatten_input = False

    else:  # transformer models
        # we will flatten the input
        bs, h, w, c = input_shape
        assert c == 1
        input_shape = (bs, h * w * c)
        model_kwargs.update({
            'vocab_size': vocab_size,
            'max_len': input_shape[1],
            'classifier': True,
            'num_classes': num_classes,
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
        factors=config.factors,
        base_learning_rate=learning_rate,
        warmup_steps=config.warmup,
        steps_per_cycle=config.get('steps_per_cycle', None),
    )

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
    if config.restore_checkpoints:
        # Restore unreplicated optimizer + model state from last checkpoint.
        state = checkpoints.restore_checkpoint(FLAGS.model_dir, state)
        # Grab last step.
        start_step = int(state.step)

    # Replicate optimizer and state
    state = jax_utils.replicate(state)
    
    p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          num_classes=num_classes,
          grad_clip_norm=config.get('grad_clip_norm', None),
          flatten_input=flatten_input),
      axis_name="batch",
      donate_argnums=(0,))  # pytype: disable=wrong-arg-types

    p_eval_step = jax.pmap(
        functools.partial(
            eval_step, model=model, num_classes=num_classes, flatten_input=flatten_input),
        axis_name='batch',
    )

    state, step = train_loop(config, dropout_rngs, eval_ds, eval_freq,
                             num_eval_steps, num_train_steps,
                             state, p_eval_step,
                             p_train_step, start_step, train_iter,
                             summary_writer)

    logging.info('Starting testing')
    logging.info('====================')
    state = checkpoints.restore_checkpoint(FLAGS.model_dir, target=jax_utils.unreplicate(state))
    # Replicate state.
    state = jax_utils.replicate(state)
        
    test(state, p_eval_step, step, test_ds, summary_writer,
         FLAGS.model_dir)


if __name__ == '__main__':
    app.run(main)
