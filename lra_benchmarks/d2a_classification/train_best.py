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
"""Document Classification tasks."""
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
from lra_benchmarks.d2a_classification import input_pipeline
from lra_benchmarks.utils import train_utils
from ml_collections import config_flags
import tensorflow.compat.v2 as tf

from sklearn.utils import class_weight
import numpy as onp

from jax.config import config

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string(
    'model_dir', default=None, help='Directory to store model data.')
flags.DEFINE_string(
    'task_name',
    default='basic_two_ptrs',
    help='Directory to store model data.')
flags.DEFINE_string(
    'data_dir', default=None, help='Directory containing datasets.')
flags.DEFINE_bool(
    'test_only', default=False, help='Run the evaluation on the test data.')

CLASS_MAP = {'imdb_reviews': 2, 'd2a_function':2, 'd2a_trace': 2, 'codexglue_defect': 2}


def create_model(flax_module, model_kwargs):
    """Creates and initializes the model."""

    def _create_model():
        module = flax_module(**model_kwargs)
        return module

    return _create_model()



def compute_metrics(logits, labels, weights):
    """Compute summary metrics."""
    if FLAGS.task_name in CLASS_MAP.keys():
        num_classes = CLASS_MAP[FLAGS.task_name]
    else:
        num_classes = 2
    loss, weight_sum = train_utils.compute_weighted_cross_entropy(
        logits, labels, num_classes=num_classes, weights=weights)
    acc, _ = train_utils.compute_weighted_accuracy(logits, labels, None)
    tn, fp, fn, tp, _ = train_utils.compute_f1_score(logits, labels)
    metrics = {
        'loss': loss,
        'accuracy': acc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'denominator': weight_sum,
    }
    metrics = jax.lax.psum(metrics, 'batch')
    return metrics


def train_step(state, batch, model, learning_rate_fn, class_weights, dropout_rng=None):
    """Perform a single training step."""
    train_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in train_keys]

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
            logits, targets, num_classes=CLASS_MAP[FLAGS.task_name], weights=class_weights)
        mean_loss = loss / weight_sum
        return mean_loss, logits

    step = state.step
    lr = learning_rate_fn(step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    ((_, logits), grads) = grad_fn(state.params)
    grads = jax.lax.pmean(grads, 'batch')
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, targets, class_weights)
    metrics['learning_rate'] = lr

    return (new_state, metrics)


def eval_step(params, batch, model, class_weights):
    eval_keys = ['inputs', 'targets']
    (inputs, targets) = [batch.get(k, None) for k in eval_keys]
    logits = model.apply({"params": params}, inputs, train=False)
    logging.info(logits)
    return compute_metrics(logits, targets, class_weights)


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

    max_length = config.max_length

    if jax.process_index() == 0:
        summary_writer = tensorboard.SummaryWriter(
            os.path.join(FLAGS.model_dir, 'summary'))

    if batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')

    train_ds, eval_ds, test_ds, encoder = input_pipeline.get_tc_datasets(
        n_devices=jax.local_device_count(),
        task_name=FLAGS.task_name,
        data_dir=FLAGS.data_dir,
        batch_size=batch_size,
        fixed_vocab=None,
        max_length=max_length,
        tokenizer="gpt2")
    
    vocab_size = encoder.vocab_size + 10
    logging.info('Vocab Size: %d', vocab_size)
    
    class_1 = 0
    class_ctr = 0
    for tbatch in iter(train_ds): 
        class_1 += sum(tbatch['targets']).numpy()
        class_ctr += tbatch['targets'].shape[0]
    
    class_0 = class_ctr - class_1
    class_weights = jnp.array([class_ctr/(2*class_0), class_ctr/(2*class_1)])
    class_weights = class_weights / jnp.sum(class_weights)
    print('Class Weight:', class_weights)
    train_ds = train_ds.repeat()

    train_iter = iter(train_ds)
    input_shape = (batch_size, max_length)

    model_kwargs = {
        'vocab_size': vocab_size,
        'emb_dim': config.emb_dim,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'qkv_dim': config.qkv_dim,
        'mlp_dim': config.mlp_dim,
        'max_len': max_length,
        'classifier': True,
        'num_classes': CLASS_MAP[FLAGS.task_name],
        'classifier_pool': config.classifier_pool
    }
    model_kwargs.update(config.model)
    
    rng = random.PRNGKey(random_seed)
    rng = jax.random.fold_in(rng, jax.process_index())
    rng, init_rng = random.split(rng)
    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, jax.local_device_count())

    # Init model
    model = train_utils.get_model(model_type, create_model, model_kwargs)
    
    initial_variables = jax.jit(model.init)(init_rng, 
                                            jnp.ones(input_shape, jnp.float32))

    learning_rate_fn = train_utils.create_learning_rate_scheduler(
        factors=config.factors,
        base_learning_rate=learning_rate,
        warmup_steps=config.warmup)

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
        functools.partial(train_step, model=model, learning_rate_fn=learning_rate_fn, class_weights=None),
        axis_name='batch', donate_argnums=(0,))
    p_eval_step = jax.pmap(functools.partial(eval_step, model=model, class_weights=None),
                           axis_name='batch')
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
        # Compute F1
        eval_summary['f1'] = eval_summary['tp'] / (eval_summary['tp'] + 0.5 * (eval_summary['fp'] + eval_summary['fn']))
        return eval_summary

    if FLAGS.test_only:
        with tf.io.gfile.GFile(os.path.join(FLAGS.model_dir, 'results.json'),
                               'w') as f:
            test_summary = run_eval(test_ds)
            json.dump(jax.tree_map(lambda x: x.tolist(), test_summary), f)
        return

    metrics_all = []
    eval_cur = 0
    tick = time.time()
    logging.info('Starting training')
    logging.info('====================')

    for step, batch in zip(range(start_step, num_train_steps), train_iter):
        batch = common_utils.shard(jax.tree_map(
            lambda x: x._numpy(), batch))  # pylint: disable=protected-access
        state, metrics = p_train_step(state, batch, dropout_rng=dropout_rngs)
        metrics_all.append(metrics)
        logging.info('train in step: %d', step)
        
        '''
        # Save a Checkpoint
        if ((step % config.checkpoint_freq == 0 and step > 0) or
                step == num_train_steps - 1):
            if jax.process_index() == 0 and config.save_checkpoints:
                # Save unreplicated optimizer + model state.
                checkpoints.save_checkpoint(FLAGS.model_dir,
                                            jax_utils.unreplicate(state), step, overwrite=True)
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
            summary['perplexity'] = jnp.clip(
                jnp.exp(summary['loss']), a_max=1.0e4)
            logging.info('train in step: %d, loss: %.4f, acc: %.4f', step,
                         summary['loss'], summary['accuracy'])
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
            logging.info('eval in step: %d, loss: %.4f, acc: %.4f, f1: %.4f', step,
                         eval_summary['loss'], eval_summary['accuracy'], eval_summary['f1'])
            if jax.process_index() == 0:
                for key, val in eval_summary.items():
                    summary_writer.scalar(f'eval_{key}', val, step)
                summary_writer.flush()
                
            # Save the best
            if FLAGS.task_name == "d2a_function":
                save_key = 'accuracy'
            else:
                save_key = 'f1'
            if eval_summary[save_key] > eval_cur:
                eval_cur = eval_summary[save_key]
                
                # Save a Checkpoint
                if jax.process_index() == 0:
                    # Save unreplicated optimizer + model state.
                    checkpoints.save_checkpoint(FLAGS.model_dir,
                                                jax_utils.unreplicate(state), step, overwrite=True)
                    
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
