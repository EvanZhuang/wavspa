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
"""Input pipeline for the imdb dataset."""

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from datasets import load_dataset
from transformers import AutoTokenizer

AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_dataset(file_path, batch_size, data_name="Source", label_name="Target"):
    """Preprocess dataset."""
    tf.logging.info(file_path)
    sel_cols = [data_name, label_name]
    #col_defaults = [tf.string, tf.int32]
    ds = tf.data.experimental.make_csv_dataset([file_path],
                                               batch_size,
                                               select_columns=sel_cols,
                                               field_delim=',',
                                               header=True,
                                               shuffle=False,
                                               num_epochs=1)
    ds = ds.unbatch()
    return ds


def get_codexglue_defect(data_dir, tokenizer):
    "Get devign dataset"
    def dataset_transform(dataset):
        dataset.set_format(type='tensorflow', columns=['input_ids', 'target', 'attention_mask'])
        features = {x: dataset[x].to_tensor(default_value=0, 
                                            shape=[None, tokenizer.model_max_length]) for x in ['input_ids', 'attention_mask']}
        tfdataset = tf.data.Dataset.from_tensor_slices((features, dataset["target"]))
        return tfdataset
    #dataset = load_dataset("code_x_glue_cc_defect_detection", cache_dir="/data/yufan/lra_jax/cache")
    
    dataset = load_dataset("csv", data_files={"train": "{}/Code-Code/Defect-detection/dataset/train.csv".format(data_dir), 
                                              "valid": "{}/Code-Code/Defect-detection/dataset/valid.csv".format(data_dir),
                                              "test": "{}/Code-Code/Defect-detection/dataset/test.csv".format(data_dir)},
                          cache_dir="./cache")

    train_raw = dataset['train'].map(lambda examples: tokenizer(examples['func'], truncation=True, padding='max_length'), batched=False)
    valid_raw = dataset['valid'].map(lambda examples: tokenizer(examples['func'], truncation=True, padding='max_length'), batched=False)
    test_raw = dataset['test'].map(lambda examples: tokenizer(examples['func'], truncation=True, padding='max_length'), batched=False)

    #train = dataset_transform(train_raw)
    #valid = dataset_transform(valid_raw)
    #test = dataset_transform(test_raw)
    train_raw = train_raw.remove_columns(["Unnamed: 0", "project", "func", "commit_id", "idx", "attention_mask"])
    train_raw = train_raw.rename_column("input_ids", "inputs")
    train_raw = train_raw.rename_column("target", "targets")
    
    valid_raw = valid_raw.remove_columns(["Unnamed: 0", "project", "func", "commit_id", "idx", "attention_mask"])
    valid_raw = valid_raw.rename_column("input_ids", "inputs")
    valid_raw = valid_raw.rename_column("target", "targets")
    
    test_raw = test_raw.remove_columns(["Unnamed: 0", "project", "func", "commit_id", "idx", "attention_mask"])
    test_raw = test_raw.rename_column("input_ids", "inputs")
    test_raw = test_raw.rename_column("target", "targets")
    
    def adapt_example(example):
        return {'inputs': example['input_ids'], 
                'targets': example['target']}
    
    train = train_raw.to_tf_dataset(batch_size=1).unbatch()
    valid = valid_raw.to_tf_dataset(batch_size=1).unbatch()
    test = test_raw.to_tf_dataset(batch_size=1).unbatch()
    
    logging.info('Data sample: %s', next(
        iter(tfds.as_numpy(train.skip(4)))))

    return train, valid, test


def get_agnews_dataset():
    """Get dataset from  agnews tfds. converts into src/tgt pairs."""
    data = tfds.load('ag_news_subset')
    train_raw = data['train']
    valid_raw = data['test']
    test_raw = data['test']
    # use test set for validation because agnews doesn't have val set.
    # Print an example.
    logging.info('Data sample: %s', next(
        iter(tfds.as_numpy(train_raw.skip(4)))))

    def adapt_example(example):
        return {'Source': example['description'], 'Target': example['label']}

    train = train_raw.map(adapt_example)
    valid = valid_raw.map(adapt_example)
    test = test_raw.map(adapt_example)

    return train, valid, test


def get_tc_datasets(n_devices,
                    task_name,
                    data_dir=None,
                    batch_size=256,
                    fixed_vocab=None,
                    max_length=512,
                    tokenizer='char'):
    """Get text classification datasets."""
    if batch_size % n_devices:
        raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                         (batch_size, n_devices))
        
    if task_name == 'd2a_function':
        train_path = data_dir + "/function/d2a_lbv1_function_train.csv"
        val_path = data_dir + "/function/d2a_lbv1_function_dev.csv"
        test_path = data_dir + "/function/d2a_lbv1_function_dev.csv"
        def adapt_example(example):
            return {'Source': example['code'], 'Target': example['label']}
                
        train_dataset = preprocess_dataset(train_path, batch_size, data_name="code", label_name="label")
        val_dataset = preprocess_dataset(val_path, batch_size, data_name="code", label_name="label")
        test_dataset = preprocess_dataset(test_path, batch_size, data_name="code", label_name="label")

        train_dataset = train_dataset.map(adapt_example)
        val_dataset = val_dataset.map(adapt_example)
        test_dataset = test_dataset.map(adapt_example)
    elif task_name == 'd2a_trace':
        train_path = data_dir + "/trace/d2a_lbv1_trace_train.csv"
        val_path = data_dir + "/trace/d2a_lbv1_trace_dev.csv"
        test_path = data_dir + "/trace/d2a_lbv1_trace_dev.csv"
        def adapt_example(example):
            return {'Source': example['trace'], 'Target': example['label']}
                
        train_dataset = preprocess_dataset(train_path, batch_size, data_name="trace", label_name="label")
        val_dataset = preprocess_dataset(val_path, batch_size, data_name="trace", label_name="label")
        test_dataset = preprocess_dataset(test_path, batch_size, data_name="trace", label_name="label")

        train_dataset = train_dataset.map(adapt_example)
        val_dataset = val_dataset.map(adapt_example)
        test_dataset = test_dataset.map(adapt_example)
    elif task_name == 'codexglue_defect':
        logging.info('Using gpt-2 vocab')
        encoder = AutoTokenizer.from_pretrained("gpt2")
        encoder.add_special_tokens({'pad_token': '[PAD]'})
        
        train_dataset, val_dataset, test_dataset = get_codexglue_defect(data_dir, tokenizer=encoder)
        '''
        train_path = data_dir + "/Code-Code/Defect-detection/dataset/train.csv"
        val_path = data_dir + "/Code-Code/Defect-detection/dataset/valid.csv"
        test_path = data_dir + "/Code-Code/Defect-detection/dataset/test.csv"
        def adapt_example(example):
            return {'Source': example['func'], 'Target': example['target']}

        train_dataset = preprocess_dataset(train_path, batch_size, data_name="func", label_name="target")
        val_dataset = preprocess_dataset(val_path, batch_size, data_name="func", label_name="target")
        test_dataset = preprocess_dataset(test_path, batch_size, data_name="func", label_name="target")

        train_dataset = train_dataset.map(adapt_example)
        val_dataset = val_dataset.map(adapt_example)
        test_dataset = test_dataset.map(adapt_example)
        '''
    else:
        train_path = data_dir + task_name + '_train.tsv'
        val_path = data_dir + task_name + '_val.tsv'
        test_path = data_dir + task_name + '_test.tsv'

        train_dataset = preprocess_dataset(train_path, batch_size)
        val_dataset = preprocess_dataset(val_path, batch_size)
        test_dataset = preprocess_dataset(test_path, batch_size)

    tf.logging.info('Finished preprocessing')

    tf.logging.info(val_dataset)

    if tokenizer == 'char':
        logging.info('Using char/byte level vocab')
        encoder = tfds.deprecated.text.ByteTextEncoder()
    elif tokenizer == 'gpt2':
        pass
        #logging.info('Using gpt-2 vocab')
        #encoder = AutoTokenizer.from_pretrained("gpt2")
    else:
        if fixed_vocab is None:
            tf.logging.info('Building vocab')
            # build vocab
            vocab_set = set()
            tokenizer = tfds.deprecated.text.Tokenizer()
            for i, data in enumerate(train_dataset):
                examples = data['Source']
                examples = tokenizer.tokenize(examples.numpy())
                examples = np.reshape(examples, (-1)).tolist()
                vocab_set.update(examples)
                if i % 1000 == 0:
                    tf.logging.info('Processed {}'.format(i))
            tf.logging.info(len(vocab_set))
            vocab_set = list(set(vocab_set))
            tf.logging.info('Finished processing vocab size={}'.format(
                len(vocab_set)))
        else:
            vocab_set = list(set(fixed_vocab))
        encoder = tfds.deprecated.text.TokenTextEncoder(vocab_set)

    def tf_encode(x):
        result = tf.py_function(lambda s: tf.constant(encoder.encode(s.numpy())), [x,], tf.int32)
        result.set_shape([None])
        return result

    def tokenize(d):
        return {
            'inputs': tf_encode(d['Source'])[:max_length],
            'targets': d['Target']
        }
    
    def gpt2_tokenize(d):
        return {
            'inputs': encoder(tf.strings.as_string(d['Source']))[:max_length],
            'targets': d['Target']
        }
    
    if tokenizer == 'gpt2':
        #train_dataset = train_dataset.map(gpt2_tokenize, num_parallel_calls=AUTOTUNE)
        #val_dataset = val_dataset.map(gpt2_tokenize, num_parallel_calls=AUTOTUNE)
        #test_dataset = test_dataset.map(gpt2_tokenize, num_parallel_calls=AUTOTUNE)
        pass
    else:
        train_dataset = train_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
        val_dataset = val_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
        test_dataset = test_dataset.map(tokenize, num_parallel_calls=AUTOTUNE)
        
    if tokenizer == 'gpt2':
        train_dataset = train_dataset.shuffle(
            buffer_size=256, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    else:
        max_shape = {'inputs': [max_length], 'targets': []}
        train_dataset = train_dataset.shuffle(
            buffer_size=256, reshuffle_each_iteration=True).padded_batch(
                batch_size, padded_shapes=max_shape, drop_remainder=True)
        val_dataset = val_dataset.padded_batch(batch_size, padded_shapes=max_shape, drop_remainder=True)
        test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=max_shape, drop_remainder=True)

    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset, encoder
