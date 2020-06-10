"""Extractor for HOLparam models. Tokenizes goals and theorems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import modeling


def get_all_thms_dataset(data_dir, batch_size, seq_length):
  name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  all_thms_file = os.path.join(data_dir, 'thms_ls.train')
  # Get a constant batch_size tensor of tokenized random train theorems.
  d = tf.data.TFRecordDataset(all_thms_file)
  d = d.repeat()
  # Shuffle within a sliding window slightly larger than the set of thms.
  d = d.shuffle(
    buffer_size=10000,
    reshuffle_each_iteration=True)

  d = d.map(lambda record: _decode_record(record, name_to_features))

  d = d.batch(7 * batch_size)

  return d


def get_negative(negatives, steps_per_epoch, seq_length, batch_size):
  global_step = tf.to_int32(tf.train.get_global_step())

  negatives = tf.slice(negatives, [0, global_step // steps_per_epoch], [batch_size, seq_length * 7])
  return tf.reshape(negatives, [7 * batch_size, seq_length])


def to_mask(n):
  return tf.cond(tf.equal(n, tf.constant(0)), lambda: tf.constant(0), lambda: tf.constant(1))


def extract(features, seq_length, steps_per_epoch, batch_size):
  """Converts 'goal' features and 'thms' labels to list of ids by vocab."""

  if 'tac_ids' not in features:
    raise ValueError('tac_id label missing.')

  # Tile the related features/labels (goals are tiled after embedding).
  goal_tiling_size = 8
  features['tac_ids'] = tf.tile(features['tac_ids'], [goal_tiling_size])
  features['is_real_example'] = tf.tile(features['is_real_example'], [goal_tiling_size])

  example = get_negative(features['negatives'], steps_per_epoch, seq_length, batch_size)

  neg_thms_input_ids = example
  neg_thms_input_mask = tf.map_fn(lambda x: tf.map_fn(to_mask, x), neg_thms_input_ids)
  neg_thms_segment_ids = tf.fill(modeling.get_shape_list(neg_thms_input_ids, expected_rank=2), 0)

  features['thm_label'] = tf.concat([
      tf.ones(tf.shape(features['thm_input_ids'])[0], dtype=tf.int32),
      tf.zeros(tf.shape(neg_thms_input_ids)[0], dtype=tf.int32)
  ], axis=0)

  features['thm_input_ids'] = tf.concat([features['thm_input_ids'], neg_thms_input_ids], axis=0)
  features['thm_input_mask'] = tf.concat([features['thm_input_mask'], neg_thms_input_mask], axis=0)
  features['thm_segment_ids'] = tf.concat([features['thm_segment_ids'], neg_thms_segment_ids], axis=0)

  return features
