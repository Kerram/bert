# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """Checks whether the casing config is consistent with the checkpoint name."""

  # The casing has to be passed in by the user and there is no explicit check
  # as to whether it matches the checkpoint. The casing information probably
  # should have been stored in the bert_config.json file, but it's not, so
  # we have to heuristically detect it to validate.

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0

  ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[MASK2]", "(", ")"]
  for sym in ctrl_symbols:
    sym = sym.strip()
    vocab[sym] = index
    index += 1

  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

class LongestTokenizer(object):
  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = load_vocab(vocab) 
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        #print("ALAARM ", token)
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens
  
  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class TensorWorkSplitter(object):
  """Extract terms/thms and tokenize based on vocab.
  Code mostly from deephol train -- extractor.py.

  Attributes:
    vocab_table: Lookup table for goal vocab embeddings.
  """

  def __init__(self, vocab_file):
    # Create vocab lookup tables from existing vocab id lists.
    with tf.variable_scope('extractor'):
      self.vocab_table = self._vocab_table_from_file(vocab_file)

  def _vocab_table_from_file(self, filename, reverse=False):
    with tf.gfile.Open(filename, 'r') as f:
      keys = [s.strip() for s in f.readlines()]
      values = tf.range(len(keys), dtype=tf.int64)
      if not reverse:
        init = tf.contrib.lookup.KeyValueTensorInitializer(keys, values)
        return tf.contrib.lookup.HashTable(init, 1)
      else:
        init = tf.contrib.lookup.KeyValueTensorInitializer(values, keys)
        return tf.contrib.lookup.HashTable(init, '')

  def tokenize(self, tm, max_seq_length):
    """Tokenizes tensor string according to lookup table."""
    tm = tf.strings.join(['[CLS] ', tf.strings.strip(tm), ' [SEP]'])
    tf.logging.info("  name = %s, shape = %s" % ("tm", tm.shape))
    # Remove parentheses - they can be recovered for S-expressions.
    #tm = tf.strings.regex_replace(tm, r'\(', ' ')
    #tm = tf.strings.regex_replace(tm, r'\)', ' ')
    words = tf.strings.split(tm)
    tf.logging.info("  name = %s, shape = %s" % ("words", words.shape))
    # Truncate long terms.
    words = tf.sparse.slice(words, [0, 0],
                            [tf.shape(words)[0], max_seq_length])

    word_values = words.values
    id_values = tf.to_int32(self.vocab_table.lookup(word_values))
    tf.logging.info("  name = %s, shape = %s" % ("id_values", id_values.shape))
    ids = tf.SparseTensor(words.indices, id_values, words.dense_shape)
    ids = tf.sparse_tensor_to_dense(ids)
    return ids
