# Mix of bert's classifier.py and deephol's architectures.py
# Code used to run deephol training on tpu outside of deephol infrastructure

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import sys

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir",
    None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.",
)

flags.DEFINE_string(
    "bert_config_file",
    None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.",
)

flags.DEFINE_string(
    "vocab_file", None, "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory where the model checkpoints will be written.",
)

flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Initial checkpoint (usually from a pre-trained BERT model).",
)


flags.DEFINE_integer(
    "max_seq_length",
    512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False, "Whether to run the model in inference mode on the test set."
)

flags.DEFINE_bool(
    "do_export", False, "Whether to export the model."
)

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 3.0, "Total number of training epochs to perform."
)

flags.DEFINE_float(
    "warmup_proportion",
    0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.",
)

flags.DEFINE_integer(
    "save_checkpoints_steps", 10000, "How often to save the model checkpoint."
)

flags.DEFINE_integer(
    "iterations_per_loop", 1000, "How many steps to make in each estimator call."
)

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name",
    None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.",
)

tf.flags.DEFINE_string(
    "tpu_zone",
    "us-central1-f",
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.",
)

tf.flags.DEFINE_string(
    "gcp_project",
    "zpp-mim-1920",
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.",
)

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores",
    8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.",
)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, goal, thm, tac_id=None, is_negative=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      goal: The untokenized goal string 
      thm:  The untokenized theorem string.
      tac_id: id of tactic for the goal
      is_negative: indicates whether the theorem matches the goal
    """
        self.guid = guid
        self.goal = goal
        self.thm = thm
        self.tac_id = tac_id
        self.is_negative = is_negative


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
     See run_classifier.py for details.
  """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        goal_input_ids,
        goal_input_mask,
        goal_segment_ids,
        thm_input_ids,
        thm_input_mask,
        thm_segment_ids,
        tac_id,
        is_negative,
        is_real_example=True,
    ):

        self.goal_input_ids = goal_input_ids
        self.goal_input_mask = goal_input_mask
        self.goal_segment_ids = goal_segment_ids
        self.thm_input_ids = thm_input_ids
        self.thm_input_mask = thm_input_mask
        self.thm_segment_ids = thm_segment_ids
        self.tac_id = tac_id
        self.is_negative = is_negative
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_tac_labels(self):
        """Gets the list tac_ids"""
        raise NotImplementedError()

    def get_is_negative_labels(self):
        """Gets the list of is_negative labels"""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class DeepholProcessor(DataProcessor):
    """Processor for Deephol dataset"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.tsv")), "valid"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_tac_labels(self):
        return [str(i) for i in range(41)]

    def get_is_negative_labels(self):
        return ["True", "False"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            """ skip header """
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                # TODO - obejrzeć dokładnie test-set
                goal = tokenization.convert_to_unicode(line[0])
                thm = tokenization.convert_to_unicode(line[1])
                is_negative = "True"
                label = "0"
                tac_id = "0"  # TODO change
            else:
                goal = tokenization.convert_to_unicode(line[0])
                thm = tokenization.convert_to_unicode(line[1])
                is_negative = tokenization.convert_to_unicode(line[2])
                tac_id = tokenization.convert_to_unicode(line[3])
            examples.append(
                InputExample(
                    guid=guid,
                    goal=goal,
                    thm=thm,
                    tac_id=tac_id,
                    is_negative=is_negative,
                )
            )
        return examples


def convert_tokens(tokens, tokenizer, max_seq_length):
    res = []
    segment_ids = []
    res.append("[CLS]")
    segment_ids.append(0)

    for token in tokens:
        res.append(token)
        segment_ids.append(0)

    res.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(res)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return (res, input_ids, input_mask, segment_ids)


def convert_single_example(
    ex_index, example, tac_label_list, is_negative_label_list, max_seq_length, tokenizer
):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            goal_input_ids=[0] * max_seq_length,
            goal_input_mask=[0] * max_seq_length,
            goal_segment_ids=[0] * max_seq_length,
            thm_input_ids=[0] * max_seq_length,
            thm_input_mask=[0] * max_seq_length,
            thm_segment_ids=[0] * max_seq_length,
            tac_id=0,
            is_negative=True,
            is_real_example=False,
        )

    tac_label_map = {}
    for (i, label) in enumerate(tac_label_list):
        tac_label_map[label] = i

    is_negative_label_map = {}
    for (i, label) in enumerate(is_negative_label_list):
        is_negative_label_map[label] = i

    g_tokens = tokenizer.tokenize(example.goal)
    t_tokens = tokenizer.tokenize(example.thm)

    if len(g_tokens) > max_seq_length - 2:
        g_tokens = g_tokens[0 : (max_seq_length - 2)]

    if len(t_tokens) > max_seq_length - 2:
        t_tokens = t_tokens[0 : (max_seq_length - 2)]

    (goal_tokens, goal_input_ids, goal_input_mask, goal_segment_ids) = convert_tokens(
        g_tokens, tokenizer, max_seq_length
    )
    (thm_tokens, thm_input_ids, thm_input_mask, thm_segment_ids) = convert_tokens(
        t_tokens, tokenizer, max_seq_length
    )

    assert len(goal_input_ids) == max_seq_length
    assert len(goal_input_mask) == max_seq_length
    assert len(goal_segment_ids) == max_seq_length
    assert len(thm_input_ids) == max_seq_length
    assert len(thm_input_mask) == max_seq_length
    assert len(thm_segment_ids) == max_seq_length

    tac_id = tac_label_map[example.tac_id]
    is_negative = is_negative_label_map[example.is_negative]

    if ex_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info(
            "goal_tokens: %s"
            % " ".join([tokenization.printable_text(x) for x in goal_tokens])
        )
        tf.logging.info(
            "thm_tokens: %s"
            % " ".join([tokenization.printable_text(x) for x in thm_tokens])
        )
        tf.logging.info(
            "goal_input_ids: %s" % " ".join([str(x) for x in goal_input_ids])
        )
        tf.logging.info("thm_input_ids: %s" % " ".join([str(x) for x in thm_input_ids]))
        tf.logging.info(
            "goal_input_mask: %s" % " ".join([str(x) for x in goal_input_mask])
        )
        tf.logging.info(
            "thm_input_mask: %s" % " ".join([str(x) for x in thm_input_mask])
        )
        tf.logging.info(
            "goal_segment_ids: %s" % " ".join([str(x) for x in goal_segment_ids])
        )
        tf.logging.info(
            "thm_segment_ids: %s" % " ".join([str(x) for x in thm_segment_ids])
        )
        tf.logging.info("tac_id: %d" % (tac_id))
        tf.logging.info("is_negative: %s" % (example.is_negative))

    feature = InputFeatures(
        goal_input_ids=goal_input_ids,
        goal_input_mask=goal_input_mask,
        goal_segment_ids=goal_segment_ids,
        thm_input_ids=thm_input_ids,
        thm_input_mask=thm_input_mask,
        thm_segment_ids=thm_segment_ids,
        tac_id=tac_id,
        is_negative=is_negative,
        is_real_example=True,
    )

    return feature


def file_based_convert_examples_to_features(
    examples, tac_label_list, is_negative_list, max_seq_length, tokenizer, output_file
):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(
            ex_index,
            example,
            tac_label_list,
            is_negative_list,
            max_seq_length,
            tokenizer,
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["goal_input_ids"] = create_int_feature(feature.goal_input_ids)
        features["goal_input_mask"] = create_int_feature(feature.goal_input_mask)
        features["goal_segment_ids"] = create_int_feature(feature.goal_segment_ids)
        features["thm_input_ids"] = create_int_feature(feature.thm_input_ids)
        features["thm_input_mask"] = create_int_feature(feature.thm_input_mask)
        features["thm_segment_ids"] = create_int_feature(feature.thm_segment_ids)
        features["tac_ids"] = create_int_feature([feature.tac_id])
        features["is_negative"] = create_int_feature([feature.is_negative])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "goal_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "goal_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "goal_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "tac_ids": tf.FixedLenFeature([], tf.int64),
        "is_negative": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
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

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        )

        return d

    return input_fn


def create_encoding(
    name,
    is_training,
    bert_config,
    input_ids,
    input_mask,
    segment_ids,
    use_one_hot_embeddings,
):
    with tf.variable_scope(name):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
        )

        output = model.get_pooled_output()

        tf.add_to_collection(name + '_net', output)

        return output


def tactic_classifier(goal_net, is_training, tac_labels, num_tac_labels):
    hidden_size = goal_net.shape[-1].value

    tf.add_to_collection('tactic_net', goal_net)

    # Adding 3 dense layers with dropout like in deephol
    # with tf.variable_scope("loss"):
    if is_training:
        goal_net = tf.nn.dropout(goal_net, keep_prob=0.7)
    goal_net = tf.layers.dense(
        goal_net, hidden_size, activation=tf.nn.relu, name="tac_dense1"
    )

    if is_training:
        goal_net = tf.nn.dropout(goal_net, keep_prob=0.7)
    goal_net = tf.layers.dense(
        goal_net, hidden_size, activation=tf.nn.relu, name="tac_dense2"
    )

    if is_training:
        goal_net = tf.nn.dropout(goal_net, keep_prob=0.7)
    tac_logits = tf.layers.dense(
        goal_net, num_tac_labels, activation=tf.nn.relu, name="tac_dense3"
    )

    tf.add_to_collection('tactic_logits', tac_logits)

    tac_probabilities = tf.nn.softmax(tac_logits, axis=-1)
    log_probs = tf.nn.log_softmax(tac_logits, axis=-1)

    one_hot_labels = tf.one_hot(tac_labels, depth=num_tac_labels, dtype=tf.float32)

    tac_per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    tac_loss = tf.reduce_mean(tac_per_example_loss)

    tf.logging.info("tac_logits.shape = %s" % (tac_logits.shape))
    tf.logging.info(
        "tac_per_example_loss.shape = %s" % (tac_per_example_loss.shape)
    )
    # tac_per_example_loss = tf.print(tac_per_example_loss, [tf.shape(tac_per_example_loss)], "tac_per_example_loss_shape: ", summarize =-1)
    # tac_logits = tf.print(tac_logits, [tf.shape(tac_logits)], "tac_logits_shape: ", summarize =-1)

    return (tac_loss, tac_per_example_loss, tac_logits, tac_probabilities)


def pairwise_classifier(goal_net, thm_net, is_training, is_negative_labels, tac_labels):
    # concat goal_net, thm_net and their dot product as in deephol
    hidden_size = goal_net.shape[-1].value
    # [batch_size, 3 * hidden_size]
    net = tf.concat([goal_net, thm_net, goal_net * thm_net], -1)

    # Adding 3 dense layers with dropout like in deephol
    # with tf.variable_scope("loss"):
    if is_training:
        net = tf.nn.dropout(net, keep_prob=0.7)
    net = tf.layers.dense(
        net, hidden_size, activation=tf.nn.relu, name="par_dense1"
    )

    if is_training:
        net = tf.nn.dropout(net, keep_prob=0.7)
    net = tf.layers.dense(
        net, hidden_size, activation=tf.nn.relu, name="par_dense2"
    )

    if is_training:
        net = tf.nn.dropout(net, keep_prob=0.7)
    net = tf.layers.dense(
        net, hidden_size, activation=tf.nn.relu, name="par_dense3"
    )

    # [batch_size, 1]
    par_logits = tf.layers.dense(net, 1, activation=None)

    tf.add_to_collection('pairwise_score', par_logits)

    # [batch_size, 1]
    par_per_example_loss = 0.5 * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(tf.expand_dims(is_negative_labels, 1), tf.float32),
        logits=par_logits,
    )

    # [batch_size]
    par_logits = tf.squeeze(par_logits, [1])
    par_per_example_loss = tf.squeeze(par_per_example_loss, [1])

    #  sum, not mean, as opposed to tactic classifier
    #  a bit strange but they did so in deephol
    #  presumably it makes the code less scalable for batch_size
    par_loss = tf.reduce_sum(par_per_example_loss)

    tf.logging.info("par_logits.shape = %s" % (par_logits.shape))
    tf.logging.info(
        "par_per_example_loss.shape = %s" % (par_per_example_loss.shape)
    )
    # par_logits = tf.print(par_logits, [tf.shape(par_logits)], "par_logits_shape: ", summarize =-1)
    # par_per_example_loss = tf.print(par_per_example_loss, [tf.shape(par_per_example_loss)], "par_per_example_loss_shape: ", summarize =-1)

    return (par_loss, par_per_example_loss, par_logits)


def create_model(
    bert_config,
    goal_input_ids,
    goal_input_mask,
    goal_segment_ids,
    thm_input_ids,
    thm_input_mask,
    thm_segment_ids,
    use_one_hot_embeddings,
    tac_labels,
    num_tac_labels,
    is_negative_labels,
    is_training,
):

    with tf.variable_scope("encoder"):
        with tf.variable_scope("dilated_cnn_pairwise_encoder"):
            goal_net = create_encoding(
                "goal",
                is_training,
                bert_config,
                goal_input_ids,
                goal_input_mask,
                goal_segment_ids,
                use_one_hot_embeddings,
            )
            thm_net = create_encoding(
                "thm",
                is_training,
                bert_config,
                thm_input_ids,
                thm_input_mask,
                thm_segment_ids,
                use_one_hot_embeddings,
            )

    with tf.variable_scope("classifier"):
        (
            tac_loss,
            tac_per_example_loss,
            tac_logits,
            tac_probabilities,
        ) = tactic_classifier(goal_net, is_training, tac_labels, num_tac_labels)

    with tf.variable_scope("pairwise_scorer"):
        (par_loss, par_per_example_loss, par_logits) = pairwise_classifier(
            goal_net, thm_net, is_training, is_negative_labels, tac_labels
        )

    total_loss = par_loss + tac_loss

    return (
        total_loss,
        par_per_example_loss,
        tac_per_example_loss,
        tac_logits,
        tac_probabilities,
        par_loss,
        par_logits,
    )


# Kuba's hack
class Remover:
    prefixes = []
    keys_list = []

    def __init__(self, sl):
        self.prefixes = sl
        self.keys_list = []

    def get(self, key):
        pref = ""
        for p in self.prefixes:
            if key.startswith(p):
                pref = p
        assert pref
        self.keys_list.append(key)
        return key[len(pref) :]

    def keys(self):
        return self.keys_list


def reduce_mean_weighted(values, weights):
    weights = tf.to_float(weights)
    w_sum = tf.math.reduce_sum(weights)
    mean = tf.divide(tf.reduce_sum(values * weights), w_sum)
    return tf.cond(
        tf.equal(w_sum, tf.constant(0.0)), lambda: tf.constant(0.0), lambda: mean
    )


def _pad_up_to(value, size, axis, name=None):
  """Pad a tensor with zeros on the right along axis to a least the given size.

  Args:
    value: Tensor to pad.
    size: Minimum size along axis.
    axis: A nonnegative integer.
    name: Optional name for this operation.

  Returns:
    Padded value.
  """
  with tf.name_scope(name, 'pad_up_to') as name:
    value = tf.convert_to_tensor(value, name='value')
    axis = tf.convert_to_tensor(axis, name='axis')
    need = tf.nn.relu(size - tf.shape(value)[axis])
    ids = tf.stack([tf.stack([axis, 1])])
    paddings = tf.sparse_to_dense(ids, tf.stack([tf.rank(value), 2]), need)
    padded = tf.pad(value, paddings, name=name)
    # Fix shape inference
    axis = tf.contrib.util.constant_value(axis)
    shape = value.get_shape()
    if axis is not None and shape.ndims is not None:
      shape = shape.as_list()
      shape[axis] = None
      padded.set_shape(shape)
    return padded


def to_mask(n):
  return tf.cond(tf.equal(n, tf.constant(0)), lambda: tf.constant(0), lambda: tf.constant(1))


def update_features_using_deephol(features, max_seq_length):
    tensor_tokenizer = tokenization.TensorWorkSplitter(vocab_file=FLAGS.vocab_file)

    with tf.variable_scope('extractor'):
        tf.logging.info("********** Tokenization of goal in Holist ********")
        goal_str = tf.Variable([''], dtype=tf.string)

        tf.add_to_collection('goal_string', goal_str)

        goal = tensor_tokenizer.tokenize(goal_str, max_seq_length)

        goal = _pad_up_to(goal, max_seq_length, 1)
        goal_input_mask = tf.map_fn(lambda x: tf.map_fn(to_mask, x), goal)
        goal_segment_ids = tf.fill(modeling.get_shape_list(goal, expected_rank=2), 0)

        features['goal_input_ids'] = goal
        features['goal_input_mask'] = goal_input_mask
        features['goal_segment_ids'] = goal_segment_ids


        tf.logging.info("********** Tokenization of theorem in Holist ********")
        thm_str = tf.Variable([''], dtype=tf.string)

        tf.add_to_collection('thm_string', thm_str)

        thm = tensor_tokenizer.tokenize(thm_str, max_seq_length)

        thm = _pad_up_to(thm, max_seq_length, 1)
        thm_input_mask = tf.map_fn(lambda x: tf.map_fn(to_mask, x), thm)
        thm_segment_ids = tf.fill(modeling.get_shape_list(thm, expected_rank=2), 0)

        features['thm_input_ids'] = thm
        features['thm_input_mask'] = thm_input_mask
        features['thm_segment_ids'] = thm_segment_ids


def model_fn_builder(
    bert_config,
    num_tac_labels,
    init_checkpoint,
    learning_rate,
    num_train_steps,
    num_warmup_steps,
    use_tpu,
    use_one_hot_embeddings,
    max_seq_length
):
    def model_fn(features, labels, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        if not use_tpu:  # HOList does not use TPU.
            update_features_using_deephol(features, max_seq_length)

        tf.logging.info("*** Features after deephol ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        goal_input_ids = features["goal_input_ids"]
        goal_input_mask = features["goal_input_mask"]
        goal_segment_ids = features["goal_segment_ids"]
        thm_input_ids = features["thm_input_ids"]
        thm_input_mask = features["thm_input_mask"]
        thm_segment_ids = features["thm_segment_ids"]
        tac_ids = features["tac_ids"]
        is_negative = features["is_negative"]

        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        (
            total_loss,
            par_per_example_loss,
            tac_per_example_loss,
            tac_logits,
            tac_probabilities,
            par_loss,
            par_logits,
        ) = create_model(
            bert_config,
            goal_input_ids,
            goal_input_mask,
            goal_segment_ids,
            thm_input_ids,
            thm_input_mask,
            thm_segment_ids,
            use_one_hot_embeddings,
            tac_ids,
            num_tac_labels,
            is_negative,
            is_training,
        )

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        scaffold_fn = None

        if init_checkpoint:
            (
                assignment_map,
                initialized_variable_names,
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            if use_tpu:

                def tpu_scaffold():
                    tf.train.warm_start(
                        init_checkpoint,
                        "encoder/dilated_cnn_pairwise_encoder/thm/bert*",
                        None,
                        Remover(["encoder/dilated_cnn_pairwise_encoder/thm/"]),
                    )

                    tf.train.warm_start(
                        init_checkpoint,
                        "encoder/dilated_cnn_pairwise_encoder/goal/bert*",
                        None,
                        Remover(["encoder/dilated_cnn_pairwise_encoder/goal/"]),
                    )

                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold

            else:
                tf.train.warm_start(
                    init_checkpoint,
                    "encoder/*")

                tf.train.warm_start(
                    init_checkpoint,
                    "classifier/*")

                tf.train.warm_start(
                    init_checkpoint,
                    "pairwise_scorer/*")

        # Ten komunikat się nie będzie do końca zgadzał
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(
                "  name = %s, shape = %s%s", var.name, var.shape, init_string
            )

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu
            )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, scaffold_fn=scaffold_fn
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(
                tac_per_example_loss,
                tac_logits,
                par_per_example_loss,
                par_logits,
                tac_ids,
                is_negative,
                is_real_example,
            ):
                tac_predictions = tf.argmax(tac_logits, axis=-1, output_type=tf.int32)

                # Tactic accuracy
                tac_accuracy = tf.metrics.accuracy(
                    labels=tac_ids, predictions=tac_predictions, weights=is_real_example
                )

                # Top 5 tactics accuracy
                topk_preds = tf.to_float(tf.nn.in_top_k(tac_logits, tac_ids, 5))
                # topk_preds = tf.boolean_mask(topk_preds, is_real_example)
                tac_topk_accuracy = tf.metrics.mean(
                    values=topk_preds, weights=is_real_example
                )

                # for evaluation we count mean of both losses
                tac_loss = tf.metrics.mean(
                    values=tac_per_example_loss, weights=is_real_example
                )
                par_loss = tf.metrics.mean(
                    values=par_per_example_loss, weights=is_real_example
                )

                tot_loss = tac_loss + par_loss

                par_pred = tf.sigmoid(par_logits)
                pos_guess = tf.to_float(tf.greater(par_pred, 0.5))
                neg_guess = tf.to_float(tf.less(par_pred, 0.5))

                is_negative = tf.to_float(is_negative)
                pos_acc = tf.metrics.mean(
                    values=pos_guess, weights=is_real_example * (is_negative)
                )
                neg_acc = tf.metrics.mean(
                    values=neg_guess, weights=is_real_example * (1 - is_negative)
                )
                # pos_logits = reduce_mean_weighted(par_logits, 1 - is_negative)
                # neg_logits = reduce_mean_weighted(pos_logits, is_negative)

                # pos_pred = reduce_mean_weighted(par_pred, 1 - is_negative)
                # neg_pred = reduce_mean_weighted(par_pred, is_negative)

                # pos_acc = reduce_mean_weighted(pos_guess, 1 - is_negative)
                # neg_acc = reduce_mean_weighted(pos_guess, is_negative)

                # pos_logits = tf.boolean_mask(par_logits, 1 - is_negative)
                # neg_logits = tf.boolean_mask(par_logits, is_negative)
                # pos_pred = tf.sigmoid(pos_logits)
                # neg_pred = tf.sigmoid(neg_logits)
                # pos_acc = tf.reduce_mean(tf.to_float(tf.greater(pos_pred, 0.5)))
                # neg_acc = tf.reduce_mean(tf.to_float(tf.less(neg_pred, 0.5)))
                # acc_50_50 = (pos_acc + neg_acc) / 2.

                res = {
                    # 'pos_logits': pos_logits,
                    # 'neg_logits': neg_logits,
                    # 'pos_pred': pos_pred,
                    # 'neg_pred': neg_pred,
                    "pos_acc": pos_acc,
                    "neg_acc": neg_acc,
                    # 'acc_50_50': acc_50_50,
                    "tac_accuracy": tac_accuracy,
                    "tac_topk_accuracy": tac_topk_accuracy,
                    "tac_loss": tac_loss,
                    "par_loss": par_loss,
                    "total_loss": tot_loss,
                }

                return res

            eval_metrics = (
                metric_fn,
                [
                    tac_per_example_loss,
                    tac_logits,
                    par_per_example_loss,
                    par_logits,
                    tac_ids,
                    is_negative,
                    is_real_example,
                ],
            )

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn,
            )

        else:
            preds = {
                "tac_probabilities": tac_probabilities,
                "is_negative_prob": tf.sigmoid(par_logits),
            }

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=preds, scaffold_fn=scaffold_fn
            )

        return output_spec

    return model_fn


def main(_):
    csv.field_size_limit(sys.maxsize)
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_export:
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_predict' or `do_predict` must be True."
        )

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d"
            % (FLAGS.max_seq_length, bert_config.max_position_embeddings)
        )

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = DeepholProcessor()

    tokenizer = tokenization.LongestTokenizer(vocab=FLAGS.vocab_file)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project
        )

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host,
        ),
    )

    tac_labels = processor.get_tac_labels()
    is_negative_labels = processor.get_is_negative_labels()

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    tf.logging.info("Preparation completed!")


    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs
        )
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_tac_labels=len(tac_labels),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        max_seq_length=FLAGS.max_seq_length,
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
    )

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples,
            tac_labels,
            is_negative_labels,
            FLAGS.max_seq_length,
            tokenizer,
            train_file,
        )
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples,
            tac_labels,
            is_negative_labels,
            FLAGS.max_seq_length,
            tokenizer,
            eval_file,
        )

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info(
            "  Num examples = %d (%d actual, %d padding)",
            len(eval_examples),
            num_actual_eval_examples,
            len(eval_examples) - num_actual_eval_examples,
        )
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
        )

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples,
            tac_labels,
            is_negative_labels,
            FLAGS.max_seq_length,
            tokenizer,
            predict_file,
        )

        tf.logging.info("***** Running prediction*****")
        tf.logging.info(
            "  Num examples = %d (%d actual, %d padding)",
            len(predict_examples),
            num_actual_predict_examples,
            len(predict_examples) - num_actual_predict_examples,
        )
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
        )

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = (
                    "\t".join(
                        str(class_probability) for class_probability in probabilities
                    )
                    + "\n"
                )
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

    if FLAGS.do_export:
        feature_spec = {
            "goal_input_ids": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "goal_input_mask": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "goal_segment_ids": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "thm_input_ids": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "thm_input_mask": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "thm_segment_ids": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "tac_ids": tf.placeholder(dtype=tf.int32, shape=[None]),
            "is_negative": tf.placeholder(dtype=tf.int32, shape=[None]),
            "is_real_example": tf.placeholder(dtype=tf.int32, shape=[None]),
        }
        label_spec = {}
        build_input = tf.contrib.estimator.build_raw_supervised_input_receiver_fn
        input_receiver_fn = build_input(feature_spec, label_spec)

        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples,
            tac_labels,
            is_negative_labels,
            FLAGS.max_seq_length,
            tokenizer,
            predict_file,
        )

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
        )

        # Export final model
        tf.logging.info('*****  Starting to export model.   *****')
        save_hook = tf.train.CheckpointSaverHook(FLAGS.output_dir, save_secs=1)
        result = estimator.predict(input_fn=predict_input_fn, hooks=[save_hook])
        # now you will get graph.pbtxt which is used in SavedModel, and then
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


        estimator._export_to_tpu = False
        estimator.export_savedmodel(
            export_dir_base=os.path.join(FLAGS.output_dir, 'export/exporter'),
            serving_input_receiver_fn=input_receiver_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
