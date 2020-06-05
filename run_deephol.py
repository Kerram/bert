# Mix of bert's classifier.py and deephol's architectures.py
# Code used to run deephol training on tpu outside of deephol infrastructure

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import wavenet
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
    "vocab_file", None, "The vocabulary file."
)

flags.DEFINE_string(
    "output_dir",
    None,
    "The output directory where the model checkpoints will be written.",
)

flags.DEFINE_integer(
    "max_seq_length",
    512,
    "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded (not in exported model).",
)

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_string("train_file", None, "Path to train tf_record file.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_string("eval_file", None, "Path to eval tf_record file.")

flags.DEFINE_bool(
    "do_export", False, "Whether to export the model."
)

flags.DEFINE_string("test_file", None, "Path to test tf_record file. It is used to export model.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 1e-4, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_train_epochs", 3.0, "Total number of training epochs to perform."
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

flags.DEFINE_bool(
    "infinite_eval",
    False,
    "If infinite we will run eval on currently newest checkpoint."
    "If finite than we will run evaluation only once."
)


def get_tac_labels():
    return [str(i) for i in range(41)]


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "goal_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
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
            d = d.shuffle(buffer_size=400000)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        )

        return d

    return input_fn


def build_predict_fake_input_fn(input_file, seq_length, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator.
    This function differs from the one used for train and valid sets, because
    we add goal and thm strings as additional features."""

    name_to_features = {
        "goal_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "thm_input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "tac_ids": tf.FixedLenFeature([], tf.int64),
        "is_negative": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        'goal_str': tf.FixedLenFeature((), tf.string, default_value=''),
        'thm_str': tf.FixedLenFeature((), tf.string, default_value=''),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        )

        return d

    return input_fn


def get_vocab_embedding(embedding_str, vocab_size):
  return tf.get_variable(
      embedding_str,
      shape=(vocab_size, 128),
      dtype=tf.float32)


def _pad_to_multiple(value, size, axis, name=None):
  """Pad a tensor with zeros on the right to a multiple of the given size.

  Args:
    value: Tensor to pad.
    size: The result will be a multiple of `size` along `axis`.
    axis: A nonnegative integer.
    name: Optional name for this operation.

  Returns:
    Padded value.
  """
  with tf.name_scope(name, 'pad_to_multiple') as name:
    length = tf.shape(value)[axis]
    new_length = length // -size * -size  # Round up to multiple of size
    return _pad_up_to(value, size=new_length, axis=axis, name=name)


def wavenet_encoding(net):
  """Embed a given input tensor using multiple wavenet blocks.

  Arguments:
    net: input tensor of shape [batch, text_length, word_embedding_size]

  Returns:
    output: output tensor of shape [batch, 1, text length, hidden_size]
  """
  net = _pad_to_multiple(net, 2**4, axis=1)
  net = tf.expand_dims(net, 2)

  for _ in range(2):
    net = wavenet.wavenet_block(
        net,
        num_layers=4,
        depth=128,
        comb_weight=1.0,
        keep_prob=1.0)
  return net


def goal_encoding(
    vocab_size,
    goal_ids,
):
    goal_embedding = get_vocab_embedding('goal_embedding', vocab_size)
    # output shape is [batch_size, goal length, word_embedding_size]
    goal_net = tf.nn.embedding_lookup(goal_embedding, goal_ids)
    tf.add_to_collection('goal_embedding', goal_net)

    with tf.variable_scope('goal', reuse=False):
        # output shape: [batch_size, 1, goal length, hidden_size]
        goal_net = wavenet_encoding(goal_net)
        # output shape is [batch_size, hidden_size]
    goal_net = tf.reduce_max(goal_net, [1, 2])

    tf.add_to_collection('goal_net', goal_net)

    return goal_net


def thm_encoding(
    vocab_size,
    thm_ids,
):
    goal_embedding = get_vocab_embedding('goal_embedding', vocab_size)
    # output shape is [batch_size, thm length, word_embedding_size]
    thm_net = tf.nn.embedding_lookup(goal_embedding, thm_ids)
    tf.add_to_collection('thm_embedding', thm_net)

    with tf.variable_scope('thm', reuse=False):
        # output shape: [batch_size, 1, thm length, hidden_size]
        thm_net = wavenet_encoding(thm_net)

    # output shape is [batch_size, hidden_size]
    thm_net = tf.reduce_max(thm_net, [1, 2])
    tf.add_to_collection('thm_net', thm_net)

    return thm_net


def tactic_classifier(goal_net, is_training, tac_ids, num_tac_labels, is_real_example):
    hidden_size = goal_net.shape[-1].value

    tf.add_to_collection('tactic_net', goal_net)

    # Adding 3 dense layers with dropout like in deephol
    # with tf.variable_scope("loss"):
    if is_training:
        goal_net = tf.nn.dropout(goal_net, rate=(1 - 0.7))
    goal_net = tf.layers.dense(
        goal_net, hidden_size, activation=tf.nn.relu, name="tac_dense1"
    )

    if is_training:
        goal_net = tf.nn.dropout(goal_net, rate=(1 - 0.7))
    goal_net = tf.layers.dense(
        goal_net, hidden_size, activation=tf.nn.relu, name="tac_dense2"
    )

    if is_training:
        goal_net = tf.nn.dropout(goal_net, rate=(1 - 0.7))
    tac_logits = tf.layers.dense(
        goal_net, num_tac_labels, activation=None, name="tac_dense3"
    )

    tf.add_to_collection('tactic_logits', tac_logits)

    # Compute the log loss for the tactic logits.
    log_prob_tactic = tf.losses.sparse_softmax_cross_entropy(
        logits=tac_logits, labels=tac_ids, weights=is_real_example)

    tac_probabilities = tf.nn.softmax(tac_logits, axis=-1)

    return log_prob_tactic, tac_logits, tac_probabilities


def pairwise_scorer(goal_net, thm_net, is_training, is_negative_labels, is_real_example):
    # concat goal_net, thm_net and their dot product as in deephol
    hidden_size = goal_net.shape[-1].value
    # [batch_size, 3 * hidden_size]
    net = tf.concat([goal_net, thm_net, goal_net * thm_net], -1)

    # Adding 3 dense layers with dropout like in deephol
    # with tf.variable_scope("loss"):
    if is_training:
        net = tf.nn.dropout(net, rate=(1 - 0.7))
    net = tf.layers.dense(
        net, hidden_size, activation=tf.nn.relu, name="par_dense1"
    )

    if is_training:
        net = tf.nn.dropout(net, rate=(1 - 0.7))
    net = tf.layers.dense(
        net, hidden_size, activation=tf.nn.relu, name="par_dense2"
    )

    if is_training:
        net = tf.nn.dropout(net, rate=(1 - 0.7))
    net = tf.layers.dense(
        net, hidden_size, activation=tf.nn.relu, name="par_dense3"
    )

    # [batch_size, 1]
    par_logits = tf.layers.dense(net, 1, activation=None)

    tf.add_to_collection('pairwise_score', par_logits)

    # scalar
    ce_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.expand_dims((1 - tf.to_float(is_negative_labels)) * is_real_example, 1),
        logits=par_logits,
        reduction=tf.losses.Reduction.SUM)

    return ce_loss, par_logits


def create_model(
    goal_input_ids,
    thm_input_ids,
    tac_ids,
    num_tac_labels,
    is_negative_labels,
    is_training,
    is_real_example,
    vocab_size,
):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("dilated_cnn_pairwise_encoder"):
            goal_net = goal_encoding(
                vocab_size,
                goal_input_ids,
            )
            thm_net = thm_encoding(
                vocab_size,
                thm_input_ids,
            )

    with tf.variable_scope("classifier"):
        (
            tac_loss,
            tac_logits,
            tac_probabilities,
        ) = tactic_classifier(goal_net, is_training, tac_ids, num_tac_labels, is_real_example)

    with tf.variable_scope("pairwise_scorer"):
        (par_loss, par_logits) = pairwise_scorer(
            goal_net, thm_net, is_training, is_negative_labels, is_real_example
        )

    total_loss = (0.5 * par_loss) + tac_loss

    return (
        total_loss,
        tac_logits,
        tac_probabilities,
        par_loss,
        par_logits,
        tac_loss,
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
        tf.add_to_collection('goal_string', features['goal_str'])

        goal = tensor_tokenizer.tokenize(features['goal_str'], max_seq_length)
        goal = _pad_up_to(goal, max_seq_length, 1)

        features['goal_input_ids'] = goal

        tf.logging.info("********** Tokenization of theorem in Holist ********")
        tf.add_to_collection('thm_string', features['thm_str'])

        thm = tensor_tokenizer.tokenize(features['thm_str'], max_seq_length)
        thm = _pad_up_to(thm, max_seq_length, 1)

        features['thm_input_ids'] = thm


def model_fn_builder(
    num_tac_labels,
    learning_rate,
    use_tpu,
    max_seq_length,
    do_export,
    vocab_size,
):
    def model_fn(features, labels, mode, params):

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        if do_export:  # HOList has different expectation towards input features.
            update_features_using_deephol(features, max_seq_length)

        tf.logging.info("*** Features after deephol ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        goal_input_ids = features["goal_input_ids"]
        thm_input_ids = features["thm_input_ids"]
        tac_ids = features["tac_ids"]
        is_negative = features["is_negative"]

        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        (
            total_loss,
            tac_logits,
            tac_probabilities,
            par_loss,
            par_logits,
            tac_loss,
        ) = create_model(
            goal_input_ids,
            thm_input_ids,
            tac_ids,
            num_tac_labels,
            is_negative,
            is_training,
            is_real_example,
            vocab_size,
        )

        scaffold_fn = None
        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            lr = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=100000,
                decay_rate=0.98)

            opt = tf.train.AdamOptimizer(lr)
            if use_tpu:
                opt = tf.contrib.tpu.CrossShardOptimizer(opt)

            def scaffold_fn():
                return tf.train.Scaffold()

            train_op = opt.minimize(total_loss, global_step=global_step)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, scaffold_fn=scaffold_fn,
            )

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(
                tac_logits,
                par_logits,
                tac_ids,
                is_negative,
                is_real_example,
                tac_loss,
                par_loss,
            ):
                tac_predictions = tf.argmax(tac_logits, axis=-1, output_type=tf.int32)

                chosen_tac = tf.metrics.mean(tac_predictions)
                chosen_tac_acc = tf.metrics.accuracy(labels=tac_predictions, predictions=tf.ones(tf.shape(tac_predictions))*5.0)

                is_negative = tf.to_float(is_negative)

                # Tactic accuracy
                tac_accuracy = tf.metrics.accuracy(
                    labels=tac_ids, predictions=tac_predictions, weights=is_real_example * (1 - is_negative)
                )

                # Top 5 tactics accuracy
                topk_preds = tf.to_float(tf.nn.in_top_k(tac_logits, tac_ids, 5))
                tac_topk_accuracy = tf.metrics.mean(
                    values=topk_preds, weights=is_real_example * (1 - is_negative)
                )

                par_pred = tf.sigmoid(par_logits)
                pos_guess = tf.to_float(tf.greater(par_pred, 0.5))
                neg_guess = tf.to_float(tf.less(par_pred, 0.5))

                pos_acc = tf.metrics.mean(
                    values=pos_guess, weights=is_real_example * (1 - is_negative)
                )
                neg_acc = tf.metrics.mean(
                    values=neg_guess, weights=is_real_example * (is_negative)
                )

                tot_loss = (0.5 * par_loss) + tac_loss
                tot_loss = tf.metrics.mean(tot_loss)

                tac_loss = tf.metrics.mean(tac_loss)
                par_loss = tf.metrics.mean(par_loss)

                res = {
                    "pos_acc": pos_acc,
                    "neg_acc": neg_acc,
                    "tac_accuracy": tac_accuracy,
                    "tac_topk_accuracy": tac_topk_accuracy,
                    "tac_loss": tac_loss,
                    "par_loss": par_loss,
                    "total_loss": tot_loss,
                    "chosen_tactic (mean)": chosen_tac,
                    "chosen_tactic (acc)": chosen_tac_acc,
                }

                return res

            eval_metrics = (
                metric_fn,
                [
                    tac_logits,
                    par_logits,
                    tac_ids,
                    is_negative,
                    is_real_example,
                    tf.ones(params['batch_size']) * tac_loss,
                    tf.ones(params['batch_size']) * par_loss,
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


def count_records(file_path):
    ans = 0
    for _ in tf.python_io.tf_record_iterator(file_path):
        ans += 1
    return ans


def main(_):
    csv.field_size_limit(sys.maxsize)
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_export:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_export` must be True."
        )

    if FLAGS.do_train and not FLAGS.train_file:
        raise ValueError(
            "When running training you must specify train file."
        )

    if FLAGS.do_eval and not FLAGS.eval_file:
        raise ValueError(
            "When running evaluation you must specify eval file."
        )

    if FLAGS.do_export and not FLAGS.test_file:
        raise ValueError(
            "When exporting model you must specify test file."
        )

    if FLAGS.do_export and FLAGS.use_tpu:
        raise ValueError(
            "You cannot export model using TPU."
        )

    if not FLAGS.do_eval and FLAGS.infinite_eval:
        raise ValueError(
            "You cannot do eval infinitely if you are not doing it at all. Set `do_eval` to True."
        )

    tf.gfile.MakeDirs(FLAGS.output_dir)

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

    tac_labels = get_tac_labels()

    train_examples_count = 0
    num_train_steps = None

    tf.logging.info("Preparation completed!")

    if FLAGS.do_train:
        num_train_steps = 3330000

    model_fn = model_fn_builder(
        num_tac_labels=len(tac_labels),
        learning_rate=FLAGS.learning_rate,
        use_tpu=FLAGS.use_tpu,
        max_seq_length=FLAGS.max_seq_length,
        do_export=FLAGS.do_export,
        vocab_size=2000,
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=1,  # Does not matter.
    )

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", train_examples_count)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=FLAGS.train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True,
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples_count = count_records(FLAGS.eval_file)

        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. Fake examples should already be in the file.
            if eval_examples_count % FLAGS.eval_batch_size != 0:
                raise ValueError(
                    "Samples from eval file not padded to eval batch size!"
                )

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert eval_examples_count % FLAGS.eval_batch_size == 0
            eval_steps = int(eval_examples_count // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=FLAGS.eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
        )

        def run_eval():
            tf.logging.info("***** Running evaluation *****")
            tf.logging.info(
                "  Num examples = %d (with assumed padding (if running on TPU))",
                eval_examples_count,
            )
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

            output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
            with tf.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        if FLAGS.infinite_eval:
            while True:
                run_eval()
        else:
            run_eval()

    if FLAGS.do_export:
        feature_spec = {
            "goal_input_ids": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "thm_input_ids": tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.max_seq_length]),
            "tac_ids": tf.placeholder(dtype=tf.int32, shape=[None]),
            "is_negative": tf.placeholder(dtype=tf.int32, shape=[None]),
            "is_real_example": tf.placeholder(dtype=tf.int32, shape=[None]),
            "goal_str": tf.placeholder(dtype=tf.string, shape=[None]),
            "thm_str": tf.placeholder(dtype=tf.string, shape=[None]),
        }
        label_spec = {}
        build_input = tf.contrib.estimator.build_raw_supervised_input_receiver_fn
        input_receiver_fn = build_input(feature_spec, label_spec)

        predict_input_fn = build_predict_fake_input_fn(
            input_file=FLAGS.test_file,
            seq_length=FLAGS.max_seq_length,
            drop_remainder=False,
        )

        # Export final model
        tf.logging.info('*****  Starting to export model.   *****')
        save_hook = tf.train.CheckpointSaverHook(FLAGS.output_dir, save_secs=1)
        result = estimator.predict(input_fn=predict_input_fn, hooks=[save_hook])
        # As result is a generator we want to force it to calculate first value (and save checkpoint needed for export).
        for _ in result:
            break

        estimator._export_to_tpu = False
        estimator.export_savedmodel(
            export_dir_base=os.path.join(FLAGS.output_dir, 'export/best_exporter'),  # It is not best exporter,
            serving_input_receiver_fn=input_receiver_fn)                             # but HOList expects it to be.


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
