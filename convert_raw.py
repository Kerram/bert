# Code used to convert tsv files to tf_record files used by run_deephol.py fine tuning.
# Code is a modified part of older run_deephol.py, which is and was a mix of
# bert's classifier.py and deephol's architectures.py.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import sys
import json
import random

import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path", None, "Path to input file. It should be in json format.",
)

flags.DEFINE_string(
    "theorems_path", None, "Path to a file with all theorems."
)

flags.DEFINE_string(
    "vocab_file", None, "The vocabulary file that the BERT model was trained on."
)

flags.DEFINE_string(
    "output_path", None, "Path where we will save tf_record file.",
)

flags.DEFINE_string(
    "set_type",
    None,
    "Flag specifying whether we are to convert train, valid or test set.",
)

flags.DEFINE_integer(
    "max_seq_length",
    512,
    "The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.",
)

flags.DEFINE_integer(
    "max_number_of_theorems",
    32,
    "The maximum number of theorems used as tactic arguments. Goals with less will be padded, with more truncated.",
)

flags.DEFINE_integer(
    "eval_batch_size",
    8,
    "Total batch size for eval."
    "Important, because we need to pad with fake samples to match it.",
)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, goal, thms, my_negatives, tac_id=None):
        """Constructs an InputExample.

    Args:
      guid: Unique id for the example.
      goal: The untokenized goal string
      thm:  The untokenized theorem string.
      tac_id: id of tactic for the goal
    """
        self.guid = guid
        self.goal = goal
        self.thms = thms
        self.tac_id = tac_id
        self.my_negatives = my_negatives


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
        thms_input_ids,
        thms_input_mask,
        thms_segment_ids,
        tac_id,
        goal_str,
        thms_str,
        number_of_theorems,
        negatives,
        is_real_example=True,
    ):

        self.goal_input_ids = goal_input_ids
        self.goal_input_mask = goal_input_mask
        self.goal_segment_ids = goal_segment_ids
        self.thms_input_ids = thms_input_ids
        self.thms_input_mask = thms_input_mask
        self.thms_segment_ids = thms_segment_ids
        self.tac_id = tac_id
        self.is_real_example = is_real_example
        self.goal_str = goal_str
        self.thms_str = thms_str
        self.number_of_theorems = number_of_theorems
        self.negatives = negatives


def get_negatives(all_thms, no_negative, no_samples):
    map(tokenization.convert_to_unicode, all_thms)

    ans = []
    index = 0
    negatives = []
    for _ in range(no_samples * no_negative):
        if index == 0:
            random.shuffle(all_thms)

        negatives.append(all_thms[index])

        index += 1
        if index == len(all_thms):
            index = 0

        if len(negatives) == no_negative:
            ans.append(negatives)
            negatives = []

    return ans


class DeepholProcessor:
    """Processor for Deephol dataset"""

    def get_examples(self, data_path, set_type, theorems_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        tf.logging.info("Dataset read successful!")

        return self._create_examples(data, set_type, theorems_path)

    def get_tac_labels(self):
        return [i for i in range(41)]

    def _create_examples(self, data, set_type, theorems_path):
        with open(theorems_path, "r") as f:
            all = f.readlines()

        all_thms = get_negatives(all, 7, 5 * len(data))

        examples = []
        for (i, sample) in enumerate(data):
            guid = "%s-%s" % (set_type, i)

            my_negatives = []
            for j in range(5):
                my_negatives.append(all_thms[i + j * len(data)])

            if set_type == "test":
                #  The values really don't matter, because we are using test set only as a hack to export a model.
                goal = tokenization.convert_to_unicode(sample["goal"])
                thms = list(map(tokenization.convert_to_unicode, sample["thms"]))
                tac_id = 0
            else:
                goal = tokenization.convert_to_unicode(sample["goal"])
                thms = list(map(tokenization.convert_to_unicode, sample["thms"]))
                tac_id = sample["tac_id"]
            examples.append(
                InputExample(guid=guid, goal=goal, thms=thms, tac_id=tac_id, my_negatives=my_negatives)
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

    return res, input_ids, input_mask, segment_ids


def convert_single_example(
    ex_index, example, tac_label_list, max_seq_length, tokenizer, max_number_of_theorems
):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            goal_input_ids=[0] * max_seq_length,
            goal_input_mask=[0] * max_seq_length,
            goal_segment_ids=[0] * max_seq_length,
            thms_input_ids=[
                [0] * max_seq_length for _ in range(max_number_of_theorems)
            ],
            thms_input_mask=[
                [0] * max_seq_length for _ in range(max_number_of_theorems)
            ],
            thms_segment_ids=[
                [0] * max_seq_length for _ in range(max_number_of_theorems)
            ],
            tac_id=0,
            is_real_example=False,
            goal_str="",
            thms_str="",
            number_of_theorems=max_number_of_theorems,
            negatives=[0] * max_seq_length * 7 * 5,
        )

    if len(example.thms) > max_number_of_theorems:
        example.thms = example.thms[:max_number_of_theorems]

    if len(example.thms) == 0:
        example.thms = [""]

    number_of_theorems = len(example.thms)

    tac_label_map = {}
    for (i, label) in enumerate(tac_label_list):
        tac_label_map[label] = i

    g_tokens = tokenizer.tokenize(example.goal)
    t_tokens = [tokenizer.tokenize(thm) for thm in example.thms]

    if len(g_tokens) > max_seq_length - 2:
        g_tokens = g_tokens[0 : (max_seq_length - 2)]

    for i, tokens in enumerate(t_tokens):
        if len(tokens) > max_seq_length - 2:
            t_tokens[i] = tokens[0 : (max_seq_length - 2)]

    (goal_tokens, goal_input_ids, goal_input_mask, goal_segment_ids) = convert_tokens(
        g_tokens, tokenizer, max_seq_length
    )

    assert len(goal_input_ids) == max_seq_length
    assert len(goal_input_mask) == max_seq_length
    assert len(goal_segment_ids) == max_seq_length

    thms_tokens = []
    thms_input_ids = []
    thms_input_mask = []
    thms_segment_ids = []
    for tokens in t_tokens:
        (thm_tokens, thm_input_ids, thm_input_mask, thm_segment_ids) = convert_tokens(
            tokens, tokenizer, max_seq_length
        )

        assert len(thm_input_ids) == max_seq_length
        assert len(thm_input_mask) == max_seq_length
        assert len(thm_segment_ids) == max_seq_length

        thms_tokens.append(thm_tokens)
        thms_input_ids.append(thm_input_ids)
        thms_input_mask.append(thm_input_mask)
        thms_segment_ids.append(thm_segment_ids)

    while len(thms_input_ids) < max_number_of_theorems:
        thms_input_ids.append([0] * max_seq_length)
        thms_input_mask.append([0] * max_seq_length)
        thms_segment_ids.append([0] * max_seq_length)

    assert len(thms_input_ids) == max_number_of_theorems
    assert len(thms_input_mask) == max_number_of_theorems
    assert len(thms_segment_ids) == max_number_of_theorems

    tac_id = tac_label_map[example.tac_id]

    if ex_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info(example.my_negatives)
        tf.logging.info(
            "goal_tokens: %s"
            % " ".join([tokenization.printable_text(x) for x in goal_tokens])
        )
        tf.logging.info(
            "number_of_theorems: %d" % (number_of_theorems,)
        )
        tf.logging.info(
            "goal_input_ids: %s" % " ".join([str(x) for x in goal_input_ids])
        )

        tf.logging.info(
            "goal_input_mask: %s" % " ".join([str(x) for x in goal_input_mask])
        )
        tf.logging.info(
            "goal_segment_ids: %s" % " ".join([str(x) for x in goal_segment_ids])
        )

        if number_of_theorems > 0:
            tf.logging.info("thm_input_ids: %s" % " ".join([str(x) for x in thms_input_ids[0]]))
            tf.logging.info(
                "thm_input_mask: %s" % " ".join([str(x) for x in thms_input_mask[0]])
            )
            tf.logging.info(
                "thm_segment_ids: %s" % " ".join([str(x) for x in thms_segment_ids[0]])
            )

        tf.logging.info("tac_id: %d" % (tac_id,))

    negatives = []
    for epoch in example.my_negatives:
        for negative in epoch:
            neg_tokens = tokenizer.tokenize(negative)
            if len(neg_tokens) > max_seq_length - 2:
                neg_tokens = neg_tokens[0: (max_seq_length - 2)]

            (_, negatives_ids, _, _) = convert_tokens(
                neg_tokens, tokenizer, max_seq_length
            )
            assert len(negatives_ids) == max_seq_length

            negatives += negatives_ids

    assert len(goal_input_ids) == max_seq_length
    assert len(goal_input_mask) == max_seq_length
    assert len(goal_segment_ids) == max_seq_length
    assert len(negatives) == max_seq_length * 7 * 5

    feature = InputFeatures(
        goal_input_ids=goal_input_ids,
        goal_input_mask=goal_input_mask,
        goal_segment_ids=goal_segment_ids,
        thms_input_ids=thms_input_ids,
        thms_input_mask=thms_input_mask,
        thms_segment_ids=thms_segment_ids,
        tac_id=tac_id,
        is_real_example=True,
        goal_str=example.goal,
        thms_str=example.thms,
        number_of_theorems=number_of_theorems,
        negatives=negatives,
    )

    return feature


def file_based_convert_examples_to_features(
    examples, tac_label_list, max_seq_length, tokenizer, output_file, max_number_of_theorems
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
            max_seq_length,
            tokenizer,
            max_number_of_theorems
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_matrix_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=[value for l in values for value in l]))
            return f

        features = collections.OrderedDict()
        features["goal_input_ids"] = create_int_feature(feature.goal_input_ids)
        features["goal_input_mask"] = create_int_feature(feature.goal_input_mask)
        features["goal_segment_ids"] = create_int_feature(feature.goal_segment_ids)
        features["thm_input_ids"] = create_matrix_feature(feature.thms_input_ids)
        features["thm_input_mask"] = create_matrix_feature(feature.thms_input_mask)
        features["thm_segment_ids"] = create_matrix_feature(feature.thms_segment_ids)
        features["tac_ids"] = create_int_feature([feature.tac_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        features["number_of_theorems"] = create_int_feature([feature.number_of_theorems])
        features['negatives'] = create_int_feature(feature.negatives)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def test_set_convert_examples_to_features(
    examples, tac_label_list, max_seq_length, tokenizer, output_file, max_number_of_theorems
):
    """Convert a set of `InputExample`s to a TFRecord file.
    In case of a test set we add goal and theorem strings as additional fetures."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(
            ex_index,
            example,
            tac_label_list,
            max_seq_length,
            tokenizer,
            max_number_of_theorems,
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        def create_matrix_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=[value for l in values for value in l]))
            return f

        features = collections.OrderedDict()
        features["goal_input_ids"] = create_int_feature(feature.goal_input_ids)
        features["goal_input_mask"] = create_int_feature(feature.goal_input_mask)
        features["goal_segment_ids"] = create_int_feature(feature.goal_segment_ids)
        features["thm_input_ids"] = create_matrix_feature(feature.thms_input_ids)
        features["thm_input_mask"] = create_matrix_feature(feature.thms_input_mask)
        features["thm_segment_ids"] = create_matrix_feature(feature.thms_segment_ids)
        features["tac_ids"] = create_int_feature([feature.tac_id])
        features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
        features["number_of_theorems"] = create_int_feature([feature.number_of_theorems])
        features['negatives'] = create_int_feature(feature.negatives)

        features["goal_str"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[bytes(feature.goal_str, encoding="utf-8")]
            )
        )

        features["thm_str"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[bytes(feature.thms_str[0], encoding="utf-8")]
            )
        )

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processor = DeepholProcessor()
    tokenizer = tokenization.LongestTokenizer(vocab=FLAGS.vocab_file)

    examples = processor.get_examples(FLAGS.data_path, FLAGS.set_type, FLAGS.theorems_path)

    tac_labels = processor.get_tac_labels()

    if FLAGS.set_type == "eval":
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on. These do NOT count towards the metric (all tf.metrics
        # support a per-instance weight, and these get a weight of 0.0).
        while len(examples) % FLAGS.eval_batch_size != 0:
            examples.append(PaddingInputExample())

    if FLAGS.set_type == "test":
        test_set_convert_examples_to_features(
            examples,
            tac_labels,
            FLAGS.max_seq_length,
            tokenizer,
            FLAGS.output_path,
            FLAGS.max_number_of_theorems,
        )
    else:
        file_based_convert_examples_to_features(
            examples,
            tac_labels,
            FLAGS.max_seq_length,
            tokenizer,
            FLAGS.output_path,
            FLAGS.max_number_of_theorems,
        )


if __name__ == "__main__":
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("output_path")
    flags.mark_flag_as_required("set_type")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required('theorems_path')
    tf.app.run()
