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

import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path", None, "Path to input file. It should be in json format.",
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
    "eval_batch_size",
    8,
    "Total batch size for eval."
    "Important, because we need to pad with fake samples to match it.",
)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, thm):
        """Constructs an InputExample.

    Args:
      guid: Unique id for the example.
      goal: The untokenized goal string
      thm:  The untokenized theorem string.
      tac_id: id of tactic for the goal
    """
        self.guid = guid
        self.thm = thm


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
     See run_classifier.py for details.
  """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        thm_input_ids,
        thm_input_mask,
        thm_segment_ids,
    ):
        self.thm_input_ids = thm_input_ids
        self.thm_input_mask = thm_input_mask
        self.thm_segment_ids = thm_segment_ids


class DeepholProcessor:
    """Processor for Deephol dataset"""

    def get_examples(self, data_path):
        with open(data_path, "r") as f:
            data = f.readlines()

        tf.logging.info("Dataset read successful!")

        return self._create_examples(data)

    def _create_examples(self, data):
        examples = []
        for (i, sample) in enumerate(data):
            guid = str(i)

            thm = tokenization.convert_to_unicode(sample)

            examples.append(
                InputExample(guid=guid, thm=thm)
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
    ex_index, example, max_seq_length, tokenizer
):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            thm_input_ids=
                [0] * max_seq_length
            ,
            thm_input_mask=
                [0] * max_seq_length
            ,
            thm_segment_ids=
                [0] * max_seq_length
            ,
        )

    tokens = tokenizer.tokenize(example.thm)

    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0 : (max_seq_length - 2)]

    (thm_tokens, thm_input_ids, thm_input_mask, thm_segment_ids) = convert_tokens(
        tokens, tokenizer, max_seq_length
    )

    assert len(thm_input_ids) == max_seq_length
    assert len(thm_input_mask) == max_seq_length
    assert len(thm_segment_ids) == max_seq_length

    if ex_index < 2:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info(
            "thm_tokens: %s"
            % " ".join([tokenization.printable_text(x) for x in thm_tokens])
        )
        tf.logging.info(
            "thm_input_ids: %s" % " ".join([str(x) for x in thm_input_ids])
        )

        tf.logging.info(
            "thm_input_mask: %s" % " ".join([str(x) for x in thm_input_mask])
        )
        tf.logging.info(
            "thm_segment_ids: %s" % " ".join([str(x) for x in thm_segment_ids])
        )

    feature = InputFeatures(
        thm_input_ids=thm_input_ids,
        thm_input_mask=thm_input_mask,
        thm_segment_ids=thm_segment_ids,
    )

    return feature


def file_based_convert_examples_to_features(
    examples, max_seq_length, tokenizer, output_file
):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(
            ex_index,
            example,
            max_seq_length,
            tokenizer,
        )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.thm_input_ids)
        features["input_mask"] = create_int_feature(feature.thm_input_mask)
        features["segment_ids"] = create_int_feature(feature.thm_segment_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processor = DeepholProcessor()
    tokenizer = tokenization.LongestTokenizer(vocab=FLAGS.vocab_file)

    examples = processor.get_examples(FLAGS.data_path)

    file_based_convert_examples_to_features(
        examples,
        FLAGS.max_seq_length,
        tokenizer,
        FLAGS.output_path,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("data_path")
    flags.mark_flag_as_required("output_path")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
