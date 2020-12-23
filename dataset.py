import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict, Tuple

import torch
from torch.utils.data.dataset import Dataset

from transformers.data.processors.utils import InputFeatures, InputExample, DataProcessor
from transformers.data.processors.utils import is_tf_available, is_torch_available

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: xxxx"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    replace_token: bool = field(
        default=False, metadata={"help": "Whether to use rare tokens to replace tokens not in vocab"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class SingleSentenceClassificationProcessor(DataProcessor):
    """ Generic processor for a single sentence classification data set."""

    def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SingleSentenceClassificationProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]

    @classmethod
    def create_from_csv(
        cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
    ):
        processor = cls(**kwargs)
        processor.add_examples_from_csv(
            file_name,
            split_name=split_name,
            column_label=column_label,
            column_text=column_text,
            column_id=column_id,
            skip_first_row=skip_first_row,
            overwrite_labels=True,
            overwrite_examples=True,
        )
        return processor

    @classmethod
    def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
        processor = cls(**kwargs)
        processor.add_examples(texts_or_text_and_labels, labels=labels)
        return processor

    def add_examples_from_csv(
        self,
        file_name,
        split_name="",
        sentence_pair=False,
        has_label=True,
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        lines = self._read_tsv(file_name)
        if skip_first_row:
            lines = lines[1:]
        texts = []
        labels = []
        ids = []
        if not has_label:
            column_text = 0
        for (i, line) in enumerate(lines):
            if sentence_pair:
                texts.append(line[column_text:])
            else:
                texts.append(line[column_text])
            if has_label:
                labels.append(line[column_label])
            if column_id is not None:
                ids.append(line[column_id])
            else:
                guid = "%s-%s" % (split_name, i) if split_name else "%s" % i
                ids.append(guid)
        if not has_label:
            labels = None

        return self.add_examples(
            texts, labels, ids, overwrite_labels=overwrite_labels, overwrite_examples=overwrite_examples
        )

    def add_examples(
        self, texts, labels=None, ids=None, overwrite_labels=False, overwrite_examples=False
    ):
        assert labels is None or len(texts) == len(labels)
        assert ids is None or len(texts) == len(ids)
        if ids is None:
            ids = [None] * len(texts)
        if labels is None:
            labels = [None] * len(texts)
        examples = []
        added_labels = set()
        for (text, label, guid) in zip(texts, labels, ids):
            if isinstance(text, (tuple, list)):
                try:
                    text_a, text_b = text
                except:
                    print(text)
                    sys.exit(-1)
            else:
                text_a = text
                text_b = None
            added_labels.add(label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # Update labels
        if overwrite_labels:
            self.labels = list(added_labels)
        else:
            self.labels = list(set(self.labels).union(added_labels))

        return self.examples

    def get_features(
        self,
        tokenizer,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=None,
    ):
        """
        Convert examples in a list of ``InputFeatures``

        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.labels)}

        all_input_ids = []
        for (ex_index, example) in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)

            input_ids = tokenizer.encode(
                example.text_a, example.text_b, add_special_tokens=True, max_length=min(max_length, tokenizer.model_max_length), truncation=True
            )
            all_input_ids.append(input_ids)

        batch_length = max(len(input_ids) for input_ids in all_input_ids)

        features = []
        for (ex_index, (input_ids, example)) in enumerate(zip(all_input_ids, self.examples)):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(self.examples)))
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = batch_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
                len(input_ids), batch_length
            )
            assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
                len(attention_mask), batch_length
            )

            if self.mode == "classification":
                label = label_map[example.label]
            elif self.mode == "regression":
                label = float(example.label)
            else:
                raise ValueError(self.mode)

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, label=label))

        if return_tensors is None:
            return features
        elif return_tensors == "tf":
            if not is_tf_available():
                raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")
            import tensorflow as tf

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask}, ex.label)

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])}, tf.TensorShape([])),
            )
            return dataset
        elif return_tensors == "pt":
            if not is_torch_available():
                raise RuntimeError("return_tensors set to 'pt' but PyTorch can't be imported")
            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            if self.mode == "classification":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            elif self.mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")


class ClassifierDataset(Dataset):

    def __init__(
        self,
        args: DataTrainingArguments,
        tokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = SingleSentenceClassificationProcessor(verbose=True)
        self.output_mode = "classification"

        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_file = os.path.join(cache_dir if cache_dir is not None else args.data_dir, "label.txt")
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            start = time.time()
            self.features = torch.load(cached_features_file)
            self.label_list = [line.strip() for line in open(label_file)]
            logger.info(
                f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
            )
            for feat in self.features[:5]:
                logger.info("*** Features ***")
                logger.info(f"features: {feat}")
        else:
            logger.info(f"Creating features from dataset file at {args.data_dir}")

            data_file = f"{self.args.data_dir}/{mode.value}.txt"
            examples = self.processor.add_examples_from_csv(
                data_file,
                has_label=False if mode.value == "test" else True,
                sentence_pair=True if "nli" in args.task_name else False)
            if limit_length is not None:
                examples = examples[:limit_length]
            self.features = self.processor.get_features(tokenizer, args.max_seq_length, return_tensors=None)
            self.label_list = self.processor.labels
            if mode.value == "train":
                with open(label_file, "w") as fw:
                    for label in self.processor.labels:
                        fw.write(label + "\n")
            start = time.time()
            torch.save(self.features, cached_features_file)
            logger.info(
                "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


@dataclass
class InputNerExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputNerFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class NerDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        labels: List[str],
        max_seq_length: Optional[int] = None,
        replace_map: Dict[str, str] = {},
        never_tokenize_words: List[str] = [],
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)

        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            examples = read_ner_examples_from_file(data_dir, mode, replace_map)
            self.features = convert_exmaples_to_features(
                examples, labels, max_seq_length, tokenizer, never_tokenize_words)
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputNerFeatures:
        return self.features[i]

def read_ner_examples_from_file(data_dir, mode: Union[Split, str], replace_map: Dict[str, str]) -> List[InputNerExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line == "\n":
                examples.append(InputNerExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                guid_index += 1
                words = []
                labels = []
            else:
                splits = line.split("\t", 1)
                if splits[0] == " ":
                    word = "[unused1]"
                elif splits[0] in replace_map:
                    word = replace_map[splits[0]]
                else:
                    word = splits[0]
                words.append(word)
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputNerExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    return examples

def convert_exmaples_to_features(
    examples: List[InputNerExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer,
    never_tokenize_words=[],
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True
) -> List[InputNerFeatures]:

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            if word in never_tokenize_words:
                word_tokens = [word]
            else:
                word_tokens = tokenizer.tokenize(word)

            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(
            InputNerFeatures(
                input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
            )
        )
    return features
