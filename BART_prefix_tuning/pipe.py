import os
import random
import numpy as np
import csv
import copy
import sys

import mindspore as ms
from mindspore import Tensor
from transformers import BartTokenizer
from fastNLP import DataSet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


class Data:

    def __init__(self, args):
        set_seed(args.seed)
        args.max_seq_length = 128
        self.max_seq_length = args.max_seq_length

        processor = DatasetProcessor()
        self.data_dir = os.path.join("../", args.data_dir, args.dataset)

        # 读取所有标签
        self.all_label_list = processor.get_labels(self.data_dir)

        # 选择 known labels
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(
            np.random.choice(
                np.array(self.all_label_list),
                self.n_known_cls,
                replace=False
            )
        )

        # 重新排序 label list：known + unseen
        self.all_label_list_ = copy.deepcopy(self.known_label_list)
        for label in self.all_label_list:
            if label not in self.known_label_list:
                self.all_label_list_.append(label)
        self.all_label_list = self.all_label_list_

        print(self.known_label_list)
        print(self.all_label_list)
        assert self.known_label_list == self.all_label_list[:len(self.known_label_list)]

        self.num_labels = len(self.known_label_list)

        # unseen label
        self.unseen_token = '<UNK>'
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

        # 加载数据
        self.train_examples = self.get_examples(processor, args, 'train')
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')

        self.tokenizer = BartTokenizer.from_pretrained(args.model_name)

        # Dataset（fastNLP 的 DataSet）
        self.train_dataset = self.get_loader(self.train_examples, args, self.tokenizer, 'train')
        self.eval_dataset  = self.get_loader(self.eval_examples,  args, self.tokenizer, 'eval')
        self.test_dataset  = self.get_loader(self.test_examples,  args, self.tokenizer, 'test')



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, label_id, labels, label_text):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_id = label_id
        self.labels = labels
        self.label_text = label_text



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir, mode)

        examples = []
        if mode == 'train':
            for example in ori_examples:
                if (example.label in self.known_label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    examples.append(example)
        elif mode == 'eval':
            for example in ori_examples:
                if example.label in self.known_label_list:
                    examples.append(example)
        elif mode == 'test':
            examples = ori_examples

        return examples

    def get_loader(self, examples, args, tokenizer, mode='train'):
        if mode == 'test':
            features = convert_examples_to_features(
                examples, self.all_label_list,
                args.max_seq_length, tokenizer, mode
            )
        else:
            features = convert_examples_to_features(
                examples, self.label_list,
                args.max_seq_length, tokenizer, mode
            )

        input_ids = [f.input_ids for f in features]
        input_mask = [f.attention_mask for f in features]
        label_ids = [f.label_id for f in features]
        labels = [f.labels for f in features]
        label_text = [f.label_text for f in features]

        dataset = DataSet({
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'label_ids': label_ids,
            'labels': labels,
            'label_text': label_text
        })

        return dataset

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    for (ex_index, example) in enumerate(examples):

        clean_label = example.label.replace("_", " ").replace("-", " ")

        # target template
        if mode == 'train':
            template_text = "<s> It was {} </s>".format(clean_label)
            with tokenizer.as_target_tokenizer():
                tg_inputs = tokenizer(
                    template_text,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=32,
                    add_special_tokens=False
                )
        else:
            template_text = "<s> It was"
            with tokenizer.as_target_tokenizer():
                tg_inputs = tokenizer(
                    template_text,
                    return_tensors='pt',
                    add_special_tokens=False
                )

        inputs = tokenizer(
            text=example.text_a,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_seq_length
        )

        # ---- 将 torch tensor → numpy → python list ----
        src_input_ids = inputs["input_ids"].squeeze().cpu().numpy().tolist()
        src_attention_mask = inputs["attention_mask"].squeeze().cpu().numpy().tolist()

        labels = tg_inputs["input_ids"].squeeze().cpu().numpy()
        labels[labels == tokenizer.pad_token_id] = -100
        labels = labels.tolist()

        label_id = label_map[example.label]
        label_text = clean_label

        if ex_index < 5:
            print("*** Example ***")
            print("tokens:", example.text_a)
            print("output:", template_text)

        features.append(
            InputFeatures(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                label_id=label_id,
                labels=labels,
                label_text=label_text
            )
        )

    return features
