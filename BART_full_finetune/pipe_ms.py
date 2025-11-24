# pipe_ms.py
import os
import random
import numpy as np
import csv
import copy
import sys

import mindspore as ms
from mindnlp.transforms import BartTokenizer
from mindspore.dataset import GeneratorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


class Data:
    def __init__(self, args):
        set_seed(args.seed)

        # 固定最大长度
        args.max_seq_length = 128
        self.max_seq_length = args.max_seq_length

        processor = DatasetProcessor()
        self.data_dir = os.path.join("../", args.data_dir, args.dataset)

        # 所有标签
        self.all_label_list = processor.get_labels(self.data_dir)

        # 已知标签
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(
            np.array(self.all_label_list),
            self.n_known_cls,
            replace=False
        ))

        # 重新排列：已知类别 + 未知类别
        self.all_label_list_ = copy.deepcopy(self.known_label_list)
        for label in self.all_label_list:
            if label not in self.known_label_list:
                self.all_label_list_.append(label)
        self.all_label_list = self.all_label_list_

        print("[Known labels]:", self.known_label_list)
        print("[All labels reordered]:", self.all_label_list)

        assert self.known_label_list == self.all_label_list[:len(self.known_label_list)]

        # 已知标签数量
        self.num_labels = len(self.known_label_list)

        # 未知标签
        self.unseen_token = "<UNK>"
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

        # 构建样本
        self.train_examples = self.get_examples(processor, args, "train")
        self.eval_examples = self.get_examples(processor, args, "eval")
        self.test_examples = self.get_examples(processor, args, "test")

        # ----------- 使用 MindNLP 的 BART tokenizer ------------
        tokenizer = BartTokenizer.from_pretrained(args.model_name)
        self.tokenizer = tokenizer

        # 特征构建
        self.train_features = convert_examples_to_features(
            self.train_examples, self.label_list, self.max_seq_length, tokenizer, "train"
        )
        self.eval_features = convert_examples_to_features(
            self.eval_examples, self.label_list, self.max_seq_length, tokenizer, "eval"
        )
        self.test_features = convert_examples_to_features(
            self.test_examples, self.all_label_list, self.max_seq_length, tokenizer, "test"
        )

        # ----------- 构建 MindSpore Dataset --------------
        self.train_dataset = self.build_ms_dataset(self.train_features)
        self.eval_dataset = self.build_ms_dataset(self.eval_features)
        self.test_dataset = self.build_ms_dataset(self.test_features)

    # ----------------------------------------------------------------
    # 选择样本（保持原逻辑）
    # ----------------------------------------------------------------
    def get_examples(self, processor, args, mode="train"):
        ori_examples = processor.get_examples(self.data_dir, mode)

        examples = []
        if mode == "train":
            for example in ori_examples:
                if (example.label in self.known_label_list) and \
                        (np.random.uniform(0, 1) <= args.labeled_ratio):
                    examples.append(example)
        elif mode == "eval":
            for example in ori_examples:
                if example.label in self.known_label_list:
                    examples.append(example)
        elif mode == "test":
            return ori_examples

        return examples

    # ----------------------------------------------------------------
    # 构造 MindSpore Dataset（关键接口）
    # ----------------------------------------------------------------
    def build_ms_dataset(self, features):
        """
        输出 MindSpore 可直接使用的 GeneratorDataset
        """
        def generator():
            for f in features:
                yield {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "label_ids": f.label_id,
                    "labels": f.labels,
                    "label_text": f.label_text
                }

        return GeneratorDataset(
            generator,
            column_names=["input_ids", "attention_mask", "label_ids", "labels", "label_text"]
        )


# =====================================================================
# 数据结构定义
# =====================================================================

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, label_id, labels, label_text):
        self.input_ids = input_ids.astype(np.int32)
        self.attention_mask = attention_mask.astype(np.int32)
        self.label_id = np.int32(label_id)
        self.labels = labels.astype(np.int32)
        self.label_text = label_text


# =====================================================================
# 读取 TSV
# =====================================================================

class DataProcessor(object):
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        lines = []
        with open(input_file, "r", encoding="utf8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            for line in reader:
                lines.append(line)
        return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == "train":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == "eval":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "eval")
        elif mode == "test":
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        import pandas as pd
        df = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        return np.unique(df["label"].values)

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue

            guid = f"{set_type}-{i}"
            text_a = line[0]
            label = line[1]

            examples.append(InputExample(guid, text_a, None, label))

        return examples


# =====================================================================
# 特征构建（从 PyTorch → 全 numpy + MindNLP tokenizer）
# =====================================================================

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (i, example) in enumerate(examples):
        clean_label = example.label.replace("_", " ").replace("-", " ")

        # ---------------- target 模板 ----------------
        if mode == "train":
            template = f"<s> It was {clean_label} </s>"
            tg_enc = tokenizer(template,
                               max_length=32,
                               padding="max_length",
                               truncation=True)
        else:
            template = "<s> It was"
            tg_enc = tokenizer(template, truncation=True)

        # ---------------- source 输入 ----------------
        src_enc = tokenizer(example.text_a,
                            max_length=max_seq_length,
                            padding="max_length",
                            truncation=True)

        # tokenizer 输出为 python list → numpy array
        src_input_ids = np.array(src_enc["input_ids"], dtype=np.int32)
        src_attention_mask = np.array(src_enc["attention_mask"], dtype=np.int32)

        labels = np.array(tg_enc["input_ids"], dtype=np.int32)
        labels[labels == tokenizer.pad_token_id] = -100

        label_id = label_map[example.label]

        features.append(InputFeatures(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            label_id=label_id,
            labels=labels,
            label_text=clean_label,
        ))

        if i < 3:
            print("*** Example ***")
            print("input:", example.text_a)
            print("template:", template)
            print("input_ids:", src_input_ids[:20])
            print("tg_ids:", labels[:20])
            print("label:", example.label, "id =", label_id)

    return features
