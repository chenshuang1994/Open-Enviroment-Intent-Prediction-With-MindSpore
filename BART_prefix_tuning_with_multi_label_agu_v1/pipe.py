import os
import copy
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BartTokenizer

import mindspore as ms
from mindspore import Tensor


#########################################
# Simple replacement for fastNLP.DataSet
#########################################
class SimpleDataset:
    def __init__(self, data_dict):
        """
        data_dict: {field: [list_of_samples]}
        """
        self.data = data_dict
        self.keys = list(data_dict.keys())
        self.size = len(list(data_dict.values())[0])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {k: self.data[k][idx] for k in self.keys}

    def as_list(self):
        return self.data


#########################################
# Random Seed
#########################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


#########################################
# Load training data
#########################################
def load_train_label_and_ds(root, dataset_name):
    filepath = os.path.join(root, f"{dataset_name}_expand_all_5.txt")
    example = []
    label_unique = []
    multi_label_map = {}

    with open(filepath, "r", encoding="utf8") as fp:
        for idx, line in enumerate(fp):
            txt = line.strip()
            txt = txt.split("###")
            text = txt[0]
            multi_labels = txt[1].split(',')
            last_label = multi_labels.pop()    # final one is real class
            label_unique.append(last_label)

            # keep only first 4 labels
            if len(multi_labels) < 4:
                raise ValueError(f"Bad line {idx}")
            multi_labels = multi_labels[:4]

            # mapping for class → multi-labels
            if last_label not in multi_label_map:
                multi_label_map[last_label] = copy.deepcopy(multi_labels)
            else:
                multi_label_map[last_label].extend(multi_labels)

            example.append((text, multi_labels, last_label))

    # unique main labels
    label_unique = np.unique(label_unique)

    # Ensure each main label → exactly 4 multi-label
    for k, v in multi_label_map.items():
        v2 = np.unique(v)
        if len(v2) != 4:
            raise ValueError("multi-label mismatch")
        multi_label_map[k] = v2

    return example, label_unique, multi_label_map


#########################################
# Load dev / test
#########################################
def load_dev_test_ds(root, mode):
    if mode == 'dev':
        filepath = os.path.join(root, "dev.tsv")
    else:
        filepath = os.path.join(root, "test.tsv")

    df = pd.read_csv(filepath, sep="\t", header=0)
    example = []
    for _, row in df.iterrows():
        example.append((row["text"], row["label"]))
    return example


#########################################
# Filter by known-class labels
#########################################
def filter_ds_by_ind_label(example, known_label_list, mode):
    """Remove OOD samples except in test"""
    new = []
    if mode == 'train':
        for tpl in example:
            if tpl[2] in known_label_list:
                new.append(tpl)
    elif mode == 'dev':
        for tpl in example:
            if tpl[1] in known_label_list:
                new.append(tpl)
    elif mode == 'test':
        return example
    return new


#########################################
# Convert examples → features
#########################################
def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    known_multi_label_mapping,
    known_multi_label_unique,
    mode
):
    label_map = {lab: i for i, lab in enumerate(label_list)}
    features = []
    mlb = MultiLabelBinarizer(classes=known_multi_label_unique)

    for idx, ex in enumerate(examples):

        if mode == 'train':
            text, multi_labels, last_label = ex
            enc = tokenizer(text=text,
                            max_length=max_seq_length,
                            truncation=True,
                            padding='max_length',
                            return_tensors="np")

            src_ids = enc["input_ids"][0]
            src_mask = enc["attention_mask"][0]

            label_ids_list = []
            for mlab in multi_labels:
                clean = mlab.replace("_", " ").replace("-", " ")
                template = f"<s> It was {clean} </s>"
                tg = tokenizer(text_target=template,
                               max_length=32,
                               truncation=True,
                               padding="max_length",
                               return_tensors="np")
                lab_ids = tg["labels"][0] if "labels" in tg else tg["input_ids"][0]
                lab_ids[lab_ids == tokenizer.pad_token_id] = -100
                label_ids_list.append(lab_ids)

            multi_hot = mlb.fit_transform([tuple(multi_labels)])[0]
            label_id = label_map[last_label]
            label_text = last_label.replace("_", " ").replace("-", " ")

            features.append({
                "input_ids": Tensor(src_ids, ms.int32),
                "attention_mask": Tensor(src_mask, ms.int32),
                "multi_labels": Tensor(multi_hot, ms.float32),
                "label_list": [Tensor(x, ms.int32) for x in label_ids_list],
                "label_ids": Tensor(label_id, ms.int32),
                "label_text": label_text
            })

        elif mode == "dev":
            text, last_label = ex
            multi_labels = known_multi_label_mapping[last_label]

            enc = tokenizer(text=text,
                            max_length=max_seq_length,
                            truncation=True,
                            padding='max_length',
                            return_tensors="np")
            src_ids = enc["input_ids"][0]
            src_mask = enc["attention_mask"][0]

            tg = tokenizer(text_target="<s> It was",
                           add_special_tokens=False,
                           return_tensors="np")
            lab_ids = tg["labels"][0] if "labels" in tg else tg["input_ids"][0]
            lab_ids[lab_ids == tokenizer.pad_token_id] = -100

            multi_hot = mlb.fit_transform([tuple(multi_labels)])[0]
            label_id = label_map[last_label]

            features.append({
                "input_ids": Tensor(src_ids, ms.int32),
                "attention_mask": Tensor(src_mask, ms.int32),
                "multi_labels": Tensor(multi_hot, ms.float32),
                "label_list": [Tensor(lab_ids, ms.int32)],
                "label_ids": Tensor(label_id, ms.int32),
                "label_text": last_label
            })

        else:  # test
            text, last_label = ex

            enc = tokenizer(text=text,
                            max_length=max_seq_length,
                            truncation=True,
                            padding='max_length',
                            return_tensors="np")
            src_ids = enc["input_ids"][0]
            src_mask = enc["attention_mask"][0]

            tg = tokenizer(text_target="<s> It was",
                           add_special_tokens=False,
                           return_tensors="np")
            lab_ids = tg["input_ids"][0]
            lab_ids[lab_ids == tokenizer.pad_token_id] = -100

            label_id = label_map[last_label]

            features.append({
                "input_ids": Tensor(src_ids, ms.int32),
                "attention_mask": Tensor(src_mask, ms.int32),
                "multi_labels": Tensor(label_id, ms.int32),
                "label_list": [Tensor(lab_ids, ms.int32)],
                "label_ids": Tensor(label_id, ms.int32),
                "label_text": last_label
            })

    return features


#########################################
# Pipe class in MindSpore
#########################################
class Pipe:

    def __init__(self, args):
        set_seed(args.seed)
        args.max_seq_length = 128

        data_dir = os.path.join("../", args.data_dir, args.dataset)

        # Load training data
        train_ex, all_labels, multi_label_map = load_train_label_and_ds(data_dir, args.dataset)
        self.multi_label_mapping = multi_label_map
        self.all_label_list = all_labels

        # Split known / unknown
        self.n_known_cls = round(len(all_labels) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(all_labels, self.n_known_cls, replace=False))

        self.known_multi_label_mapping = {
            lab: self.multi_label_mapping[lab] for lab in self.known_label_list
        }
        known_multi_unique = np.unique([m for lab in self.known_label_list
                                           for m in self.multi_label_mapping[lab]])
        self.known_multi_label_unique = known_multi_unique

        # Full label list (known + OOD)
        full_list = list(self.known_label_list) + \
                    [l for l in all_labels if l not in self.known_label_list]
        self.all_label_list = full_list

        # Load dev & test
        dev_ex = load_dev_test_ds(data_dir, 'dev')
        test_ex = load_dev_test_ds(data_dir, 'test')

        self.train_example = filter_ds_by_ind_label(train_ex, self.known_label_list, 'train')
        self.dev_example = filter_ds_by_ind_label(dev_ex, self.known_label_list, 'dev')
        self.test_example = filter_ds_by_ind_label(test_ex, self.known_label_list, 'test')

        # Load tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(args.model_name)

        # UNK label
        self.num_labels = len(self.known_label_list)
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + ["<UNK>"]

        # Build datasets
        self.train_dataset = self.get_loader(self.train_example, args, 'train')
        self.eval_dataset = self.get_loader(self.dev_example, args, 'dev')
        self.test_dataset = self.get_loader(self.test_example, args, 'test')

    def get_loader(self, examples, args, mode):
        if mode == 'test':
            feats = convert_examples_to_features(
                examples,
                self.all_label_list,
                args.max_seq_length,
                self.tokenizer,
                self.multi_label_mapping,
                self.known_multi_label_unique,
                mode
            )
        else:
            feats = convert_examples_to_features(
                examples,
                self.label_list,
                args.max_seq_length,
                self.tokenizer,
                self.known_multi_label_mapping,
                self.known_multi_label_unique,
                mode
            )

        # pack to SimpleDataset
        data_dict = {
            "input_ids": [f["input_ids"] for f in feats],
            "attention_mask": [f["attention_mask"] for f in feats],
            "multi_labels": [f["multi_labels"] for f in feats],
            "label_ids": [f["label_ids"] for f in feats],
            "labels": [f["label_list"] for f in feats],
            "label_text": [f["label_text"] for f in feats]
        }
        return SimpleDataset(data_dict)
