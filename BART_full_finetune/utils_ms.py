# utils_ms.py

import argparse

def load_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--save_results_path", type=str, default='outputs')
    parser.add_argument("--pretrain_dir", default='models', type=str)
    parser.add_argument("--model_name", default="facebook/bart-base", type=str)
    parser.add_argument("--max_seq_length", default=None, type=int)
    parser.add_argument("--dataset", default='banking', type=str)
    parser.add_argument("--known_cls_ratio", default=0.75, type=float)
    parser.add_argument("--labeled_ratio", default=1.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="3")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--feat_dim", default=768, type=int)
    parser.add_argument("--lr_boundary", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", default=30.0, type=float)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--wait_patient", default=10, type=int)
    parser.add_argument("--p_node", default=0.4, type=float)
    parser.add_argument("--lamda_loss", default=0.5, type=float)
    parser.add_argument("--delta", default=1.0, type=float)

    args = parser.parse_args()
    return args


def get_unkdataset_of_dataset(dataset, unk_label_id):
    """
    在 MindSpore 设置中，dataset 是 Python list(dict)，不再是 fastNLP 的 DataSet。

    dataset: List[dict]
    """
    unk_list = []

    for item in dataset:
        if item["label_ids"] >= unk_label_id:
            unk_list.append({
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"],
                "label_ids": item["label_ids"]
            })

    return unk_list
