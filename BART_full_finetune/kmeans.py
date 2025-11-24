import os
import sys
sys.path.append("../")

import pickle
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score, adjusted_rand_score,
    adjusted_mutual_info_score, confusion_matrix
)

from mindspore import Tensor, ops
import mindspore.dataset as ds

from BART_full_finetune.pipe_ms import Data, set_seed       # ← 你需要迁移
from BART_full_finetune.model_ms import Model               # ← 你需要迁移
from BART_full_finetune.utils_ms import load_parameters


# ========================== 工具函数 ===============================

def hungray_aligment(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment

    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D))
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(np.array(y_true), np.array(y_pred))
    return sum([w[i, j] for i, j in ind]) / len(y_pred)


def clustering_score(y_true, y_pred):
    return {
        "ACC": round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
        "ARI": round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        "NMI": round(normalized_mutual_info_score(y_true, y_pred) * 100, 2),
        "AMI": round(adjusted_mutual_info_score(y_true, y_pred) * 100, 2)
    }


def save_results(arg, test_results):
    if not os.path.exists(arg.save_results_path):
        os.makedirs(arg.save_results_path)

    var = [arg.dataset, arg.known_cls_ratio, arg.labeled_ratio, arg.seed, arg.delta]
    names = ["dataset", "known_cls_ratio", "labeled_ratio", "seed", "delta"]

    results = {**test_results, **dict(zip(names, var))}

    file_path = os.path.join(arg.save_results_path, "kmeans_results.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.append(results, ignore_index=True)
    else:
        df = pd.DataFrame([results])

    df.to_csv(file_path, index=False)
    print("Saved:", file_path)
    print(df)


def ood_of_ind_ratio_(y_true, threshold):
    y_true = np.array(y_true)
    return float(np.sum(y_true < threshold) / len(y_true))


# ========================== 主流程（MindSpore） ===============================

if __name__ == "__main__":
    args = load_parameters()
    set_seed(args.seed)

    # ------------------ 载入 Dataset ------------------
    datafile = f"datafile/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}"
    if not os.path.exists(datafile):
        os.makedirs(datafile)

    data = Data(args)
    args.num_labels = data.num_labels

    # ------------------ 载入 noisy-unk data ------------------
    if args.dataset == "clinc":
        unk_path = f"unkdataset_with_noise/unkdataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}-delta-{args.delta}.pkl"
    else:
        unk_path = f"unkdataset_with_noise/unkdataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-train_batch_size-{args.train_batch_size}.pkl"

    with open(unk_path, "rb") as fp:
        unkDs = pickle.load(fp)["data"]

    # ------------------ MindSpore Dataset ------------------
    def gen():
        for item in unkDs:
            yield {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "label_ids": item["label_ids"],
                "labels": item["labels"],
            }

    dl = ds.GeneratorDataset(gen, column_names=["input_ids", "attention_mask", "label_ids", "labels"])
    dl = dl.batch(32)

    # ------------------ 载入 MindSpore 模型 ------------------
    model = Model(args)
    ckpt_path = f"model/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}/checkpoint.ckpt"

    import mindspore as ms
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)

    # ------------------ 提取句子表示 ------------------
    label_list = []
    sent_vecs = []

    for batch in tqdm(dl, desc="Extract feat"):
        input_ids = Tensor(batch["input_ids"], ms.int32)
        attention_mask = Tensor(batch["attention_mask"], ms.int32)
        label_ids = batch["label_ids"]

        # 收集真实标签
        label_list.extend(label_ids.asnumpy().tolist())

        # Forward
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=Tensor(batch["labels"], ms.int32),
                    label_ids=None)

        feat = out["sent_emb"]   # (B, hidden)
        sent_vecs.append(feat.asnumpy())

    sent_vecs = np.concatenate(sent_vecs, axis=0)

    # ------------------ KMeans 聚类 ------------------
    y_true = np.array(label_list)

    cluster_num = len(data.all_label_list) * 3
    km = KMeans(n_clusters=cluster_num).fit(sent_vecs)
    y_pred = km.labels_

    print("Cluster num:", len(np.unique(y_pred)))

    # ------------------ 输出聚类可视化 ------------------
    results = clustering_score(y_true, y_pred)
    print("clustering:", results)

    print("OOD IND ratio:", ood_of_ind_ratio_(y_true, data.unseen_token_id))

    save_results(args, results)
