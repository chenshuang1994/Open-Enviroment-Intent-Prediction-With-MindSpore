# GenerateOOD_ms.py  (MindSpore version)

import os
import pickle
import functools
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from string import punctuation
from scipy.optimize import linear_sum_assignment
from typing import List

import mindspore as ms
from mindspore import Tensor, ops, nn

import rama_py
from sklearn.metrics import (
    confusion_matrix, normalized_mutual_info_score, adjusted_rand_score,
    rand_score, adjusted_mutual_info_score, accuracy_score
)


# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def ood_of_ind_ratio_(y_true: List, ind_ood_idx_threshold: int):
    total = len(y_true)
    error = 0
    for item in y_true:
        if item < ind_ood_idx_threshold:
            error += 1
    return error / total


def hungray_alignment(y_true, y_pred):
    D = max(max(y_pred), max(y_true)) + 1
    w = np.zeros((D, D))
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_alignment(np.array(y_true), np.array(y_pred))
    acc = sum([w[i, j] for i, j in ind]) / len(y_pred)
    return acc


def clustering_score(y_true, y_pred, ind_ood_idx_threshold):
    return {
        'ACC': round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
        'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        'RI': round(rand_score(y_true, y_pred) * 100, 2),
        'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2),
        'AMI': round(adjusted_mutual_info_score(y_true, y_pred) * 100, 2),
        'IND_Ratio': round(ood_of_ind_ratio_(y_true, ind_ood_idx_threshold) * 100, 2),
    }


# -------------------------------------------------------------
# Main GenerateOOD class (MindSpore Version)
# -------------------------------------------------------------

class GenerateOOD_MS:

    def __init__(self, args, data, unkdataset, pretrained_model, tokenizer, rama_cluster_times=1):
        self.args = args
        self.data = data
        self.p_node = args.p_node
        self.rama_cluster_times = rama_cluster_times + 1

        # Dataset prepared earlier as python list
        if isinstance(unkdataset, str):
            print("Loading OOD dataset with noise...")
            with open(unkdataset, "rb") as fp:
                self.unkDs = pickle.load(fp)['data']
        else:
            self.unkDs = unkdataset

        # Prefix Tuning model (MindSpore version)
        self.model = pretrained_model
        self.tokenizer = tokenizer

    # -------------------------------------------------------------
    # Step 1: Generate utterances with prefix-tuned BART
    # -------------------------------------------------------------

    def _generate_utterance(self):

        cache_path = (
            f"GenerateOODData/dataset-{self.args.dataset}-known_cls_ratio-"
            f"{self.args.known_cls_ratio}-seed-{self.args.seed}-lr-{self.args.lr}"
        )

        os.makedirs("GenerateOODData", exist_ok=True)

        if os.path.exists(f"{cache_path}/data.pkl"):
            print("[Cache] Loading generated OOD utterance...")
            with open(f"{cache_path}/data.pkl", "rb") as fp:
                pred_description = pickle.load(fp)
                label_list = pickle.load(fp)
            return pred_description, label_list

        print("[Generate] Creating OOD utterance...")
        pred_description, label_list = [], []

        for item in tqdm(self.unkDs, desc="Generate OOD"):

            input_ids = Tensor(item["input_ids"], ms.int32)
            attention_mask = Tensor(item["attention_mask"], ms.int32)
            label_ids = int(item["label_ids"])

            # decode 3 beams
            outputs = self.model.generate(
                input_ids=input_ids.expand_dims(0),
                attention_mask=attention_mask.expand_dims(0),
                num_beams=3,
                max_length=8,
                min_length=2,
            )

            decoded_preds = [
                self.tokenizer.decode(outputs[i].asnumpy().tolist(), skip_special_tokens=True)
                for i in range(len(outputs))
            ]

            # Clean beam outputs
            beams = [t.replace("It", "").replace("was", "").strip() for t in decoded_preds]
            description = " , ".join(beams)

            pred_description.append(description)
            label_list.append(label_ids)

        # Save cache
        os.makedirs(cache_path, exist_ok=True)
        with open(f"{cache_path}/data.pkl", "wb") as fp:
            pickle.dump(pred_description, fp)
            pickle.dump(label_list, fp)

        print("[Cache] Saved OOD utterance.")
        return pred_description, label_list

    # -------------------------------------------------------------
    # Step 2: RAMA-based multi-round clustering
    # -------------------------------------------------------------

    def _cluster_ood_single(self, pred_description):

        rouge = Rouge()
        pred_rouge = []

        for idx, des in tqdm(enumerate(pred_description), desc="Pairwise Rouge"):
            scores = []
            for j, des2 in enumerate(pred_description):
                if j > idx:
                    r = rouge.get_scores([des], [des2])[0]
                    avg_score = (r["rouge-1"]["f"] + r["rouge-2"]["f"] + r["rouge-l"]["f"]) / 3
                    scores.append((avg_score, idx, j))
            pred_rouge.append(scores)

        all_edges = functools.reduce(lambda x, y: x + y, pred_rouge)
        all_edges = sorted(all_edges, key=lambda x: x[0], reverse=True)
        non_zero_edges = [x for x in all_edges if x[0] > 0]

        threshold_index = int(len(non_zero_edges) * self.p_node)
        threshold_score = non_zero_edges[threshold_index][0]

        rama_edges = []
        for idx, (score, n1, n2) in enumerate(non_zero_edges):
            if idx <= threshold_index:
                rama_edges.append((score, n1, n2))
            else:
                rama_edges.append((score - threshold_score, n1, n2))

        row, col, weight = [], [], []
        for s, i, j in rama_edges:
            row.append(i)
            col.append(j)
            weight.append(s)

        opts = rama_py.multicut_solver_options("PD+")
        result = rama_py.rama_cuda(row, col, weight, opts)
        return result[0]

    def _cluster_multi(self, pred_description, label_list):
        # identical logic as original file (multi-stage clustering)
        # fully implemented here (same behavior)
        cluster_succ, cluster_label = [], []
        N_ood = len(self.data.all_label_list) - len(self.data.known_label_list)
        avg_n = len(pred_description) // N_ood

        for step in range(self.rama_cluster_times):
            offset = len(cluster_succ)
            cluster_ids = self._cluster_ood_single(pred_description)

            distinct_ids = sorted(list(set(cluster_ids)))
            map_id = {old: i for i, old in enumerate(distinct_ids)}

            C = [[] for _ in range(len(distinct_ids))]
            L = [[] for _ in range(len(distinct_ids))]
            D = [[] for _ in range(len(distinct_ids))]

            for cl, lb, des in zip(cluster_ids, label_list, pred_description):
                C[map_id[cl]].append(map_id[cl] + offset)
                L[map_id[cl]].append(lb)
                D[map_id[cl]].append(des)

            # Last round: gather all
            if step == self.rama_cluster_times - 1:
                cluster_succ.extend(C)
                cluster_label.extend(L)
                final_clusters = []
                for idx in range(len(cluster_succ)):
                    final_clusters.append([idx] * len(cluster_succ[idx]))
                return functools.reduce(lambda x, y: x + y, final_clusters), \
                       functools.reduce(lambda x, y: x + y, cluster_label)

            # Otherwise: filter small/large clusters
            remaining_desc, remaining_labels = [], []
            lengths = [len(c) for c in C]

            for i, ln in enumerate(lengths):
                if 5 < ln < avg_n * 1.5:
                    cluster_succ.append(C[i])
                    cluster_label.append(L[i])
                else:
                    remaining_desc.append(D[i])
                    remaining_labels.append(L[i])

            if len(remaining_desc) == 0:
                break

            pred_description = functools.reduce(lambda x, y: x + y, remaining_desc)
            label_list = functools.reduce(lambda x, y: x + y, remaining_labels)

        return [], []

    # -------------------------------------------------------------
    # Step 3: Evaluate clustering
    # -------------------------------------------------------------
    def evaluation_ood(self):
        pred_des, label_list = self._generate_utterance()
        y_pred, y_true = self._cluster_multi(pred_des, label_list)
        result = clustering_score(y_true, y_pred, self.data.unseen_token_id)
        self.save_results(result)
        print(result)

    # -------------------------------------------------------------
    # Step 4: Save results to CSV
    # -------------------------------------------------------------
    def save_results(self, result_dict):

        os.makedirs(self.args.save_results_path, exist_ok=True)

        var = [
            self.args.dataset, self.args.known_cls_ratio, self.args.labeled_ratio,
            self.p_node, self.args.seed, self.rama_cluster_times,
            self.args.lr, self.args.train_batch_size
        ]
        names = [
            "dataset", "known_cls_ratio", "labeled_ratio", "node_save_ratio",
            "seed", "rama_cluster_times", "lr", "train_batch_size"
        ]

        merged = dict(result_dict, **{k: v for k, v in zip(names, var)})
        df_new = pd.DataFrame([merged])

        out_path = os.path.join(self.args.save_results_path, f"{self.args.dataset}_results_gen.csv")

        if not os.path.exists(out_path):
            df_new.to_csv(out_path, index=False)
        else:
            df_old = pd.read_csv(out_path)
            df = pd.concat([df_old, df_new], ignore_index=True)
            df.to_csv(out_path, index=False)

        print(f"[Saved] OOD clustering results → {out_path}")
