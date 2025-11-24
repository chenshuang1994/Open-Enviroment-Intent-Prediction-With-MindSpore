import sys
import pickle
import os
import functools
from typing import List

sys.path.append("../")

import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from scipy.optimize import linear_sum_assignment

import mindspore as ms
from mindspore import Tensor, ops, nn, context
from mindspore.dataset import GeneratorDataset

from sklearn.metrics import (
    confusion_matrix, normalized_mutual_info_score, adjusted_rand_score,
    accuracy_score, rand_score, adjusted_mutual_info_score
)

from BART_full_finetune.pipe_ms import Data   # 你需要迁移到 mindspore 版本
from BART_full_finetune.model_ms import Model  # 需要迁移为 mindspore 版本


context.set_context(mode=context.PYNATIVE_MODE)


# ----------------  工具函数 ------------------

def ood_of_ind_ratio_(y_true: List, ind_ood_idx_threshold: int):
    y_true = np.array(y_true)
    return float(np.sum(y_true < ind_ood_idx_threshold) / len(y_true))


def hungray_aligment(y_true, y_pred):
    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D))
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) / len(y_pred)


def clustering_score(y_true, y_pred, ind_ood_idx_threshold):
    return {
        "ACC": round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
        "ARI": round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        "RI": round(rand_score(y_true, y_pred) * 100, 2),
        "NMI": round(normalized_mutual_info_score(y_true, y_pred) * 100, 2),
        "AMI": round(adjusted_mutual_info_score(y_true, y_pred) * 100, 2),
        "IND_Ratio": round(ood_of_ind_ratio_(y_true, ind_ood_idx_threshold) * 100, 2)
    }


# ---------------- 主类 --------------------

class GenerateOOD:

    def __init__(self, args, data: Data, unkdataset_or_filename,
                 pretrain_model, tokenizer, rama_cluster_times=1):

        self.args = args
        self.data = data
        self.tokenizer = tokenizer

        self.p_node = args.p_node
        self.rama_cluster_times = rama_cluster_times + 1

        # ---------- 加载 UNK dataset ----------
        if isinstance(unkdataset_or_filename, str):
            print("Loading dataset with noise!")
            with open(unkdataset_or_filename, "rb") as fp:
                self.unkDs = pickle.load(fp)["data"]
        else:
            self.unkDs = unkdataset_or_filename

        # ---------- MindSpore 版本的模型 ----------
        if isinstance(pretrain_model, str):
            raise ValueError("需要将CausalLM模型迁移为mindspore版本")
        else:
            print("Loading Model parameter (MindSpore)")
            self.model = pretrain_model
            ckpt_path = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-lmd-{}/checkpoint.ckpt".format(
                args.dataset, args.known_cls_ratio, args.seed, args.lr, args.lamda_loss
            )
            param_dict = ms.load_checkpoint(ckpt_path)
            ms.load_param_into_net(self.model, param_dict)

    # =================== 入口函数 =====================
    def evaluation_ood(self):
        pred_description, label_list = self._generate_utterance()
        y_pred, y_true = self._cluster_ood_again(pred_description, label_list)

        result = clustering_score(y_true, y_pred, self.data.unseen_token_id)
        print(result)
        self.save_results(result)

    # ------------------ 聚类主流程 -------------------

    def _cluster_ood_again(self, pred_description, label_list):
        cluster_succ, cluster_label = [], []

        num_label = len(self.data.all_label_list) - len(self.data.known_label_list)
        avg_sample_numb = len(pred_description) // num_label

        for i in range(self.rama_cluster_times):
            cluster_idx_offset = len(cluster_succ)
            cluster_list = self._cluster_ood(pred_description)

            # 所有簇的分配结构
            cluster_cnt = len(np.unique(cluster_list))
            cluster_all = [[] for _ in range(cluster_cnt)]
            cluster_label_all = [[] for _ in range(cluster_cnt)]
            cluster_desp = [[] for _ in range(cluster_cnt)]

            node_map = {c: idx for idx, c in enumerate(np.unique(cluster_list))}

            for c_node, label_node, desp in zip(cluster_list, label_list, pred_description):
                idx2 = node_map[c_node]
                cluster_all[idx2].append(c_node + cluster_idx_offset)
                cluster_label_all[idx2].append(label_node)
                cluster_desp[idx2].append(desp)

            if i == self.rama_cluster_times - 1:
                cluster_succ.extend(cluster_all)
                cluster_label.extend(cluster_label_all)

                # 连续化
                re_cluster_succ = []
                for reIdx in range(len(cluster_succ)):
                    re_cluster_succ.append([reIdx] * len(cluster_succ[reIdx]))

                cluster_succ = re_cluster_succ
                break

            # 处理小簇/大簇
            other_desp, other_label = [], []
            for idx, lst in enumerate(cluster_all):
                if 5 < len(lst) < avg_sample_numb * 1.5:
                    cluster_succ.append(cluster_all[idx])
                    cluster_label.append(cluster_label_all[idx])
                else:
                    other_desp.append(cluster_desp[idx])
                    other_label.append(cluster_label_all[idx])

            if len(other_desp) == 0:
                break

            pred_description = functools.reduce(lambda a, b: a + b, other_desp)
            label_list = functools.reduce(lambda a, b: a + b, other_label)

        return (
            functools.reduce(lambda a, b: a + b, cluster_succ),
            functools.reduce(lambda a, b: a + b, cluster_label)
        )

    # ------------------ RAMA 聚类 -------------------
    def _cluster_ood(self, pred_description):
        rouge = Rouge()
        pred_rouge = []

        for idx, desp in tqdm(enumerate(pred_description)):
            des_list = [desp] * len(pred_description)
            scores = rouge.get_scores(des_list, pred_description)

            group = []
            for idy, sc in enumerate(scores):
                if idy > idx:
                    fscore = (sc["rouge-1"]["f"] + sc["rouge-2"]["f"] + sc["rouge-l"]["f"]) / 3
                    group.append((fscore, idx, idy))

            group = sorted(group, key=lambda x: x[0], reverse=True)
            pred_rouge.append(group)

        all_edge = functools.reduce(lambda x, y: x + y, pred_rouge)
        all_edge_non_zero = [e for e in all_edge if e[0] > 0]
        all_edge_non_zero = sorted(all_edge_non_zero, key=lambda x: x[0], reverse=True)

        threshold_idx = int(len(all_edge_non_zero) * self.p_node)
        threshold_score = all_edge_non_zero[threshold_idx][0]

        rama_edge = []
        for i, (score, x, y) in enumerate(all_edge_non_zero):
            if i <= threshold_idx:
                rama_edge.append((score, x, y))
            else:
                rama_edge.append((score - threshold_score, x, y))

        rows = [e[1] for e in rama_edge]
        cols = [e[2] for e in rama_edge]
        weights = [e[0] for e in rama_edge]

        # 保留你的 rama_py CUDA 调用
        opts = rama_py.multicut_solver_options("PD+")
        res = rama_py.rama_cuda(rows, cols, weights, opts)
        return res[0]

    # ------------------- 生成自然语言描述 -------------------
    def _generate_utterance(self):
        file_name = "GenerateOODData"
        pickle_name = f"dataset-{self.args.dataset}-known_cls_ratio-{self.args.known_cls_ratio}-seed-{self.args.seed}-lr-{self.args.lr}"
        file_path = os.path.join(file_name, pickle_name)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        cache_path = os.path.join(file_path, "data.pkl")

        if os.path.exists(cache_path):
            print("Loading data from cache...")
