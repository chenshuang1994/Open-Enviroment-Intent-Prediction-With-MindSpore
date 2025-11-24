import sys
import os
sys.path.append('../')

from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import pickle

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import GeneratorDataset

from sklearn.metrics import confusion_matrix, accuracy_score

from BART_full_finetune.model_ms import Model   # ← 你需将模型同步迁移
from BART_full_finetune.pipe_ms import Data, set_seed   # ← 同步迁移
from BART_full_finetune.metric_ms import BoundaryLoss, F_measure, euclidean_metric
from BART_full_finetune.utils_ms import load_parameters


context.set_context(mode=context.PYNATIVE_MODE)  # 或 GRAPH_MODE

class ModelManager:

    def __init__(self, args, data, pretrained_model=None):

        self.model = pretrained_model

        self.device = args.device
        if self.model is None:
            self.model = Model(args)
            self.restore_model(args)

        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None


    # ----------- 开放集分类逻辑 ------------
    def open_classify(self, features, data):

        logits = euclidean_metric(features, self.centroids)
        probs = ops.softmax(logits, axis=1)
        preds = ops.argmax(probs, axis=1)

        # 欧氏距离
        euc_dis = ops.norm(features - self.centroids[preds], ord=2, axis=1)

        # 开放集判断
        mask = euc_dis >= self.delta[preds]
        preds = preds.asnumpy()
        preds[mask.asnumpy()] = data.unseen_token_id

        return Tensor(preds, ms.int64)

    # ----------------- 评估 -------------------
    def evaluation(self, args, data, mode="eval"):
        self.model.set_train(False)

        total_labels = []
        total_preds = []

        unkDataset = []

        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            self.delta = args.delta * self.delta
            dataloader = data.test_dataloader
        else:
            raise ValueError()

        for batch in tqdm(dataloader, desc="Iteration"):
            batch.pop("label_text")

            input_ids = Tensor(batch["input_ids"], ms.int64)
            input_mask = Tensor(batch["attention_mask"], ms.int64)
            label_ids = Tensor(batch["label_ids"], ms.int64)
            labels = Tensor(batch["labels"], ms.int64)

            outputs = self.model(input_ids=input_ids,
                                 attention_mask=input_mask,
                                 labels=labels,
                                 label_ids=None)

            pooled_output = outputs["feature"]
            preds = self.open_classify(pooled_output, data)

            if mode == "test":
                for i in range(len(preds)):
                    if preds[i].asnumpy() == data.unseen_token_id:
                        unkDataset.append({
                            "input_ids": input_ids[i].asnumpy(),
                            "attention_mask": input_mask[i].asnumpy(),
                            "label_ids": label_ids[i].asnumpy(),
                            "labels": labels[i].asnumpy()
                        })

            total_labels.extend(label_ids.asnumpy().tolist())
            total_preds.extend(preds.asnumpy().tolist())

        y_pred = np.array(total_preds)
        y_true = np.array(total_labels)

        # 记录结果
        if mode == 'test':
            self.predictions = [data.label_list[idx] for idx in y_pred]
            self.true_labels = [data.all_label_list[idx] for idx in y_true]
        else:
            self.predictions = [data.label_list[idx] for idx in y_pred]
            self.true_labels = [data.label_list[idx] for idx in y_true]

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['F1-score']
            print('acc', round(accuracy_score(y_true, y_pred) * 100, 2))
            return eval_score

        else:
            # 保存 UNK 数据
            unk_path = f"unkdataset_with_noise/unkdataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}-delta-{args.delta}.pkl"
            with open(unk_path, 'wb') as fp:
                pickle.dump({'data': unkDataset}, fp)

            y_true2 = [t if t < data.unseen_token_id else data.unseen_token_id for t in y_true]
            cm = confusion_matrix(y_true2, y_pred)
            results = F_measure(cm)
            acc = accuracy_score(y_true2, y_pred) * 100
            results['Accuracy'] = acc

            self.test_results = results
            self.save_results(args)


    # --------------------- 训练主流程 -------------------------
    def train(self, args, data):

        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.feat_dim)
        self.delta = ops.softplus(criterion_boundary.delta)

        optimizer = nn.AdamWeightDecay(criterion_boundary.trainable_params(), learning_rate=args.lr_boundary)

        self.centroids = self.centroids_cal(args, data)

        wait = 0
        best_delta, best_centroids = None, None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.set_train(True)

            tr_loss = 0
            nb_tr_steps = 0

            for batch in tqdm(data.train_dataloader, desc="Iteration"):

                batch.pop("label_text")
                input_ids = Tensor(batch["input_ids"], ms.int64)
                input_mask = Tensor(batch["attention_mask"], ms.int64)
                label_ids = Tensor(batch["label_ids"], ms.int64)
                labels = Tensor(batch["labels"], ms.int64)

                def forward_fn(input_ids, input_mask, labels, label_ids):
                    features = self.model(input_ids=input_ids,
                                          attention_mask=input_mask,
                                          labels=labels,
                                          label_ids=None)["feature"]
                    loss, delta = criterion_boundary(features, self.centroids, label_ids)
                    return loss, delta

                grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
                (loss, delta), grads = grad_fn(input_ids, input_mask, labels, label_ids)
                optimizer(grads)

                self.delta = delta
                tr_loss += float(loss.asnumpy())
                nb_tr_steps += 1

            self.delta_points.append(self.delta)
            print("train_loss", tr_loss / nb_tr_steps)

            eval_score = self.evaluation(args, data, mode="eval")
            print(f"eval_score {eval_score} wait {wait}")
            print("best_score", self.best_eval_score)

            if eval_score > self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.delta = best_delta
        self.centroids = best_centroids


    # ---------------------- 计算类别中心 ----------------------
    def class_count(self, labels):
        labels = np.array(labels)
        return [np.sum(labels == l) for l in np.unique(labels)]

    def centroids_cal(self, args, data):
        centroids = ops.zeros((data.num_labels, args.feat_dim), ms.float32)

        total_labels = []

        for batch in data.train_dataloader:
            batch.pop("label_text")

            input_ids = Tensor(batch["input_ids"], ms.int64)
            input_mask = Tensor(batch["attention_mask"], ms.int64)
            labels = Tensor(batch["labels"], ms.int64)
            label_ids = Tensor(batch["label_ids"], ms.int64)

            features = self.model(input_ids=input_ids,
                                  attention_mask=input_mask,
                                  labels=labels,
                                  label_ids=None)["feature"]

            total_labels.extend(label_ids.asnumpy().tolist())

            for i, lab in enumerate(label_ids.asnumpy()):
                centroids[lab] += features[i]

        cnt = Tensor(self.class_count(total_labels), ms.float32).expand_dims(1)
        centroids = centroids / cnt

        return centroids


    # ---------------------- 加载/保存 ---------------------------
    def restore_model(self, args):
        print("ReLoading Model Parameter")
        model_filepath = f"model/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}/checkpoint.ckpt"
        param_dict = ms.load_checkpoint(model_filepath)
        ms.load_param_into_net(self.model, param_dict)

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed, args.lamda_loss, args.delta]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed', 'lmda', 'delta']
        vars_dict = {k: v for k, v in zip(names, var)}

        results = dict(self.test_results, **vars_dict)

        file_name = f'{args.dataset}_results.csv'
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            df = pd.DataFrame([results])
        else:
            df = pd.read_csv(results_path)
            df = df.append(results, ignore_index=True)

        df.to_csv(results_path, index=False)
        print("test_results", df)



# ----------------------------- 主程序 -----------------------------
if __name__ == "__main__":
    args = load_parameters()
    set_seed(args.seed)

    datafile = f"datafile/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}"

    if not os.path.exists(datafile):
        os.makedirs(datafile)

    data = Data(args)
    print(f"train {len(data.train_dataset)}  dev {len(data.eval_dataset)}  test {len(data.test_dataset)}")

    # Dataset → GeneratorDataset（MindSpore）
    data.build_mindspore_dataloaders(args)

    manager = ModelManager(args, data, None)
    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data, mode="test")
    print('Evaluation finished!')
