# adb_model_ms.py (MindSpore version)

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import mindspore as ms
from mindspore import nn, ops, Tensor

from BART_prefix_tuning.prefixtuning_ms import Model
from BART_prefix_tuning.pipe_ms import Data, set_seed
from BART_prefix_tuning.metric_ms import BoundaryLoss, F_measure, euclidean_metric
from BART_prefix_tuning.utils_ms import load_parameters


class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        self.args = args
        self.data = data
        self.model = pretrained_model if pretrained_model is not None else Model(args)

        # MindSpore 默认使用 GPU
        ms.set_context(device_target="GPU")

        # restore ckpt
        if pretrained_model is None:
            self.restore_model(args)

        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    # -----------------------------------------------------
    #  open-set 分类逻辑
    # -----------------------------------------------------
    def open_classify(self, features, data):

        logits = euclidean_metric(features, self.centroids)
        probs = ops.softmax(logits, axis=1)
        preds = ops.argmax(probs, axis=1)

        # 欧式距离
        centroids_pred = ops.gather(self.centroids, preds, axis=0)
        euc_dis = ops.norm(features - centroids_pred, ord=2, axis=1)

        # delta threshold
        delta_vals = ops.gather(self.delta, preds, axis=0)
        mask = (euc_dis >= delta_vals).astype(ms.int32)

        preds_np = preds.asnumpy()
        mask_np = mask.asnumpy()

        # 将 exceeded delta 的标记为 unseen_token_id
        preds_final = preds_np.copy()
        preds_final[mask_np == 1] = data.unseen_token_id

        return preds_final

    # -----------------------------------------------------
    #  eval / test
    # -----------------------------------------------------
    def evaluation(self, args, data, mode="eval"):

        total_labels = []
        total_preds = []

        unk_list = []

        if mode == "eval":
            dataloader = data.eval_dataset.batch(args.eval_batch_size)
        else:
            dataloader = data.test_dataset.batch(args.eval_batch_size)
            self.delta = args.delta * self.delta

        self.model.set_train(False)

        for batch in tqdm(dataloader, desc=f"{mode}"):
            batch.pop("label_text")
            input_ids = Tensor(batch["input_ids"])
            attention_mask = Tensor(batch["attention_mask"])
            label_ids = Tensor(batch["label_ids"])
            labels = Tensor(batch["labels"])

            result = self.model(input_ids, attention_mask, labels=labels, label_ids=None)
            features = result["feature"]

            preds = self.open_classify(features, data)

            total_preds.extend(preds.tolist())
            total_labels.extend(label_ids.asnumpy().tolist())

            # 收集未知类样本（仅 test）
            if mode == "test":
                for i in range(len(preds)):
                    if preds[i] == data.unseen_token_id:
                        unk_list.append({
                            "input_ids": batch["input_ids"][i],
                            "attention_mask": batch["attention_mask"][i],
                            "labels": batch["labels"][i],
                            "label_ids": batch["label_ids"][i]
                        })

        y_pred = np.array(total_preds)
        y_true = np.array(total_labels)

        # eval 模式计算 F1
        if mode == "eval":
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)["F1-score"]
            print("Acc=", round(accuracy_score(y_true, y_pred) * 100, 2))
            return eval_score

        # test 模式计算 open-set F1
        if mode == "test":
            # 保存未知类数据
            out_dir = "unkdataset_with_noise"
            os.makedirs(out_dir, exist_ok=True)

            file_name = f"unkdataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}.pkl"
            file_path = os.path.join(out_dir, file_name)
            with open(file_path, "wb") as fp:
                pickle.dump({"data": unk_list}, fp)

            # unseen collapse into a single class
            y_true_ = np.array([
                t if t < data.unseen_token_id else data.unseen_token_id
                for t in y_true
            ])

            cm = confusion_matrix(y_true_, y_pred)
            results = F_measure(cm)
            results["Accuracy"] = round(accuracy_score(y_true_, y_pred) * 100, 2)

            self.test_results = results
            self.save_results(args)

    # -----------------------------------------------------
    #  训练 Boundary Loss（第二阶段）
    # -----------------------------------------------------
    def train(self, args, data):

        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.feat_dim)
        self.delta = ops.softplus(criterion_boundary.delta)

        optimizer = nn.AdamWeightDecay(params=criterion_boundary.trainable_params(),
                                       learning_rate=args.lr_boundary)

        self.centroids = self.centroids_cal(args, data)

        best_delta = None
        best_centroids = None
        patience = 0

        for epoch in range(int(args.num_train_epochs)):
            print(f"\n=== Epoch {epoch} ===")
            total_loss = []

            for batch in tqdm(data.train_dataset.batch(args.train_batch_size), desc="Train"):
                batch.pop("label_text")
                input_ids = Tensor(batch["input_ids"])
                attention_mask = Tensor(batch["attention_mask"])
                label_ids = Tensor(batch["label_ids"])
                labels = Tensor(batch["labels"])

                result = self.model(input_ids, attention_mask, labels=labels, label_ids=None)
                features = result["feature"]

                loss, delta = criterion_boundary(features, self.centroids, label_ids)
                # backward
                grads = ops.GradOperation(get_by_list=True)(criterion_boundary,
                                                            optimizer.parameters)(features, self.centroids, label_ids)
                optimizer(grads)

                total_loss.append(float(loss.asnumpy()))

                self.delta = delta

            print("Train loss:", np.mean(total_loss))

            eval_score = self.evaluation(args, data, mode="eval")

            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                patience = 0
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                patience += 1
                if patience >= args.wait_patient:
                    break

        self.delta = best_delta
        self.centroids = best_centroids

    # -----------------------------------------------------
    #  计算类中心
    # -----------------------------------------------------
    def centroids_cal(self, args, data):
        centroids = []
        all_labels = []

        for batch in data.train_dataset.batch(args.train_batch_size):
            batch.pop("label_text")

            input_ids = Tensor(batch["input_ids"])
            attention_mask = Tensor(batch["attention_mask"])
            labels = Tensor(batch["label_ids"])
            real_labels = labels.asnumpy()

            result = self.model(input_ids, attention_mask, labels=batch["labels"], label_ids=None)
            feats = result["feature"].asnumpy()

            for i in range(len(real_labels)):
                all_labels.append(real_labels[i])

        all_labels = np.array(all_labels)
        num_classes = data.num_labels

        centroids_np = np.zeros((num_classes, args.feat_dim), dtype=np.float32)

        # accumulate
        counts = np.zeros(num_classes)
        for batch in data.train_dataset.batch(args.train_batch_size):
            input_ids = Tensor(batch["input_ids"])
            attention_mask = Tensor(batch["attention_mask"])
            labels = Tensor(batch["label_ids"])

            result = self.model(input_ids, attention_mask, labels=batch["labels"], label_ids=None)
            feats = result["feature"].asnumpy()

            for i, lab in enumerate(labels.asnumpy()):
                centroids_np[lab] += feats[i]
                counts[lab] += 1

        centroids_np /= counts[:, None]
        return Tensor(centroids_np, ms.float32)

    # -----------------------------------------------------
    def restore_model(self, args):
        print("Reloading prefix-tuning model...")
        model_path = f"model/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}/checkpoint.ckpt"

        if os.path.exists(model_path):
            ms.load_checkpoint(model_path, self.model)
        else:
            print("Warning: no checkpoint found.")

    # -----------------------------------------------------
    def save_results(self, args):
        os.makedirs(args.save_results_path, exist_ok=True)

        var = [
            args.dataset,
            args.known_cls_ratio,
            args.labeled_ratio,
            args.seed,
            args.pre_seq_len,
            args.lr,
            args.delta,
            args.lamda_loss
        ]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed', 'psl', 'lr', 'delta', 'lmd']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)

        df_name = f"{args.dataset}_adb_results.csv"
        df_path = os.path.join(args.save_results_path, df_name)

        if not os.path.exists(df_path):
            pd.DataFrame([results]).to_csv(df_path, index=False)
        else:
            df_old = pd.read_csv(df_path)
            df_new = pd.DataFrame([results])
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all.to_csv(df_path, index=False)

        print("Test results saved:", df_path)


# -----------------------------------------------------
#  main
# -----------------------------------------------------

if __name__ == "__main__":
    args = load_parameters()
    set_seed(args.seed)

    datafile = (
        f"datafile/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-"
        f"{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}"
    )
    os.makedirs(datafile, exist_ok=True)

    # load data
    data = Data(args)
    print(f"Train={len(data.train_dataset)}, Dev={len(data.eval_dataset)}, Test={len(data.test_dataset)}")
    args.num_labels = data.num_labels

    manager = ModelManager(args, data)
    manager.train(args, data)
    manager.evaluation(args, data, mode="test")
