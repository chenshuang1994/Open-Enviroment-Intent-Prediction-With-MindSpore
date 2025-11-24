import sys
import os
import functools

sys.path.append('../')

from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import pickle

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from fastNLP import cache_results, DataSet, Instance, TorchDataLoader

from BART_prefix_tuning_with_multi_label_agu_v1_ms.prefix_model_ms import Model
from BART_prefix_tuning_with_multi_label_agu_v1_ms.pipe_ms import Pipe, set_seed
from BART_prefix_tuning_with_multi_label_agu_v1_ms.metric_ms import BoundaryLoss, F_measure, euclidean_metric
from BART_prefix_tuning_with_multi_label_agu_v1_ms.utils_ms import load_parameters, pack_labels_batch


class ModelManager:

    def __init__(self, args, data, pretrained_model=None):

        self.model = pretrained_model

        ms.set_context(device_target="GPU")

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

    def open_classify(self, features, data):

        logits = euclidean_metric(features, self.centroids)
        softmax = ops.Softmax(axis=-1)
        probs = softmax(logits)
        preds = ops.Argmax(axis=1)(probs)

        gather_centroids = ops.Gather()(self.centroids, preds, 0)
        euc_dis = ops.Sqrt()(ops.ReduceSum()((features - gather_centroids) ** 2, 1))

        delta_selected = ops.Gather()(self.delta, preds, 0)
        mask = euc_dis >= delta_selected
        preds = ops.Select()(mask, 
                             Tensor([data.unseen_token_id], ms.int64).repeat(preds.shape[0]), 
                             preds)

        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.set_train(False)

        total_labels = []
        total_preds = []
        unkDataset = DataSet()

        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader
            self.delta = self.delta * args.delta
        else:
            raise ValueError("mode 不合法")

        for batch in tqdm(dataloader, desc="Iteration"):
            batch.pop("label_text")
            labels = batch.pop("labels")
            labels = pack_labels_batch(labels)[0]

            # MindSpore 张量
            labels = Tensor(labels.numpy(), ms.int64)

            # 剩下的 feature
            batch_tensors = []
            for _, t in batch.items():
                batch_tensors.append(Tensor(t.numpy()))

            input_ids, input_mask, multi_labels, label_ids = batch_tensors

            outputs = self.model(input_ids=input_ids, attention_mask=input_mask, labels=labels)
            pooled_output = outputs['feature']

            preds = self.open_classify(pooled_output, data)

            total_labels.append(label_ids.asnumpy())
            total_preds.append(preds.asnumpy())

            if mode == "test":
                for inp, att, lab, pred, lbid in zip(
                        input_ids.asnumpy(),
                        input_mask.asnumpy(),
                        labels.asnumpy(),
                        preds.asnumpy(),
                        label_ids.asnumpy()):
                    if pred == data.unseen_token_id:
                        unkDataset.append(
                            Instance(input_ids=inp, attention_mask=att,
                                     label_ids=lbid, labels=lab)
                        )

        y_true = np.concatenate(total_labels)
        y_pred = np.concatenate(total_preds)

        if mode == 'test':
            unk_path_dir = "unkdataset_with_noise"
            os.makedirs(unk_path_dir, exist_ok=True)

            unk_path = f"unkdataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}-delta-{args.delta}.pkl"
            unk_path = os.path.join(unk_path_dir, unk_path)
            with open(unk_path, 'wb') as fp:
                pickle.dump({'data': unkDataset}, fp)

            y_true_open = [item if item < data.unseen_token_id else data.unseen_token_id for item in y_true]

            from sklearn.metrics import confusion_matrix, accuracy_score
            cm = confusion_matrix(y_true_open, y_pred)
            results = F_measure(cm)
            results['Accuracy'] = round(accuracy_score(y_true_open, y_pred) * 100, 2)

            self.test_results = results
            self.save_results(args)
            return

        # mode = eval
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        eval_score = F_measure(cm)['F1-score']
        print("eval accuracy:", np.mean(y_true == y_pred) * 100)

        return eval_score

    def train(self, args, data):

        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.feat_dim)
        self.delta = ops.Softplus()(criterion_boundary.delta)

        optimizer = nn.Adam(criterion_boundary.trainable_params(), learning_rate=args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        wait = 0
        best_delta, best_centroids = None, None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.set_train(True)
            tr_loss = []

            for batch in tqdm(data.train_dataloader, desc="Iteration"):
                batch.pop("label_text")
                labels = pack_labels_batch(batch.pop("labels"))[0]
                labels = Tensor(labels.numpy(), ms.int64)

                batch_tensors = [Tensor(t.numpy()) for _, t in batch.items()]
                input_ids, input_mask, multi_labels, label_ids = batch_tensors
                label_ids = Tensor(label_ids.asnumpy(), ms.int64)

                outputs = self.model(input_ids=input_ids, attention_mask=input_mask, labels=labels)
                features = outputs['feature']

                loss, self.delta = criterion_boundary(features, self.centroids, label_ids)
                grad_fn = ops.value_and_grad(lambda: loss, None, optimizer.parameters)
                loss_val, grads = grad_fn()
                optimizer(grads)

                tr_loss.append(loss_val.asnumpy())

            avg_loss = np.mean(tr_loss)
            print("train_loss:", avg_loss)

            eval_score = self.evaluation(args, data, mode="eval")
            print('eval_score:', eval_score, "wait:", wait)

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

    def centroids_cal(self, args, data):

        centroids = Tensor(np.zeros((args.adb_num_labels, args.feat_dim), dtype=np.float32))
        total_labels = []

        for batch in data.train_dataloader:
            batch.pop("label_text")
            labels = pack_labels_batch(batch.pop("labels"))[0]
            labels = Tensor(labels.numpy(), ms.int64)

            batch_tensors = [Tensor(t.numpy()) for _, t in batch.items()]
            input_ids, input_mask, multi_labels, label_ids = batch_tensors
            label_ids = Tensor(label_ids.asnumpy(), ms.int64)

            features = self.model(input_ids=input_ids, attention_mask=input_mask, labels=labels)['feature']

            for i in range(len(label_ids)):
                lab = int(label_ids[i].asnumpy())
                centroids[lab] += features[i]

            total_labels.extend(label_ids.asnumpy())

        total_labels = np.array(total_labels)
        cls_count = np.bincount(total_labels)

        for i in range(len(cls_count)):
            if cls_count[i] > 0:
                centroids[i] = centroids[i] / Tensor(cls_count[i], ms.float32)

        return centroids

    def restore_model(self, args):
        print("Reloading model...")
        model_path = f"model/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}-m_loss-{args.m_loss}"
        self.model.load_state_dict(ms.load_checkpoint(os.path.join(model_path, "checkpoint.ckpt")))

    def save_results(self, args):
        os.makedirs(args.save_results_path, exist_ok=True)

        var = [
            args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed,
            args.pre_seq_len, args.lr, args.lamda_loss, args.m_loss
        ]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed', 'psl',
                 'lr', 'lmd', 'm_loss']

        result_dict = {k: v for k, v in zip(names, var)}
        result_dict.update(self.test_results)

        df = pd.DataFrame([result_dict])
        out_path = os.path.join(args.save_results_path, f"{args.dataset}_adb_results.csv")

        if not os.path.exists(out_path):
            df.to_csv(out_path, index=False)
        else:
            old = pd.read_csv(out_path)
            old = pd.concat([old, df], ignore_index=True)
            old.to_csv(out_path, index=False)

        print("test_results saved:", out_path)


if __name__ == '__main__':

    args = load_parameters()
    set_seed(args.seed)

    datafile = f"datafile/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}"
    os.makedirs(datafile, exist_ok=True)

    @cache_results(f"{datafile}/data.pkl", _hash_param=False)
    def load_data(arg):
        return Pipe(arg)

    data = load_data(args)
    print("train:", len(data.train_dataset), " eval:", len(data.eval_dataset), " test:", len(data.test_dataset))

    train_dl = TorchDataLoader(data.train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dl = TorchDataLoader(data.eval_dataset, batch_size=args.eval_batch_size, shuffle=True)
    test_dl = TorchDataLoader(data.test_dataset, batch_size=args.eval_batch_size, shuffle=True)

    data.train_dataloader = train_dl
    data.eval_dataloader = dev_dl
    data.test_dataloader = test_dl

    args.num_labels = len(data.known_multi_label_unique)
    args.adb_num_labels = data.num_labels

    manager = ModelManager(args, data)
    manager.train(args, data)
    manager.evaluation(args, data, mode="test")
