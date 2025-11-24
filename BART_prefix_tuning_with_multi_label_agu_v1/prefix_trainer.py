import sys
import os

sys.path.append("../")

import logging
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, average_precision_score, hamming_loss

import mindspore as ms
from mindspore import nn, ops, Tensor, context

from mindspore.dataset import GeneratorDataset, BatchDataset

from BART_prefix_tuning_with_multi_label_agu_v1.pipe_ms import Pipe, set_seed
from BART_prefix_tuning_with_multi_label_agu_v1.prefix_model_ms import Model_MS
from BART_prefix_tuning_with_multi_label_agu_v1.utils_ms import (
    load_parameters,
    pack_labels_batch_ms,
)
from BART_prefix_tuning_with_multi_label_agu_v1.metric_ms import RougeMetricMS

context.set_context(mode=0, device_target="GPU")  # GRAPH mode


###############################################
# Warmup + Linear Decay LR
###############################################
def get_warmup_linear_lr(total_steps, warmup, base_lr):
    lr_list = []
    warmup_steps = int(total_steps * warmup)

    for i in range(total_steps):
        if i < warmup_steps:
            lr = base_lr * float(i) / float(warmup_steps)
        else:
            lr = base_lr * max(
                0.0,
                float(total_steps - i) / float(total_steps - warmup_steps)
            )
        lr_list.append(lr)
    return Tensor(lr_list, ms.float32)


#########################################################
# Dataset Wrapper for MindSpore
#########################################################
class TrainDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        item = self.data[index]
        return (
            item["input_ids"].asnumpy(),
            item["attention_mask"].asnumpy(),
            item["multi_labels"],
            item["label_ids"],
            item["labels"],
            item["label_text"],
        )

    def __len__(self):
        return len(self.data)


#########################################################
if __name__ == "__main__":
    print("Data and Parameters Initialization...")
    args = load_parameters()
    set_seed(args.seed)

    ###############################################
    # Path define
    ###############################################
    datafile = f"datafile/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}"

    model_filepath = f"model/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-pre_seq_len-{args.pre_seq_len}-lmd-{args.lamda_loss}-m_loss-{args.m_loss}"

    os.makedirs(datafile, exist_ok=True)
    os.makedirs(model_filepath, exist_ok=True)

    ###############################################
    # Load data
    ###############################################
    pipe = Pipe(args)
    train_data = TrainDataset(pipe.train_dataset)
    dev_data = TrainDataset(pipe.eval_dataset)

    train_ds = GeneratorDataset(train_data, column_names=["ids", "mask", "multi", "label_id", "label_list", "label_text"], shuffle=True)
    train_ds = train_ds.batch(batch_size=args.train_batch_size, drop_remainder=True)

    dev_ds = GeneratorDataset(dev_data, column_names=["ids", "mask", "multi", "label_id", "label_list", "label_text"], shuffle=False)
    dev_ds = dev_ds.batch(batch_size=args.eval_batch_size)

    ###############################################
    # Model and Optimizer
    ###############################################
    print("Initializing Model...")
    args.num_labels = len(pipe.known_multi_label_unique)

    model = Model_MS(args)

    # prepare LR schedule
    total_steps = len(train_ds) * int(args.num_train_epochs)
    lr = get_warmup_linear_lr(total_steps, warmup=0.1, base_lr=args.lr)

    optim = nn.AdamWeightDecay(model.trainable_params(), learning_rate=lr)

    ###############################################
    # Loss + Grad
    ###############################################
    grad_fn = ops.GradOperation(get_by_list=True)

    def train_step(input_ids, mask, multi_labels, label_ids, labels):
        def forward_func(input_ids, mask, multi_labels, label_ids, labels):
            out = model(
                input_ids=Tensor(input_ids, ms.int32),
                attention_mask=Tensor(mask, ms.int32),
                labels=labels,
                multi_labels=Tensor(multi_labels, ms.float32),
                label_ids=Tensor(label_ids, ms.int32),
            )
            return out["loss"]

        grads = grad_fn(forward_func, model.trainable_params())(
            input_ids, mask, multi_labels, label_ids, labels
        )
        loss = forward_func(input_ids, mask, multi_labels, label_ids, labels)
        optim(grads)
        return loss

    ###############################################
    # Evaluator
    ###############################################
    rougeMetric = RougeMetricMS(pipe.tokenizer)

    ###############################################
    # Training Loop
    ###############################################
    print("Training begin!")

    best_eval = 0
    patience = 0

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.set_train(True)
        epoch_loss = []

        for batch in tqdm(train_ds, desc="TrainIter"):
            ids, mask, multi, label_id, label_list, _ = batch

            labels_packed = pack_labels_batch_ms(label_list)
            labels_packed = labels_packed[0]  # only first sequence

            loss = train_step(ids, mask, multi, label_id, labels_packed)
            epoch_loss.append(loss.asnumpy())

        print(f"Epoch {epoch} training loss = {np.mean(epoch_loss):.4f}")

        ###############################################################
        # Dev Eval
        ###############################################################
        model.set_train(False)

        all_preds = []
        all_label = []

        for batch in tqdm(dev_ds, desc="DevIter"):
            ids, mask, multi, label_id, label_list, label_text = batch

            labels_packed = pack_labels_batch_ms(label_list)[0]

            result = model.evaluate_step(
                Tensor(ids, ms.int32),
                Tensor(mask, ms.int32),
                labels_packed
            )

            rougeMetric.update(label_text, result["decoder_tokens"].asnumpy())

            logits = ops.Sigmoid()(result["logits"])
            all_preds.append(logits.asnumpy())
            all_label.append(multi)

        all_preds = np.concatenate(all_preds, 0)
        all_label = np.concatenate(all_label, 0)

        # sample-wise F1
        pred_bin = np.rint(all_preds)
        eval_score = f1_score(all_label, pred_bin, average="samples")

        rouge_result = rougeMetric.get_metric()

        print(f"Dev F1 = {eval_score:.4f}, Rouge = {rouge_result['rouge_score']:.4f}")

        ###############################################################
        # Save Best
        ###############################################################
        if eval_score > best_eval:
            patience = 0
            best_eval = eval_score

            ms.save_checkpoint(model, os.path.join(model_filepath, "best.ckpt"))
        else:
            patience += 1
            if patience >= args.wait_patient:
                print("Early stopping.")
                break

    print("Training finished!")
