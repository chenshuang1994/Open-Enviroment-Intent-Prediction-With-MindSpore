# train_ms.py

import os
import sys
sys.path.append("../")

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.dataset import GeneratorDataset

from tqdm import tqdm
import numpy as np
import logging

from BART_full_finetune.pipe_ms import Data, set_seed
from BART_full_finetune.model_ms import Model  # ← 注意：使用你的 mindspore 版本模型
from BART_full_finetune.utils_ms import load_parameters
from BART_full_finetune.metric_ms import RougeMetric, Accuracy


# ========================
#     TrainOneStep Cell
# ========================

class TrainStepCell(nn.Cell):
    """
    MindSpore 的 loss + backward + update 的封装
    """
    def __init__(self, network, optimizer):
        super(TrainStepCell, self).__init__()
        self.network = network
        self.network.set_train()
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters)(*inputs)
        self.optimizer(grads)
        return loss


# ==========================
#         MAIN
# ==========================

if __name__ == '__main__':
    print("MindSpore Training initialized...")

    args = load_parameters()
    set_seed(args.seed)

    # 设置模式
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")

    # 数据文件路径
    datafile = f"datafile/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}"
    model_dir = f"model/dataset-{args.dataset}-known_cls_ratio-{args.known_cls_ratio}-seed-{args.seed}-lr-{args.lr}-lmd-{args.lamda_loss}"

    os.makedirs(datafile, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ==========================
    #        加载数据
    # ==========================
    data = Data(args)
    args.num_labels = data.num_labels

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Train size = {len(data.train_dataset)}, Dev size = {len(data.eval_dataset)}")

    # 训练集、验证集采用 MindSpore Dataset
    train_ds = data.train_dataset.batch(args.train_batch_size, drop_remainder=True)
    dev_ds = data.eval_dataset.batch(args.eval_batch_size, drop_remainder=False)

    # ==========================
    #        模型初始化
    # ==========================

    model = Model(args)                  # 已迁移的 mindspore 版本 BART
    model.set_train()

    # 优化器
    optimizer = nn.AdamWeightDecay(
        params=model.trainable_params(),
        learning_rate=args.lr
    )

    # 封装训练单步
    train_step = TrainStepCell(model, optimizer)

    # metric
    rouge_metric = RougeMetric(data.tokenizer)
    acc_metric = Accuracy()

    # ==========================
    #     Training Loop
    # ==========================

    best_eval_acc = 0
    patience = 0

    for epoch in range(int(args.num_train_epochs)):
        print(f"\n===== Epoch {epoch} =====")
        total_loss = []
        total_gen_loss = []
        total_cls_loss = []

        # ------- 训练 -------
        for batch in tqdm(train_ds, desc="Train"):
            # batch 是 dict，取出字段
            input_ids = Tensor(batch["input_ids"])
            attention_mask = Tensor(batch["attention_mask"])
            label_ids = Tensor(batch["label_ids"])
            labels = Tensor(batch["labels"])

            # 直接计算 loss （Model.construct 返回 loss）
            loss = train_step(input_ids, attention_mask, labels, label_ids)
            total_loss.append(float(loss.asnumpy()))

        print(f"Train Loss = {np.mean(total_loss):.4f}")

        # ------- 验证 -------
        model.set_train(False)
        for batch in tqdm(dev_ds, desc="Dev"):
            label_text_list = batch["label_text"]

            input_ids = Tensor(batch["input_ids"])
            attention_mask = Tensor(batch["attention_mask"])
            label_ids = Tensor(batch["label_ids"])
            labels = Tensor(batch["labels"])

            # evaluate_step 返回 logits + decoder_tokens
            result = model.evaluate_step(input_ids, attention_mask, labels)

            rouge_metric.update(label_text_list, result["decoder_tokens"])
            acc_metric.update(result["logits"], label_ids)

        rouge_score = rouge_metric.get_metric()["rouge_score"]
        acc_score = acc_metric.get_metric()["acc"]

        print(f"[Dev] Rouge = {rouge_score:.4f}, Acc = {acc_score:.4f}")

        # ==========================
        #       Early Stop & Save
        # ==========================

        if acc_score > best_eval_acc:
            best_eval_acc = acc_score
            patience = 0

            # 保存参数
            ms.save_checkpoint(model, os.path.join(model_dir, "checkpoint.ckpt"))
            print("Model saved!")
        else:
            patience += 1
            if patience >= args.wait_patient:
                print("Early stopping triggered!")
                break

        model.set_train(True)
