import sys
import os

sys.path.append("../")

import mindspore as ms
from mindspore import nn, ops, Tensor
import logging
from tqdm import tqdm, trange
from mindspore.train.serialization import save_checkpoint
from fastNLP import cache_results, RandomSampler

from BART_prefix_tuning.pipe_ms import Data, set_seed
from BART_prefix_tuning.prefixtuning_ms import Model      # 你之前迁移后的 MindSpore 模型
from BART_prefix_tuning.utils import load_parameters
from BART_prefix_tuning.metric_ms import RougeMetric, Accuracy


if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    args = load_parameters()
    set_seed(args.seed)

    datafile = "datafile/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.pre_seq_len, args.lamda_loss)

    model_filepath = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.pre_seq_len, args.lamda_loss)

    if not os.path.exists(datafile):
        os.makedirs(datafile)


    @cache_results("{}/data.pkl".format(datafile), _hash_param=False)
    def load_data(arg):
        pipe = Data(arg)
        return pipe


    # --------------------- Logger ---------------------
    log_dir = "log/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.pre_seq_len)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data = load_data(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("train dataset length: {}\t dev dataset length: {}\t test dataset length: {}".format(
        len(data.train_dataset), len(data.eval_dataset), len(data.test_dataset)))

    # --------------------- Data Loader ---------------------
    # fastNLP DataSet 本身可直接迭代，不需要 DataLoader
    train_sampler = RandomSampler(data.train_dataset, shuffle=True, seed=args.seed)
    dev_sampler = RandomSampler(data.eval_dataset, shuffle=False)

    # --------------------- Device ---------------------
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")

    # --------------------- Model ---------------------
    args.num_labels = data.num_labels
    model = Model(args)

    # --------------------- Optimizer ---------------------
    optimizer = nn.AdamWeightDecay(
        params=model.trainable_params(),
        learning_rate=args.lr,
        eps=1e-8
    )

    # --------------------- Scheduler ---------------------
    total_steps = len(data.train_dataset) // args.train_batch_size * args.num_train_epochs

    scheduler = nn.PolynomialDecayLR(
        learning_rate=args.lr,
        end_learning_rate=0.0,
        decay_steps=total_steps,
        power=1.0
    )

    # 手动更新 LR
    def update_lr(step):
        lr = scheduler(step)
        optimizer.learning_rate = lr

    # --------------------- Metrics ---------------------
    rougeMetric = RougeMetric(tokenizer=data.tokenizer)
    accMetric = Accuracy()

    # --------------------- Automatic Gradient ---------------------
    grad_fn = ops.GradOperation(get_by_list=True)

    def train_step(inputs):
        (input_ids, input_mask, label_ids, labels) = inputs

        def forward_fn(input_ids, input_mask, label_ids, labels):
            outputs = model(
                input_ids=Tensor(input_ids, ms.int32),
                attention_mask=Tensor(input_mask, ms.int32),
                labels=Tensor(labels, ms.int32),
                label_ids=Tensor(label_ids, ms.int32)
            )
            loss = outputs['loss']
            return loss

        loss = forward_fn(input_ids, input_mask, label_ids, labels)
        grads = grad_fn(forward_fn, optimizer.parameters)(input_ids, input_mask, label_ids, labels)
        optimizer(grads)

        return loss.asnumpy()

    # --------------------- Training ---------------------
    best_eval_score = 0.0
    wait = 0
    global_step = 0

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.set_train()

        epoch_losses = []
        gen_losses = []
        cls_losses = []

        for batch_idx, batch in enumerate(tqdm(train_sampler, desc="Train_Iteration")):
            batch.pop("label_text")
            input_ids, attention_mask, label_ids, labels = batch["input_ids"], batch["attention_mask"], batch["label_ids"], batch["labels"]

            # MindSpore 不需要 to(device)，直接用 Tensor
            loss = train_step((input_ids, attention_mask, label_ids, labels))

            # 记录子 loss
            outputs = model(
                input_ids=Tensor(input_ids, ms.int32),
                attention_mask=Tensor(attention_mask, ms.int32),
                labels=Tensor(labels, ms.int32),
                label_ids=Tensor(label_ids, ms.int32)
            )

            gen_losses.append(float(outputs['gen_loss'].asnumpy()))
            cls_losses.append(float(outputs['cls_loss'].asnumpy()))

            epoch_losses.append(loss)

            update_lr(global_step)
            global_step += 1

        # ---------------- Logging ----------------
        logger.info(f"[Epoch {epoch}] Train Loss = {sum(epoch_losses)/len(epoch_losses)}")
        logger.info(f"gen_loss = {sum(gen_losses)/len(gen_losses)}")
        logger.info(f"cls_loss = {sum(cls_losses)/len(cls_losses)}")

        # ---------------- Dev Evaluation ----------------
        model.set_train(False)

        for batch_idx, batch in enumerate(tqdm(dev_sampler, desc="Dev_Iteration")):
            label_text = batch.pop("label_text")

            input_ids, attention_mask, label_ids, labels = batch["input_ids"], batch["attention_mask"], batch["label_ids"], batch["labels"]

            outputs = model.evaluate_step(
                input_ids=Tensor(input_ids, ms.int32),
                attention_mask=Tensor(attention_mask, ms.int32),
                labels=Tensor(labels, ms.int32)
            )

            rougeMetric.update(label_text, outputs['decoder_tokens'])
            accMetric.update(outputs['logits'], Tensor(label_ids, ms.int32))

        rouge_result = rougeMetric.get_metric()
        acc_result = accMetric.get_metric()

        eval_score = acc_result["acc"]

        logger.info(f"Eval Rouge: {rouge_result}")
        logger.info(f"Eval Acc:   {acc_result}")

        if eval_score > best_eval_score:
            best_eval_score = eval_score
            wait = 0

            if not os.path.exists(model_filepath):
                os.makedirs(model_filepath)

            save_checkpoint(model, os.path.join(model_filepath, "checkpoint.ckpt"))
            logger.info("Save best model ...")
        else:
            wait += 1
            if wait >= args.wait_patient:
                logger.info("Early stopping")
                break

    logger.info("Training finished!")
