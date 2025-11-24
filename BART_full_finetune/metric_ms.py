# metric_ms.py

import numpy as np
from rouge import Rouge

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Normal


# ============================================================
#     1. RougeMetric (不依赖 fastNLP, 直接计算)
# ============================================================

class RougeMetric:
    def __init__(self, tokenizer):
        self.score = []
        self.tokenizer = tokenizer
        self.rouge = Rouge()

    def update(self, label_text, decoder_tokens):
        decoded_preds = self.tokenizer.batch_decode(
            decoder_tokens.asnumpy().tolist(),
            skip_special_tokens=True
        )

        descript = []
        beams = []
        for b in decoded_preds:
            beams.append(b.strip())
            if len(beams) == 3:
                descript.append("; ".join(beams))
                beams = []

        score = self.rouge.get_scores(hyps=descript, refs=label_text, avg=True)
        final_score = (
                score['rouge-1']['f'] * 0.2 +
                score['rouge-2']['f'] * 0.5 +
                score['rouge-l']['f'] * 0.3
        )
        self.score.append(final_score)

    def get_metric(self):
        if len(self.score) == 0:
            return {"rouge_score": 0.0}
        val = sum(self.score) / len(self.score)
        self.score = []
        return {"rouge_score": val}


# ============================================================
#     2. Accuracy (MindSpore 版本)
# ============================================================

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, label_ids):
        """
        logits: [B, C] Tensor
        label_ids: [B] Tensor
        """
        probs = ops.softmax(logits, axis=-1)
        pred = ops.argmax(probs, axis=-1)

        pred_np = pred.asnumpy()
        label_np = label_ids.asnumpy()

        self.correct += (pred_np == label_np).sum()
        self.total += len(label_np)

    def get_metric(self):
        if self.total == 0:
            return {"acc": 0.0}
        return {"acc": float(self.correct) / self.total}


# ============================================================
#     3. Euclidean Metric (MindSpore 版本)
# ============================================================

def euclidean_metric(a, b):
    """
    a: [N, D]
    b: [M, D]
    return: [N, M]
    """
    N = a.shape[0]
    M = b.shape[0]

    a_exp = ops.expand_dims(a, 1)   # [N, 1, D]
    a_exp = ops.broadcast_to(a_exp, (N, M, a.shape[1]))

    b_exp = ops.expand_dims(b, 0)   # [1, M, D]
    b_exp = ops.broadcast_to(b_exp, (N, M, b.shape[1]))

    logits = -ops.sum((a_exp - b_exp) ** 2, axis=2)  # [N, M]
    return logits


# ============================================================
#     4. BoundaryLoss (核心 open-set loss)
# ============================================================

class BoundaryLoss(nn.Cell):
    def __init__(self, num_labels=10, feat_dim=2):
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim

        # trainable delta
        self.delta = ms.Parameter(
            Tensor(np.random.randn(num_labels).astype(np.float32)),
            name="delta"
        )

    def construct(self, pooled_output, centroids, labels):
        """
        pooled_output: [B, D]
        centroids: [C, D]
        labels: [B]
        """
        logits = euclidean_metric(pooled_output, centroids)
        delta = ops.softplus(self.delta)

        # 获取每个样本对应的 centroid & delta
        c = ops.gather(centroids, labels, axis=0)   # [B, D]
        d = ops.gather(delta, labels, axis=0)       # [B]

        x = pooled_output

        # Euclidean distance
        euc_dis = ops.norm(x - c, ord=2, axis=1)  # [B]

        # pos: euc_dis > d
        pos_mask = (euc_dis > d).astype(ms.float32)
        neg_mask = (euc_dis < d).astype(ms.float32)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask

        loss = pos_loss.mean() + neg_loss.mean()

        return loss, delta


# ============================================================
#     5. F_measure
# ============================================================

def F_measure(cm):
    """
    cm: ndarray confusion matrix (C x C)
    """
    n_class = cm.shape[0]
    recalls, precisions, fs = [], [], []

    for i in range(n_class):
        TP = cm[i][i]
        r = TP / cm[i].sum() if cm[i].sum() != 0 else 0
        p = TP / cm[:, i].sum() if cm[:, i].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0

        recalls.append(r * 100)
        precisions.append(p * 100)
        fs.append(f * 100)

    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)

    return {
        "Known": f_seen,
        "Open": f_unseen,
        "F1-score": f
    }
