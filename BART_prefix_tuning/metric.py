import numpy as np
from rouge import Rouge
import mindspore as ms
from mindspore import ops, nn, Tensor


# ============================================================
# 1. RougeMetric （MindSpore 版）
# ============================================================
class RougeMetric:
    """
    Replace fastNLP Metric
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = Rouge()
        self.scores = []

    def update(self, label_text, decoder_tokens):
        """
        decoder_tokens: [batch, seq_len]
        label_text: list[str]
        """
        if isinstance(decoder_tokens, Tensor):
            decoder_tokens = decoder_tokens.asnumpy()

        decoded_preds = self.tokenizer.batch_decode(decoder_tokens, skip_special_tokens=True)

        descript = []
        beams = []
        for pred in decoded_preds:
            beams.append(pred.strip())
            if len(beams) == 3:
                descript.append("; ".join(beams))
                beams = []

        rouge_score = self.rouge.get_scores(hyps=descript, refs=label_text, avg=True)
        final_score = (
            rouge_score["rouge-1"]["f"] * 0.2
            + rouge_score["rouge-2"]["f"] * 0.5
            + rouge_score["rouge-l"]["f"] * 0.3
        )
        self.scores.append(final_score)

    def get_metric(self):
        if len(self.scores) == 0:
            return {"rouge_score": 0.0}

        score = sum(self.scores) / len(self.scores)
        self.scores = []
        return {"rouge_score": score}


# ============================================================
# 2. Accuracy （MindSpore 版）
# ============================================================
class Accuracy:

    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, label_ids):
        """
        logits: Tensor [batch, num_classes]
        label_ids: Tensor [batch]
        """
        if isinstance(logits, Tensor):
            probs = ops.Softmax(axis=-1)(logits)
            preds = ops.Argmax(axis=-1)(probs)
        else:
            preds = logits

        if isinstance(preds, Tensor):
            preds = preds.asnumpy()
        if isinstance(label_ids, Tensor):
            label_ids = label_ids.asnumpy()

        self.correct += int((preds == label_ids).sum())
        self.total += len(label_ids)

    def get_metric(self):
        if self.total == 0:
            return {"acc": 0.0}
        return {"acc": round(float(self.correct) / float(self.total), 6)}


# ============================================================
# 3. euclidean_metric（MindSpore 版）
# ============================================================
def euclidean_metric(a: Tensor, b: Tensor):
    """
    a: [n, d]
    b: [m, d]
    return: [n, m]
    """
    n = a.shape[0]
    m = b.shape[0]

    a_exp = ops.BroadcastTo((n, m, a.shape[-1]))(ops.expand_dims(a, 1))
    b_exp = ops.BroadcastTo((n, m, b.shape[-1]))(ops.expand_dims(b, 0))

    logits = -ops.ReduceSum()((a_exp - b_exp) ** 2, -1)
    return logits


# ============================================================
# 4. BoundaryLoss （MindSpore 版）
# ============================================================
class BoundaryLoss(nn.Cell):

    def __init__(self, num_labels=10, feat_dim=2):
        super().__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim

        # delta initialization
        normal = ops.StandardNormal()
        delta_init = normal((num_labels,), ms.float32)
        self.delta = ms.Parameter(delta_init)

        self.softplus = ops.Softplus()
        self.norm = ops.norm

    def construct(self, pooled_output, centroids, labels):
        """
        pooled_output: [batch, feat_dim]
        centroids: [num_labels, feat_dim]
        labels: [batch]
        """
        logits = euclidean_metric(pooled_output, centroids)

        delta_soft = self.softplus(self.delta)

        c = ops.gather(centroids, labels, 0)
        d = ops.gather(delta_soft, labels, 0)

        x = pooled_output
        euc_dis = self.norm(x - c, ord=2, axis=1)

        pos_mask = (euc_dis > d).astype(ms.float32)
        neg_mask = (euc_dis < d).astype(ms.float32)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask

        loss = ops.ReduceMean()(pos_loss) + ops.ReduceMean()(neg_loss)

        return loss, delta_soft


# ============================================================
# 5. F_measure（MindSpore 版）
# ============================================================
def F_measure(cm: np.ndarray):
    """
    cm: numpy confusion matrix (n_class, n_class)
    """
    n_class = cm.shape[0]
    fs = []

    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() > 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() > 0 else 0
        f = 2 * r * p / (r + p) if (r + p) > 0 else 0
        fs.append(f * 100)

    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)

    return {
        "Known": f_seen,
        "Open": f_unseen,
        "F1-score": f,
    }
