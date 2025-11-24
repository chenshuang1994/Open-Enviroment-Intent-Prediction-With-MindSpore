import numpy as np
from mindspore import nn, ops, Tensor, Parameter
from mindspore.dataset import vision
import mindspore.numpy as mnp
import mindspore as ms
from gensim.summarization import bm25
from rouge import Rouge
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


###############################################
# Rouge Metric
###############################################
class RougeMetric:
    """ replacement of fastNLP.Metric """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = Rouge()
        self.score = []

    def update(self, label_text, decoder_tokens):
        decoded_preds = self.tokenizer.batch_decode(
            decoder_tokens, skip_special_tokens=True
        )

        descript = []
        beams = []
        for p in decoded_preds:
            beams.append(p.strip())
            if len(beams) == 3:
                descript.append("; ".join(beams))
                beams = []

        rouge_score = self.rouge.get_scores(hyps=descript, refs=label_text, avg=True)
        score = (
            rouge_score["rouge-1"]["f"] * 0.2 +
            rouge_score["rouge-2"]["f"] * 0.5 +
            rouge_score["rouge-l"]["f"] * 0.3
        )
        self.score.append(score)

    def get_metric(self):
        if len(self.score) == 0:
            return {"rouge_score": 0.0}
        avg = sum(self.score) / len(self.score)
        self.score = []
        return {"rouge_score": float(avg)}


###############################################
# BM25 Metric
###############################################
class BM25Metric:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.score = []

    def update(self, label_text, decoder_tokens):
        decoded_preds = self.tokenizer.batch_decode(
            decoder_tokens, skip_special_tokens=True
        )

        beams = []
        idx = 0
        for p in decoded_preds:
            beams.append(p.split())
            if len(beams) == 3:
                bm25Model = bm25.BM25(beams)
                label_words = label_text[idx].split()
                s = bm25Model.get_scores(label_words)
                self.score.append(max(s))
                beams = []
                idx += 1

    def get_metric(self):
        if len(self.score) == 0:
            return {"bm25_score": 0.0}
        avg = sum(self.score) / len(self.score)
        self.score = []
        return {"bm25_score": float(avg)}


###############################################
# Accuracy (for classification logits)
###############################################
class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, logits, label_ids):
        # logits: [bsz, num_labels]
        pred = ops.Softmax()(logits).argmax(axis=-1)
        label_ids = label_ids.astype("int32")

        self.correct += int((pred == label_ids).sum().asnumpy())
        self.total += int(label_ids.shape[0])

    def get_metric(self):
        if self.total == 0:
            return {"acc": 0.0}
        acc = self.correct / self.total
        self.correct = 0
        self.total = 0
        return {"acc": float(acc)}


###############################################
# Euclidean Metric (unchanged logic)
###############################################
def euclidean_metric(a, b):
    # a: [N, D], b: [M, D]
    n, m = a.shape[0], b.shape[0]
    a_expand = ops.ExpandDims()(a, 1).broadcast_to((n, m, a.shape[1]))
    b_expand = ops.ExpandDims()(b, 0).broadcast_to((n, m, b.shape[1]))
    logits = -((a_expand - b_expand) ** 2).sum(axis=2)
    return logits


###############################################
# BoundaryLoss (MindSpore version)
###############################################
class BoundaryLoss(nn.Cell):

    def __init__(self, num_labels=10, feat_dim=2):
        super().__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = Parameter(
            Tensor(np.random.randn(num_labels).astype(np.float32)),
            name="delta"
        )
        self.softplus = ops.Softpl
