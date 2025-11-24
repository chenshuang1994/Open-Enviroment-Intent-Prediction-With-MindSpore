# model_ms.py
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindnlp.models.bart import BartForConditionalGeneration


class BARTClassificationHeadMS(nn.Cell):
    """MindSpore version of classification head."""

    def __init__(self, input_dim, inner_dim, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(1 - dropout)   # MindSpore: keep_prob = 1 - p
        self.dense = nn.Dense(input_dim, inner_dim)
        self.out_proj = nn.Dense(inner_dim, num_classes)
        self.relu = ops.ReLU()

    def construct(self, hidden_states):
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = self.relu(x)
        features = self.dropout(x)
        logits = self.out_proj(features)
        return logits, features


class Model(nn.Cell):

    def __init__(self, args):
        super().__init__()

        # ----------------------
        # 载入 MindNLP BART
        # ----------------------
        self.base_model = BartForConditionalGeneration.from_pretrained(args.model_name)
        self.config = self.base_model.config
        self.n_embd = self.config.d_model

        self.lamda_loss = args.lamda_loss

        # 分类头
        self.classifier = BARTClassificationHeadMS(
            input_dim=self.n_embd,
            inner_dim=self.n_embd,
            num_classes=args.num_labels,
            dropout=0.1
        )

        # Loss
        self.ce_loss = nn.CrossEntropyLoss()

    def construct(self, input_ids, attention_mask, labels=None, label_ids=None):
        """
        BART forward: 返回 encoder 最后层 hidden states + decoder loss
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # (batch, hidden)
        sent_emb = ops.mean(outputs.encoder_last_hidden_state, axis=1)

        # 分类头
        cls_logits, cls_features = self.classifier(sent_emb)

        gen_loss = outputs.loss
        cls_loss = Tensor(0.0, ms.float32)
        total_loss = Tensor(0.0, ms.float32)

        if label_ids is not None:
            cls_loss = self.ce_loss(cls_logits, label_ids)
            total_loss = gen_loss * self.lamda_loss + cls_loss * (1 - self.lamda_loss)

        return {
            "loss": total_loss,
            "feature": cls_features,
            "logits": cls_logits,
            "cls_loss": cls_loss,
            "gen_loss": gen_loss,
            "sent_emb": sent_emb
        }

    # ----------------- 用于 OOD 生成 -----------------
    def evaluate_step(self, input_ids, attention_mask, labels=None):
        """
        为 GenerateOOD 生成 beam=3 输出
        """

        generate_outputs = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=labels,
            num_return_sequences=3,
            num_beams=3,
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=8,
            min_length=2,
        )

        decode_tokens = generate_outputs["sequences"][:, labels.shape[1]:]

        forward_out = self.construct(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        forward_out.update({
            "decoder_tokens": decode_tokens
        })

        return forward_out
