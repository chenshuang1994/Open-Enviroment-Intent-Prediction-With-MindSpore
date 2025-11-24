import sys
sys.path.append("../")

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from transformers import AutoTokenizer
from BART_prefix_tuning_with_multi_label_agu_v1.base_ms import PushToHubFriendlyModel, BARTClassificationHead_MS
from BART_prefix_tuning_with_multi_label_agu_v1.modeling_bart_ms import BartForConditionalGeneration
import numpy as np


###############################################
# Multi-label Circle Loss（MindSpore 版本）
###############################################
class MultiLabelCircleLoss_MS(nn.Cell):
    def __init__(self, reduction="mean", inf=1e12):
        super().__init__()
        self.reduction = reduction
        self.inf = inf

        self.logsumexp = ops.ReduceLogSumExp(keep_dims=False)
        self.concat = ops.Concat(axis=-1)
        self.zeros_like = ops.ZerosLike()

    def construct(self, logits, labels):
        # logits, labels: (batch, num_classes)
        logits = (1 - 2 * labels) * logits

        logits_neg = logits - labels * self.inf
        logits_pos = logits - (1 - labels) * self.inf

        zeros = self.zeros_like(logits[..., :1])

        logits_neg = self.concat((logits_neg, zeros))
        logits_pos = self.concat((logits_pos, zeros))

        neg_loss = self.logsumexp(logits_neg, -1)
        pos_loss = self.logsumexp(logits_pos, -1)

        loss = neg_loss + pos_loss
        if self.reduction == "mean":
            return ops.mean(loss)
        else:
            return ops.sum(loss)


###############################################
# Prefix-tuning with Multi-label Classification
###############################################
class Model_MS(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.preseqlen = args.pre_seq_len
        self.mid_dim = args.prefix_hidden_size
        self.dropout = nn.Dropout(1 - args.prefix_dropout)

        print(f"prefix length = {self.preseqlen}")

        #############################################
        # tokenizer & pretrained BART (HF transformers)
        #############################################
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        self.pretrain_model = BartForConditionalGeneration.from_pretrained(args.model_name)

        cfg = self.pretrain_model.config
        self.match_n_layer = cfg.decoder_layers
        self.match_n_head = cfg.decoder_attention_heads
        self.n_embd = cfg.d_model
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head

        #############################################
        # prefix embeddings
        #############################################
        self.input_tokens = ms.Parameter(
            Tensor(np.arange(self.preseqlen), ms.int32),
            requires_grad=False
        )

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.SequentialCell(
            nn.Dense(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Dense(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        # encoder prefix
        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.SequentialCell(
            nn.Dense(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Dense(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        # decoder cross-attention
        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.SequentialCell(
            nn.Dense(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Dense(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        #############################################
        # classification head
        #############################################
        self.trans_layer = nn.TransformerEncoderLayer(
            hidden_size=self.n_embd,
            ffn_hidden_size=self.n_embd * 4,
            num_heads=self.match_n_head,
            dropout=1 - args.prefix_dropout,
            batch_first=True
        )

        self.classifier = BARTClassificationHead_MS(
            input_dim=self.n_embd,
            inner_dim=self.n_embd,
            num_classes=args.num_labels,
            pooler_dropout=0.1
        )

        #############################################
        # losses
        #############################################
        self.lamda_loss = args.lamda_loss
        self.m_loss = args.m_loss

        self.ce = nn.CrossEntropyLoss()
        self.circle_loss = MultiLabelCircleLoss_MS()

        #############################################
        # Freeze pretrained BART
        #############################################
        for p in self.pretrain_model.get_parameters():
            p.requires_grad = False


    #####################################################
    # prefix prompt generation（MindSpore）
    #####################################################
    def get_prompt(self, bsz, sample_size=1):
        full_bsz = bsz * sample_size

        input_tok = ops.repeat_interleave(self.input_tokens[None, :], full_bsz, axis=0)
        enc_tok = ops.repeat_interleave(self.input_tokens[None, :], bsz, axis=0)

        # decoder self-attention
        dec_embed = self.wte(input_tok)
        p = self.control_trans(dec_embed)
        p = p.reshape(full_bsz, self.preseqlen,
                      self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
        p = self.dropout(p)
        p = ops.transpose(p, (2, 0, 3, 1, 4)).split(2, axis=0)

        # cross attention
        dec_embed2 = self.wte_dec(input_tok)
        p2 = self.control_trans_dec(dec_embed2)
        p2 = p2.reshape(full_bsz, self.preseqlen,
                        self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
        p2 = self.dropout(p2)
        p2 = ops.transpose(p2, (2, 0, 3, 1, 4)).split(2, axis=0)

        # encoder prefix
        enc_embed = self.wte_enc(enc_tok)
        p3 = self.control_trans_enc(enc_embed)
        p3 = p3.reshape(bsz, self.preseqlen,
                        self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
        p3 = self.dropout(p3)
        p3 = ops.transpose(p3, (2, 0, 3, 1, 4)).split(2, axis=0)

        result = []
        for i in range(len(p)):
            result.append(dict(
                decoder_prompt=dict(
                    prev_key=p[i][0],
                    prev_value=p[i][1],
                    prev_key_padding_mask=ops.zeros((full_bsz, self.preseqlen), ms.bool_)
                ),
                cross_attention_prompt=dict(
                    prev_key=p2[i][0],
                    prev_value=p2[i][1],
                    prev_key_padding_mask=ops.zeros((full_bsz, self.preseqlen), ms.bool_)
                ),
                encoder_prompt=dict(
                    prev_key=p3[i][0],
                    prev_value=p3[i][1],
                    prev_key_padding_mask=ops.zeros((bsz, self.preseqlen), ms.bool_)
                )
            ))
        return result


    #####################################################
    # forward
    #####################################################
    def construct(self, input_ids, attention_mask, labels=None, multi_labels=None, label_ids=None):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(bsz)

        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
            output_hidden_states=True,
        )

        enc_hid = outputs.encoder_last_hidden_state
        enc_hid = self.trans_layer(enc_hid)
        pooled = ops.mean(enc_hid, axis=1)

        cls_logits, cls_features = self.classifier(pooled)

        gen_loss = outputs.loss
        ce_loss = Tensor(0.0, ms.float32)
        circle_loss = Tensor(0.0, ms.float32)
        total_loss = Tensor(0.0, ms.float32)

        if multi_labels is not None:
            ce_loss = self.ce(cls_logits, label_ids)
            circle_loss = self.circle_loss(cls_logits, multi_labels)

            cls_loss_final = circle_loss * self.m_loss + ce_loss * (1 - self.m_loss)
            total_loss = gen_loss * self.lamda_loss + cls_loss_final * (1 - self.lamda_loss)

        return dict(
            loss=total_loss,
            feature=cls_features,
            logits=cls_logits,
            gen_loss=gen_loss,
            cls_loss=circle_loss,
            ce_loss=ce_loss
        )


    #####################################################
    # generation
    #####################################################
    def evaluate_step(self, input_ids, attention_mask, labels=None):
        decoder_input_ids = labels

        gen = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            num_beams=3,
            num_return_sequences=3,
            output_scores=True,
            max_length=decoder_input_ids.shape[1] + 10,
            min_length=decoder_input_ids.shape[1] + 2,
            return_dict_in_generate=True,
        )

        result = {
            "decoder_tokens": gen.sequences[:, decoder_input_ids.shape[1]:]
        }
        result.update(self.construct(input_ids, attention_mask, labels=labels))
        return result


    def generate(self, input_ids, attention_mask, **kwargs):
        bsz = input_ids.shape[0]
        past_prompt = self.get_prompt(bsz, sample_size=kwargs["num_beams"])
        return self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )
