import sys
sys.path.append("../")

import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
import numpy as np

from transformers import AutoTokenizer
from BART_prefix_tuning.base_ms import PushToHubFriendlyModel
from BART_prefix_tuning.modeling_bart_ms import BartForConditionalGeneration
from BART_prefix_tuning.classifier_ms import BARTClassificationHead   # 你之前迁移的分类头


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """Prefix-tuning parameters"""
        self.preseqlen = args.pre_seq_len
        self.mid_dim = args.prefix_hidden_size

        print(f"prefix-tuning sequence length is {self.preseqlen}.")

        # ----------------------
        # Load tokenizer & BART
        # ----------------------
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        self.pretrain_model = BartForConditionalGeneration.from_pretrained(args.model_name)

        self.config = self.pretrain_model.config
        self.match_n_layer = self.config.decoder_layers
        self.match_n_head = self.config.decoder_attention_heads
        self.n_embd = self.config.d_model

        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head

        # ----------------------
        # Prefix Embedding
        # ----------------------
        self.input_tokens = Parameter(
            Tensor(np.arange(self.preseqlen), ms.int32), requires_grad=False
        )

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.SequentialCell([
            nn.Dense(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Dense(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        ])

        # encoder prefix
        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.SequentialCell([
            nn.Dense(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Dense(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        ])

        # cross prefix
        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.SequentialCell([
            nn.Dense(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Dense(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        ])

        self.dropout = nn.Dropout(args.prefix_dropout)

        # classifier
        self.transformer_layer = nn.TransformerEncoderLayer(
            hidden_size=self.n_embd,
            ffn_hidden_size=self.n_embd * 4,
            num_heads=self.match_n_head,
            dropout_rate=0.1,
            batch_first=True
        )
        self.classifier = BARTClassificationHead(
            input_dim=self.n_embd,
            inner_dim=self.n_embd,
            num_classes=args.num_labels,
            pooler_dropout=0.1
        )

        self.lamda_loss = args.lamda_loss

        # freeze bart parameters
        for p in self.pretrain_model.get_parameters():
            p.requires_grad = False

        # count prefix params
        total_trainable = sum([p.size for p in self.get_parameters() if p.requires_grad])
        bart_params = sum([p.size for p in self.pretrain_model.get_parameters()])

        print(f"Prefix tunable params: {total_trainable}")
        print(f"BART params: {bart_params}")

    # -------------------------------------------------------------------
    #                      Construct Prefix Prompt
    # -------------------------------------------------------------------
    def get_prompt(self, bsz, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size

        input_tokens = ops.ExpandDims()(self.input_tokens, 0)      # (1, preseqlen)
        input_tokens = ops.Tile()(input_tokens, (bsz, 1))          # (bsz, preseqlen)

        # -------- decoder prefix --------
        temp_control = self.wte(input_tokens)
        past_key_values = self.control_trans(temp_control)

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.reshape(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = ops.Transpose()(past_key_values, (2, 0, 3, 1, 4))
        past_key_values = ops.Split(axis=0, output_num=self.match_n_layer)(past_key_values)

        # -------- cross prefix --------
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(temp_control_dec)
        past_key_values_dec = past_key_values_dec.reshape(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = ops.Transpose()(past_key_values_dec, (2, 0, 3, 1, 4))
        past_key_values_dec = ops.Split(axis=0, output_num=self.match_n_layer)(past_key_values_dec)

        # -------- encoder prefix --------
        input_tokens_enc = ops.Tile()(
            ops.ExpandDims()(self.input_tokens, 0),
            (old_bsz, 1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)
        past_key_values_enc = past_key_values_enc.reshape(
            old_bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = ops.Transpose()(past_key_values_enc, (2, 0, 3, 1, 4))
        past_key_values_enc = ops.Split(axis=0, output_num=self.match_n_layer)(past_key_values_enc)

        # return prefix dictionary per layer
        result = []
        for i in range(self.match_n_layer):
            r = {}
            # decoder prefix
            dec_key, dec_value = ops.Split(axis=0, output_num=2)(past_key_values[i])
            r["decoder_prompt"] = {
                "prev_key": dec_key,
                "prev_value": dec_value,
                "prev_key_padding_mask": ops.Zeros()((bsz, seqlen), ms.bool_)
            }
            # cross-attn prefix
            cross_key, cross_value = ops.Split(axis=0, output_num=2)(past_key_values_dec[i])
            r["cross_attention_prompt"] = {
                "prev_key": cross_key,
                "prev_value": cross_value,
                "prev_key_padding_mask": ops.Zeros()((bsz, seqlen), ms.bool_)
            }
            # encoder prefix
            enc_key, enc_value = ops.Split(axis=0, output_num=2)(past_key_values_enc[i])
            r["encoder_prompt"] = {
                "prev_key": enc_key,
                "prev_value": enc_value,
                "prev_key_padding_mask": ops.Zeros()((old_bsz, seqlen), ms.bool_)
            }
            result.append(r)

        return result

    # -------------------------------------------------------------------
    #                          Forward
    # -------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, labels=None, label_ids=None):
        bsz = input_ids.shape[0]
        past_prompt = self.get_prompt(bsz)

        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt
        )

        # classification: use encoder hidden state
        encoder_hidden = outputs.encoder_last_hidden_state
        trans_output = self.transformer_layer(encoder_hidden)
        pooled_output = ops.ReduceMean(keep_dims=False)(trans_output, 1)

        cls_logits, cls_features = self.classifier(pooled_output)

        gen_loss = outputs.loss
        cls_loss = ms.Tensor(0.0, ms.float32)

        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits, label_ids)

        total_loss = gen_loss * self.lamda_loss + cls_loss * (1 - self.lamda_loss)

        return {
            "loss": total_loss,
            "feature": cls_features,
            "logits": cls_logits,
            "gen_loss": gen_loss,
            "cls_loss": cls_loss
        }

    # -------------------------------------------------------------------
    #                          Evaluate
    # -------------------------------------------------------------------
    def evaluate_step(self, input_ids, attention_mask, labels=None):
        decoder_input_ids = labels

        generate_tokens = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            early_stopping=True,
            num_beams=3,
            num_return_sequences=3,
            output_scores=True,
            max_length=decoder_input_ids.shape[1] + 8,
            min_length=decoder_input_ids.shape[1] + 2,
            return_dict_in_generate=True,
        )

        result = {
            "decoder_tokens": generate_tokens.sequences[:, decoder_input_ids.shape[1]:]
        }

        # classification part
        result.update(
            self.forward(input_ids=input_ids, attention_mask=attention_mask)
        )
        return result

    # -------------------------------------------------------------------
    #                         Generate
    # -------------------------------------------------------------------
    def generate(self, input_ids, attention_mask, **kwargs):
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(bsz, sample_size=kwargs["num_beams"])

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs
        )
        return generated_ids
