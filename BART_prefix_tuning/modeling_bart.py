import mindspore as ms
from mindspore import nn, ops, Tensor
import mindspore.numpy as msnp
import math

def shift_tokens_right(input_ids: Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    MindSpore version: shift input ids one token right.
    """
    # create tensor filled with zeros (same shape)
    shifted_input_ids = ops.zeros_like(input_ids)
    # move 0...n-2 to 1...n-1
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    # set first token
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace -100 with pad_token_id
    mask = shifted_input_ids == -100
    shifted_input_ids = ops.select(mask, 
                                   Tensor(pad_token_id, shifted_input_ids.dtype), 
                                   shifted_input_ids)
    return shifted_input_ids


def _make_causal_mask(input_ids_shape, dtype=ms.float32, past_key_values_length=0):
    """
    MindSpore version of PyTorch _make_causal_mask.
    input_ids_shape: (batch, tgt_len)
    """
    bsz, tgt_len = input_ids_shape

    # mask: (tgt_len, tgt_len)
    mask = ops.full((tgt_len, tgt_len), float("-inf"), dtype=dtype)

    cond = msnp.arange(tgt_len)
    cond = cond.reshape(tgt_len, 1)
    mask = ops.select(cond >= msnp.arange(tgt_len), Tensor(0, dtype), mask)

    if past_key_values_length > 0:
        zero_block = ops.zeros((tgt_len, past_key_values_length), dtype=dtype)
        mask = ops.concat([zero_block, mask], axis=-1)

    mask = mask.expand_dims(0).expand_dims(0)
    mask = ops.tile(mask, (bsz, 1, 1, 1))
    return mask


def _expand_mask(mask: Tensor, dtype=ms.float32, tgt_len=None):
    """
    MindSpore version of PyTorch expand_mask.
    mask shape: (bsz, src_len)
    return: (bsz,1,tgt_len,src_len)
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    # shape → (bsz, 1, tgt_len, src_len)
    expanded = mask.reshape(bsz, 1, 1, src_len)
    expanded = ops.tile(expanded, (1, 1, tgt_len, 1)).astype(dtype)

    inverted = 1.0 - expanded
    large_neg = Tensor(-1e9, dtype)
    inverted = ops.select(inverted.astype(ms.bool_), large_neg, inverted)

    return inverted
class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    MindSpore version of Bart positional embedding
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # same offset logic as original BART
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def construct(self, input_ids_shape, past_key_values_length: int = 0):
        """
        input_ids_shape: tuple(batch, seq_len)
        """
        bsz, seq_len = input_ids_shape[:2]
        start = past_key_values_length
        end = past_key_values_length + seq_len

        positions = ops.arange(start, end, dtype=ms.int32)
        positions = positions + self.offset
        return super().construct(positions)
class BartAttention(nn.Cell):
    """MindSpore version of BART Multi-Head Attention"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, is_decoder=False, bias=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_decoder = is_decoder

        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.scaling = self.head_dim ** -0.5

        # projections
        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

        self.softmax = ops.Softmax(axis=-1)
        self.dropout_op = nn.Dropout(1 - dropout)

    def _shape(self, tensor, seq_len, bsz):
        # (bsz, seq, embed) → (bsz, num_heads, seq, head_dim)
        tensor = tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        return ops.transpose(tensor, (0, 2, 1, 3))

    def construct(
            self,
            hidden_states,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False,
            past_prompt=None,
    ):
        """
        MindSpore version of forward()
        """
        bsz, tgt_len, _ = hidden_states.shape
        prompt = past_prompt

        # ---- Query -----
        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = query_states.reshape(bsz * self.num_heads, tgt_len, self.head_dim)

        # ----- Key / Value ------
        if key_value_states is not None and past_key_value is not None:
            # cross-attention reuse past kv
            key_states, value_states = past_key_value
        elif key_value_states is not None:
            # cross-attention fresh
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)

            # prefix tuning: concat prefix kv
            if prompt is not None:
                key_states = ops.concat([prompt["prev_key"], key_states], axis=2)
                value_states = ops.concat([prompt["prev_value"], value_states], axis=2)

        elif past_key_value is not None:
            # reuse self-att past key/value
            key_states_new = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states_new = self._shape(self.v_proj(hidden_states), -1, bsz)

            key_states = ops.concat([past_key_value[0], key_states_new], axis=2)
            value_states = ops.concat([past_key_value[1], value_states_new], axis=2)

        else:
            # normal self-attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            # prefix
            if prompt is not None:
                key_states = ops.concat([prompt["prev_key"], key_states], axis=2)
                value_states = ops.concat([prompt["prev_value"], value_states], axis=2)

        # ----- flatten kv -----
        src_len = key_states.shape[2]
        key_states = key_states.reshape(bsz * self.num_heads, src_len, self.head_dim)
        value_states = value_states.reshape(bsz * self.num_heads, src_len, self.head_dim)

        # ----- Attention weights -----
        attn_weights = ops.bmm(query_states, ops.transpose(key_states, (0, 2, 1)))

        if attention_mask is not None:
            # reshape to (bsz * heads, tgt, src)
            attn_mask = attention_mask.reshape(bsz, 1, tgt_len, src_len)
            attn_mask = ops.tile(attn_mask, (1, self.num_heads, 1, 1))
            attn_mask = attn_mask.reshape(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights + attn_mask

        attn_weights = self.softmax(attn_weights)

        # head mask
        if layer_head_mask is not None:
            head_mask = ops.tile(layer_head_mask.reshape(1, -1, 1, 1),
                                 (bsz, 1, tgt_len, src_len))
            head_mask = head_mask.reshape(bsz * self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights * head_mask

        attn_probs = self.dropout_op(attn_weights)

        # ----- Output -----
        attn_output = ops.bmm(attn_probs, value_states)
        attn_output = attn_output.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        if self.is_decoder:
            past_key_value = (key_states.reshape(bsz, self.num_heads, src_len, self.head_dim),
                              value_states.reshape(bsz, self.num_heads, src_len, self.head_dim))
        else:
            past_key_value = None

        attn_weights_return = (
            attn_weights.reshape(bsz, self.num_heads, tgt_len, src_len)
            if output_attentions else None
        )

        return attn_output, attn_weights_return, past_key_value
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from transformers.activations import ACT2FN


class BartEncoderLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm((self.embed_dim,))
        self.dropout = config.dropout

        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Dense(config.encoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm((self.embed_dim,))

        self.dropout_layer = nn.Dropout(1 - self.dropout)
        self.activation_dropout_layer = nn.Dropout(1 - self.activation_dropout)

    def construct(
            self,
            hidden_states,
            attention_mask,
            layer_head_mask,
            output_attentions=False,
            past_prompt=None,
    ):
        """
        hidden_states: (bsz, seq_len, dim)
        """
        prompt = past_prompt
        residual = hidden_states

        # ===== Self Attention =====
        attn_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            past_prompt=prompt["encoder_prompt"] if prompt else None,
        )

        attn_output = self.dropout_layer(attn_output)
        hidden_states = residual + attn_output
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # ===== FFN =====
        residual = hidden_states
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.activation_dropout_layer(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        # ===== numerical stability =====
        if hidden_states.dtype == ms.float16:
            # clamp to avoid NaN/Inf
            clamp_value = Tensor(ms.finfo(ms.float16).max - 1000, ms.float16)
            hidden_states = ops.clip_by_value(hidden_states, -clamp_value, clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from transformers.activations import ACT2FN


class BartDecoderLayer(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.d_model

        # ===== decoder self-attention =====
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm((self.embed_dim,))

        # ===== cross-attention =====
        self.encoder_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_cross_attention=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm((self.embed_dim,))

        # ===== FFN =====
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.fc1 = nn.Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Dense(config.decoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm((self.embed_dim,))

        # ===== dropouts =====
        self.dropout = config.dropout
        self.dropout_layer = nn.Dropout(1 - self.dropout)
        self.activation_dropout_layer = nn.Dropout(1 - self.activation_dropout)

    def construct(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        layer_head_mask,
        cross_attn_layer_head_mask,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        past_prompt=None,
    ):
        """
        hidden_states: (bsz, tgt_len, dim)
        encoder_hidden_states: (bsz, src_len, dim)
        past_key_value: ((prev_self_k, prev_self_v), (prev_cross_k, prev_cross_v))
        """
        present_key_value = ()

        # =====================
        # 1. Decoder Self Attention
        # =====================
        residual = hidden_states

        # prefix prompt
        self_prompt = past_prompt["decoder_prompt"] if past_prompt else None

        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value[0] if past_key_value is not None else None,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
            past_prompt=self_prompt,
        )

        attn_output = self_attn_outputs[0]

        # update kv cache
        if use_cache:
            present_key_value = present_key_value + (self_attn_outputs[1],)

        # dropout + residual + LN
        hidden_states = self.dropout_layer(attn_output)
        hidden_states = self.self_attn_layer_norm(hidden_states + residual)

        # numeric stability
        hidden_states = self._numerical_stable(hidden_states)

        outputs = ()
        if output_attentions:
            outputs = outputs + (self_attn_outputs[2],)

        # =====================
        # 2. Cross Attention (Encoder–Decoder Attention)
        # =====================
        if encoder_hidden_states is not None:
            residual = hidden_states

            cross_attn_outputs = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value[1] if past_key_value is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                past_prompt=past_prompt["encoder_prompt"] if past_prompt else None,
            )

            attn_output = cross_attn_outputs[0]

            if use_cache:
                present_key_value = present_key_value + (cross_attn_outputs[1],)

            hidden_states = self.dropout_layer(attn_output)
            hidden_states = self.encoder_attn_layer_norm(hidden_states + residual)

            hidden_states = self._numerical_stable(hidden_states)

            if output_attentions:
                outputs = outputs + (cross_attn_outputs[2],)

        # =====================
        # 3. Feed Forward Layer
        # =====================
        residual = hidden_states

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.activation_dropout_layer(hidden_states)
        hidden_states = self.fc2(hidden_states)

        hidden_states = self.dropout_layer(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states + residual)

        hidden_states = self._numerical_stable(hidden_states)

        outputs = (hidden_states,) + outputs

        if use_cache:
            outputs = outputs + (present_key_value,)

        return outputs

    def _numerical_stable(self, x):
        """防止 float16 下的 NaN/Inf"""
        if x.dtype == ms.float16:
            limit = Tensor(ms.finfo(ms.float16).max - 1000, ms.float16)
            x = ops.clip_by_value(x, -limit, limit)
        return x
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class BartEncoder(nn.Cell):
    def __init__(self, config, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.dropout_layer = nn.Dropout(1 - self.dropout)

        self.layerdrop = config.encoder_layerdrop
        self.embed_scale = config.scale_embedding

        embed_dim = config.d_model
        self.embed_tokens = embed_tokens
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim
        )

        self.layers = nn.CellList([
            BartEncoderLayer(config)
            for _ in range(config.encoder_layers)
        ])

        self.layernorm_embedding = nn.LayerNorm((embed_dim,))

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        past_prompt=None,
    ):
        """
        past_prompt: {"encoder_prompt": (bsz, prompt_len, dim)}
        """
        inputs_embeds = self.embed_tokens(input_ids)

        if self.embed_scale:
            inputs_embeds = inputs_embeds * self.embed_scale

        # add encoder prompt before positional encoding
        if past_prompt is not None:
            prompt = past_prompt["encoder_prompt"]     # (bsz, p_len, dim)
            inputs_embeds = ops.concat([prompt, inputs_embeds], axis=1)

            # attention_mask 也要补前缀
            if attention_mask is not None:
                bsz, plen = prompt.shape[0], prompt.shape[1]
                prefix_mask = ops.ones((bsz, plen), ms.int32)
                attention_mask = ops.concat([prefix_mask, attention_mask], axis=1)

        # position embeddings
        pos_embeds = self.embed_positions(input_ids.shape)
        hidden_states = inputs_embeds + pos_embeds

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)

        # expand attention mask 到 (bsz, 1, 1, seq_len)
        if attention_mask is not None:
            attention_mask = self.invert_attention_mask(attention_mask)

        # encoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_prompt=None,      # encoder 无 cross_attention
                output_attentions=False,
            )[0]

        return hidden_states, attention_mask

    def invert_attention_mask(self, mask):
        """
        将 1→0，0→-inf 供 attention 使用
        """
        mask = mask.astype(ms.float32)
        inverted_mask = (1.0 - mask) * -10000.0
        return inverted_mask[:, None, None, :]
class BartDecoder(nn.Cell):
    def __init__(self, config, embed_tokens):
        super().__init__()

        embed_dim = config.d_model
        self.dropout = config.dropout
        self.dropout_layer = nn.Dropout(1 - self.dropout)
        self.embed_scale = config.scale_embedding

        self.embed_tokens = embed_tokens
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )

        self.layers = nn.CellList([
            BartDecoderLayer(config)
            for _ in range(config.decoder_layers)
        ])

        self.layernorm_embedding = nn.LayerNorm((embed_dim,))

    def construct(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        past_prompt=None,
    ):
        """
        input_ids shape: (bsz, tgt_len)
        past_prompt: {"decoder_prompt", "encoder_prompt"}
        """

        # embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        if self.embed_scale:
            inputs_embeds = inputs_embeds * self.embed_scale

        # decoder prompt
        if past_prompt is not None:
            prompt = past_prompt["decoder_prompt"]  # (bsz, p_len, dim)
            inputs_embeds = ops.concat([prompt, inputs_embeds], axis=1)

            # attention mask 也要补 prefix
            if attention_mask is not None:
                bsz, plen = prompt.shape[0], prompt.shape[1]
                prefix_mask = ops.ones((bsz, plen), ms.int32)
                attention_mask = ops.concat([prefix_mask, attention_mask], axis=1)

        # pos embeddings
        pos_embeds = self.embed_positions(inputs_embeds.shape)
        hidden_states = inputs_embeds + pos_embeds
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.dropout_layer(hidden_states)

        # mask expand
        if attention_mask is not None:
            attention_mask = self.invert_attention_mask(attention_mask)

        if encoder_attention_mask is not None:
            encoder_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )

        # init cache
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_decoder_cache = () if use_cache else None

        # decoder layers
        all_attentions = []
        for i, decoder_layer in enumerate(self.layers):

            past_key_value = past_key_values[i]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                past_prompt=past_prompt,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions.append(layer_outputs[1])

        outputs = (hidden_states, next_decoder_cache)

        if output_attentions:
            outputs += (all_attentions,)

        return outputs

    def invert_attention_mask(self, mask):
        mask = mask.astype(ms.float32)
        return (1.0 - mask) * -10000.0
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class PrefixEncoder(nn.Cell):
    """
    Prefix prefix encoder for Prefix-Tuning
    作用：输入 prefix_id（0…pre_seq_len-1）
         输出：encoder_prompt & decoder_prompt
    """

    def __init__(self, config):
        super().__init__()

        self.pre_seq_len = config.pre_seq_len
        self.hidden_size = config.d_model

        # prefix embedding table
        self.prefix_embeddings = nn.Embedding(
            config.pre_seq_len,
            config.prefix_hidden_size
        )

        # MLP 将 prefix_hidden_size → d_model
        self.trans = nn.SequentialCell([
            nn.Dense(config.prefix_hidden_size, config.prefix_hidden_size),
            nn.Tanh(),
            nn.Dense(config.prefix_hidden_size, config.d_model)
        ])

    def construct(self, batch_size):
        """
        return:
            {
                "encoder_prompt": (bsz, pre_seq_len, d_model)
                "decoder_prompt": (bsz, pre_seq_len, d_model)
            }
        """
        prefix_tokens = ops.arange(
            0, self.pre_seq_len, dtype=ms.int32
        )  # (pre_seq_len,)

        prefix_embeds = self.prefix_embeddings(prefix_tokens)  # (pre_seq_len, prefix_hidden)
        past_prompt = self.trans(prefix_embeds)                # (pre_seq_len, d_model)

        # batch expand
        past_prompt = ops.expand_dims(past_prompt, 0)          # (1, pre_seq_len, d_model)
        past_prompt = ops.tile(past_prompt, (batch_size, 1, 1))

        # for BART, encoder & decoder prefix 可以相同
        return {
            "encoder_prompt": past_prompt,
            "decoder_prompt": past_prompt
        }
class BartModel(nn.Cell):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Embedding shared
        self.shared = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        # encoder/decoder
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # prefix tuning
        if config.use_prefix:
            self.prefix_encoder = PrefixEncoder(config)
        else:
            self.prefix_encoder = None

        # output head
        self.lm_head = nn.Dense(config.d_model, config.vocab_size, has_bias=False)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

    def _prepare_decoder_inputs(self, input_ids):
        """
        decoder_start: 将 decoder 输入右移
        """
        pad = self.config.pad_token_id
        bos = self.config.decoder_start_token_id

        shifted_ids = ops.zeros_like(input_ids)
        shifted_ids[:, 0] = bos
        shifted_ids[:, 1:] = input_ids[:, :-1]

        return shifted_ids

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
    ):

        bsz = input_ids.shape[0]

        # prefix tuning
        if self.prefix_encoder is not None:
            past_prompt = self.prefix_encoder(bsz)
        else:
            past_prompt = None

        # encoder forward
        encoder_hidden_states, encoder_mask = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
        )

        # prepare decoder inputs
        if decoder_input_ids is None:
            decoder_input_ids = self._prepare_decoder_inputs(labels)

        # decoder forward
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_mask,
            past_prompt=past_prompt,
            use_cache=False,
            output_attentions=False
        )

        last_hidden_state = decoder_outputs[0]  # (bsz, tgt_len, d_model)
        logits = self.lm_head(last_hidden_state)

        output = {"logits": logits}

        # compute generation (LM) loss
        if labels is not None:
            lm_logits = logits.reshape((-1, self.config.vocab_size))
            lm_labels = labels.reshape((-1,))
            loss = self.loss_fn(lm_logits, lm_labels)
            output["loss"] = loss

        return output
class BartModel(nn.Cell):
    ...
    def generate(
        self,
        input_ids,
        attention_mask,
        max_length=50
    ):
        bsz = input_ids.shape[0]

        # prefix
        if self.prefix_encoder is not None:
            past_prompt = self.prefix_encoder(bsz)
        else:
            past_prompt = None

        # encoder
        encoder_hidden_states, encoder_mask = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
        )

        # 初始 decoder 输入
        cur_ids = ops.full(
            (bsz, 1),
            self.config.decoder_start_token_id,
            dtype=ms.int32
        )

        past_key_values = [None] * self.config.decoder_layers

        for step in range(max_length):

            outputs = self.decoder(
                input_ids=cur_ids[:, -1:],       # 只输入最后一个 token
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_mask,
                past_key_values=past_key_values,
                use_cache=True,
                past_prompt=past_prompt,
            )

            hidden, past_key_values = outputs
            logits = self.lm_head(hidden[:, -1, :])  # last token

            next_token = ops.argmax(logits, axis=-1).reshape((bsz, 1))
            cur_ids = ops.concat([cur_ids, next_token], axis=1)

        return cur_ids
