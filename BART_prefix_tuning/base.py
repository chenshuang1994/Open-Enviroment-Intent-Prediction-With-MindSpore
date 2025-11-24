# MindSpore version of save_pretrained and ClassificationHead

import os
import mindspore as ms
from mindspore import nn, ops, Tensor


class SaveLoadMixin:
    """
    Lightweight version of HuggingFace's save_pretrained/load_pretrained,
    adapted for MindSpore. Handles:
        - save checkpoint
        - load checkpoint
        - save config.json
        - keep directory structure similar to PyTorch version
    """

    def save_pretrained(self, save_directory, config=None):
        """
        Save model checkpoint + config into a directory.

        save_directory/
            ├── config.json
            └── checkpoint.ckpt
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 1. save config
        if config is not None:
            config_path = os.path.join(save_directory, "config.json")
            config.to_json_file(config_path)

        # 2. save weight
        ckpt_path = os.path.join(save_directory, "checkpoint.ckpt")
        ms.save_checkpoint(self, ckpt_path)
        print(f"[MindSpore] Model saved at: {ckpt_path}")

    def load_pretrained(self, directory):
        """
        Load MindSpore checkpoint from a directory.
        """
        ckpt_path = os.path.join(directory, "checkpoint.ckpt")
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}")

        param_dict = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(self, param_dict)
        print(f"[MindSpore] Loaded checkpoint from: {ckpt_path}")


# ---------------------------------------------------------------
#  Classification Head (MindSpore)
# ---------------------------------------------------------------

class BARTClassificationHeadMS(nn.Cell):
    """
    MindSpore version of the sentence-level classification head.
    """

    def __init__(self, input_dim, inner_dim, num_classes, dropout=0.1):
        super().__init__()
        self.dense = nn.Dense(input_dim, inner_dim)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        self.out_proj = nn.Dense(inner_dim, num_classes)
        self.relu = ops.ReLU()

    def construct(self, hidden_states: Tensor):
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.relu(hidden_states)
        features = self.dropout(hidden_states)
        logits = self.out_proj(features)
        return logits, features
