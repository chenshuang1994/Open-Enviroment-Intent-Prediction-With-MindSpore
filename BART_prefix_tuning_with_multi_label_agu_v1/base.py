import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, save_checkpoint, load_checkpoint

from transformers.modeling_utils import (
    ModuleUtilsMixin, PushToHubMixin,
    logging, Union, Optional, Callable,
    FLAX_WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, WEIGHTS_NAME,
    is_offline_mode, is_remote_url, unwrap_model, get_parameter_dtype
)

logger = logging.get_logger(__name__)


class PushToHubFriendlyModel(nn.Cell, ModuleUtilsMixin, PushToHubMixin):
    """
    MindSpore 版本
    - 用 ms.save_checkpoint 替换 torch.save
    - 用 load_checkpoint 替换 torch.load
    - 去掉 device 相关处理
    """

    def __init__(self):
        super(PushToHubFriendlyModel, self).__init__()

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            save_config: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = save_checkpoint,
            push_to_hub: bool = False,
            **kwargs,
    ):
        """
        MindSpore 改写版本
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = unwrap_model(self)

        dtype = get_parameter_dtype(model_to_save)
        self.pretrain_model.config.mindspore_dtype = str(dtype).split(".")[-1]

        self.pretrain_model.config.architectures = [model_to_save.__class__.__name__]

        if save_config:
            self.pretrain_model.config.save_pretrained(save_directory)

        if state_dict is None:
            # MS 需要把参数转成 list[dict]
            ms_state_dict = []
            for name, param in model_to_save.parameters_and_names():
                ms_state_dict.append({"name": name, "data": param.data})
        else:
            ms_state_dict = state_dict

        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        save_function(ms_state_dict, output_model_file)
        logger.info(f"Model weights saved in {output_model_file}")

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Model pushed to the hub in this commit: {url}")

    def load(self, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        MindSpore 版本 load
        """
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        mirror = kwargs.pop("mirror", None)

        from_pt = not (from_tf or from_flax)

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)

            if os.path.isdir(pretrained_model_name_or_path):
                if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        f"Could not find checkpoint in {pretrained_model_name_or_path}"
                    )

            elif os.path.isfile(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            else:
                raise EnvironmentError(f"Invalid model path {pretrained_model_name_or_path}")

            logger.info(f"Loading weights file {archive_file}")

        else:
            raise EnvironmentError("pretrained_model_name_or_path is None")

        if state_dict is None:
            try:
                state_dict = load_checkpoint(archive_file)
            except Exception as e:
                raise OSError(f"Unable to load weights from MindSpore checkpoint: {e}")

        # MindSpore 需要调用 load_state_dict (你模型内部需要自行实现)
        self.load_state_dict(state_dict)


class BARTClassificationHead(nn.Cell):
    """MindSpore 版本的 BART Classification Head"""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super(BARTClassificationHead, self).__init__()
        self.dense = nn.Dense(input_dim, inner_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=1 - pooler_dropout)
        self.out_proj = nn.Dense(inner_dim, num_classes)

    def construct(self, hidden_states: Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.relu(hidden_states)
        features = self.dropout(hidden_states)
        logits = self.out_proj(features)
        return logits, features
