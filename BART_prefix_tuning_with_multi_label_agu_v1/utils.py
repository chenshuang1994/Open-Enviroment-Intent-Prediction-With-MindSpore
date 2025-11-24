import mindspore as ms
from mindspore import ops


def pack_labels_batch_ms(labels_batch):
    """
    将多组 label_list(batch) 按照 beam 位置对齐，
    与 PyTorch pack_labels_batch 逻辑保持一致，只是改为 MindSpore Tensor.
    """

    # 初始化：每个位置存一个 list
    batch = [[] for _ in range(len(labels_batch[0]))]

    # 重新组织 labels
    for label_batch in labels_batch:
        for idx, label in enumerate(label_batch):
            batch[idx].append(label)

    # stack 成 MS Tensor
    batch = [ops.stack(item, axis=0) for item in batch]

    return batch
