import argparse
from fastNLP import DataSet, Instance   # 若你使用 fastNLP-ms，请保持一致


def load_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default='data', type=str,
                        help="Input data directory.")
    parser.add_argument("--save_results_path", type=str, default='outputs',
                        help="Where to save results.")
    parser.add_argument("--pretrain_dir", default='models', type=str,
                        help="Directory to save checkpoints.")
    parser.add_argument("--model_name", default="facebook/bart-base", type=str,
                        help="Pretrained BART name or path.")
    parser.add_argument("--max_seq_length", default=None, type=int)

    parser.add_argument("--dataset", default='banking', type=str)
    parser.add_argument("--known_cls_ratio", default=0.75, type=float)
    parser.add_argument("--labeled_ratio", default=1.0, type=float)

    parser.add_argument('--seed', type=int, default=52)
    parser.add_argument("--gpu_id", type=str, default="1")

    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)

    parser.add_argument("--wait_patient", default=10, type=int)
    parser.add_argument("--feat_dim", default=768, type=int)
    parser.add_argument("--lr_boundary", default=0.05, type=float)

    # prefix-tuning
    parser.add_argument("--pre_seq_len", default=12, type=int)
    parser.add_argument("--prefix_projection", default=False, type=bool)
    parser.add_argument("--prefix_hidden_size", default=768, type=int)
    parser.add_argument("--prefix_dropout", default=0.5, type=float)

    parser.add_argument("--lamda_loss", default=0.5, type=float)
    parser.add_argument("--p_node", default=0.4, type=float)
    parser.add_argument("--delta", default=1.0, type=float)

    args = parser.parse_args()
    return args


def get_unkdataset_of_dataset(dataset: DataSet, unk_label_id: int) -> DataSet:
    """
    提取未知类数据（label_id >= unk_label_id）。
    """
    unkDs = DataSet()
    for ins in dataset:
        if int(ins['label_ids']) >= int(unk_label_id):
            unkDs.append(
                Instance(
                    input_ids=ins['input_ids'],
                    attention_mask=ins['attention_mask'],
                    labels=ins['labels'],
                    label_ids=ins['label_ids']
                )
            )
    return unkDs


def load_ptuning_v2_model_tokenizer(args, data):
    """
    由于你现在已经改成 MindSpore 框架，
    此处我们需要加载 MindSpore 版的 Ptuningv2 模型。
    
    你可以后续把 Ptuningv2 迁移后放在：
        Ptuningv2_Bart_ms/prefixtuning_ms.py
    """
    from transformers import GPT2Config
    from Ptuningv2_Bart_ms.prefixtuning_ms import Model   # ← 这是你迁移后的 MindSpore 版本

    config = GPT2Config.from_pretrained(args.gpt_model)
    model = Model(args)

    # MindSpore 版 tokenizer 通常无需 resize embedding
    # 若你需要，可自行调用 model.resize_token_embeddings()
    try:
        model.resize_token_embeddings(len(data.tokenizer))
    except:
        pass

    return model, data.tokenizer
