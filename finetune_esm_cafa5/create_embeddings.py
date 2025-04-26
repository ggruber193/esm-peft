from pathlib import Path

import torch
import argparse
from transformers import EsmTokenizer, EsmModel

from finetune_esm_cafa5.dataset.cafa5 import Cafa5Dataset
from finetune_esm_cafa5.dataset.utils import sample_dataset


def tokenize(x, tokenizer):
    tokenizer_out = tokenizer(x,
                               return_attention_mask=False,
                               truncation=True,
                               max_length=1024,
                               padding=True,
                               return_tensors="pt")
    return {"input_ids": tokenizer_out["input_ids"],}


def inference(x, device, model):
    tokens = x["input_ids"]
    tokens = tokens.to(device)
    output = model(tokens)
    sequence_repr = output.last_hidden_state[:, 1: , :].squeeze(0).mean(0)
    del x
    torch.cuda.empty_cache()
    return sequence_repr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-name', type="str", default="esm2_t6_8M_UR50D")
    parser.add_argument('--cpu', type=int, default=1)

    args = parser.parse_args()
    data_dir = args.data_dir
    model_name = args.model_name

    cafa5_dataset = Cafa5Dataset(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = cafa5_dataset.embedding_dir.joinpath(model_name)

    model_ckpt = f"facebook/{model_name}"

    tokenizer = EsmTokenizer.from_pretrained(model_ckpt)

    dataset = cafa5_dataset.get_dataset().with_format("torch")

    keep_cols = {"sequence", "id"}
    for key in dataset:
        cols = set(dataset[key].column_names)
        cols = cols.difference(keep_cols)
        dataset[key] = dataset[key].remove_columns(list(cols))

    # tokenize sequences // truncate input_ids to esm2 training sequence length
    dataset = dataset.map(lambda x: tokenize(x["sequence"], tokenizer),
                          remove_columns=["sequence"], num_proc=8)

    model = EsmModel.from_pretrained(model_ckpt, torch_dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dataset = dataset.map(lambda x: {"embedding": inference(x, device, model)},
                              remove_columns=["input_ids"])

    dataset.save_to_disk(output_dir)


if __name__ == '__main__':
    main()
