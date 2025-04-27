from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_from_disk
import litdata
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from finetune_esm_cafa5.dataset.cafa5 import Cafa5Dataset
from finetune_esm_cafa5.models.classifier import Classifier, TrainingConfig
from finetune_esm_cafa5.models.mlp import MLPConfig


def collate(x: list[dict[str, list]]) -> dict[str, torch.Tensor]:
    return {"embedding": torch.tensor([i["embedding"] for i in x]), "labels": torch.tensor([i["labels"] for i in x], dtype=torch.float32)}


if __name__ == '__main__':
    model_name = "esm2_t6_8M_UR50D"
    total_batch_size = 128
    batch_size = 128
    grad_accum_steps = total_batch_size // batch_size
    data_dir = Path("../data/cafa5")
    subdir = "baseline"
    epochs = 100

    dataset, input_size, output_size, pos_weight = Cafa5Dataset("../data/cafa5", filter_terms="../data/cafa5/relevant_labels.txt").get_dataset_for_classification(model_name)


    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=8)
    eval_loader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=collate, num_workers=8)

    model_config = MLPConfig(input_size=input_size, output_size=output_size, hidden_sizes=[280])


    training_config = TrainingConfig(lr=0.001, pos_weight=pos_weight)
    classifier = Classifier(model_config=model_config, training_config=training_config)

    logger = TensorBoardLogger(save_dir=data_dir, version=subdir, sub_dir=model_name)

    trainer = Trainer(logger=logger,
                      max_epochs=epochs,
                      accumulate_grad_batches=grad_accum_steps)

    trainer.fit(classifier, train_loader, eval_loader)
