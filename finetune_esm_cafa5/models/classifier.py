from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch import nn
import torchmetrics

from finetune_esm_cafa5.models.mlp import ClassifierConfig

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    pos_weight: torch.Tensor = None

class Classifier(LightningModule):
    def __init__(self, model_config: ClassifierConfig, training_config: TrainingConfig = TrainingConfig()):
        super().__init__()
        self.save_hyperparameters()

        self.model = model_config.get_model()
        self.model_config = model_config
        self.training_config = training_config

        self.pos_weight = training_config.pos_weight

        self.train_metrics = torchmetrics.MetricCollection({
            "f1_macro": torchmetrics.F1Score(task="multilabel", average="macro", num_labels=model_config.output_size),
            "f1_micro": torchmetrics.F1Score(task="multilabel", average="micro", num_labels=model_config.output_size),
        }, prefix="train/")

        self.val_metrics = self.train_metrics.clone(prefix="valid/")

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.training_config.lr)
        return optimizer

    def _common_step(self, batch, batch_idx, stage="train"):
        x = batch["embedding"]
        y = batch["labels"]

        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return y_hat, y, loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y_hat, y, loss = self._common_step(batch, batch_idx, "train")

        self.train_metrics(y_hat, y)
        self.log(f"train/loss", loss, on_step=True, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y_hat, y, loss = self._common_step(batch, batch_idx, "valid")

        self.val_metrics(y_hat, y)
        self.log(f"valid/loss", loss, on_step=False, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss
