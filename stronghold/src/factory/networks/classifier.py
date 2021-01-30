import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate

import stronghold.src.schema as schema


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        optimizer_cfg: schema.OptimizerConfig,
        scheduler_cfg: schema.SchedulerConfig,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.encoder(x)

    def configure_optimizers(self):
        # instantiate optimizer.
        try:
            optimizer = instantiate(
                self.optimizer_cfg.instance, params=self.encoder.parameters()
            )
        except Exception as e:
            print(f"unknown error: {e}")

        # instantiate scheduler.
        try:
            scheduler = instantiate(self.scheduler_cfg.instance)
        except Exception as e:
            print(f"unknown error: {e}")

        print(optimizer)
        return [optimizer], [scheduler]
