import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import hydra
import omegaconf
import torch
import torchvision
import pytorch_lightning

from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
from submodules.ModelBuilder.model_builder import ModelBuilder


class LitModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)
        # self.model = ModelBuilder(num_classes=cfg.dataset.num_class, pretrained=False)[]

    def forward(self, x):
        return 0


@hydra.main(config_path='../conf/train.yaml')
def main(cfg: omegaconf.DictConfig) -> None:
    print(cfg)

    logger = pytorch_lightning.loggers.mlflow.MLFlowLogger(
        experiment_name='mlflow_output',
        tags=None
    )
    trainer = pytorch_lightning.trainer.Trainer(
        gpus=1,
        max_epochs=cfg.train.epochs,
        min_epochs=cfg.train.epochs,
        logger=logger,
        default_save_path='.',
        weights_save_path='.'
    )

    model = LitModel(cfg)


if __name__ == '__main__':
    main()
