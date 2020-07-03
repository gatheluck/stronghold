import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import hydra
import omegaconf

from libs.trainutil import lightning_train
from submodules.ModelBuilder.model_builder import ModelBuilder


@hydra.main(config_path='../conf/train.yaml')
def main(cfg: omegaconf.DictConfig) -> None:
    # fix relative path because hydra automatically change the current working dirctory.
    if (cfg.resume_ckpt_path) and (not cfg.resume_ckpt_path.startswith('/')):
        cfg.resume_ckpt_path = os.path.join(hydra.utils.get_original_cwd(), cfg.resume_ckpt_path)

    # build model and train
    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[cfg.arch]
    lightning_train(model, cfg)


if __name__ == '__main__':
    main()
