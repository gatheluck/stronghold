import os
import sys

import hydra
import logging
import omegaconf

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

from libs.trainutil import lightning_train
from submodules.ModelBuilder.model_builder import ModelBuilder


@hydra.main(config_path="../conf/train.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    exp_id = os.environ.get('EXP_ID')  # get exp_id

    if cfg.resume_ckpt_path is not None:  # fix relative path because hydra automatically change the current working dirctory.
        cfg.resume_ckpt_path = (
            os.path.join(hydra.utils.get_original_cwd(), cfg.resume_ckpt_path)
            if not cfg.resume_ckpt_path.startswith("/")
            else cfg.resume_ckpt_path  # absolute path is not affected by hydra.
        )
    elif exp_id:  # set 'cfg.savedir'
        cfg.savedir = os.path.join(hydra.utils.get_original_cwd(), '../logs/{exp_id}/train'.format(exp_id=exp_id))
        os.makedirs(cfg.savedir, exist_ok=True)

    if cfg.hydra_id is None:
        cfg.hydra_id = os.path.basename(os.getcwd())

    # build model and train
    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[
        cfg.arch
    ]
    lightning_train(model, cfg)

    logging.info('[train.py] is successfully ended.')


if __name__ == "__main__":
    main()
