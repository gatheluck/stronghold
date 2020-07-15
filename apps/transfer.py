import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import hydra
import omegaconf
import logging

from libs.trainutil import lightning_train
from libs.utils import UNFREEZE_PARAMS
from libs.io import load_model
from libs.utils import replace_final_fc
from libs.utils import freeze_params
from submodules.ModelBuilder.model_builder import ModelBuilder


@hydra.main(config_path='../conf/transfer.yaml')
def main(cfg: omegaconf.DictConfig) -> None:
    assert cfg.weight, 'please specify [weight] option.'
    assert (cfg.unfreeze_params is not None) or (cfg.unfreeze_level is not None), 'please specify either [unfreeze_params] or [unfreeze_level]'

    # fix relative path because hydra automatically change the current working dirctory.
    if not cfg.weight.startswith('/'):
        cfg.weight = os.path.join(hydra.utils.get_original_cwd(), cfg.weight)

    # get keys of nufreeze params. if you want specify more detail, please use [unfreeze_params] option directry
    if not cfg.unfreeze_params:
        cfg.unfreeze_params = UNFREEZE_PARAMS[cfg.arch][cfg.unfreeze_level]

    if cfg.hydra_id is None:
        cfg.hydra_id = os.path.basename(os.getcwd())

    # build model and train
    model = ModelBuilder(num_classes=cfg.source_num_classes, pretrained=False)[cfg.arch]
    load_model(model, cfg.weight)  # load model weight
    replace_final_fc(cfg.arch, model, cfg.dataset.num_classes)  # replace fc
    freeze_params(model, cfg.unfreeze_params)  # freeze some params
    lightning_train(model, cfg)

    logging.info('[transfer.py] is successfully ended.')


if __name__ == '__main__':
    main()
