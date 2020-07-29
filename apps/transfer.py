import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import glob 
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
    exp_id = os.environ.get('EXP_ID')  # get exp_id

    assert cfg.weight or exp_id, 'please specify [weight] option or environmental variable [EXP_ID].'
    assert (cfg.unfreeze_params is not None) or (cfg.unfreeze_level is not None), 'please specify either [unfreeze_params] or [unfreeze_level]'

    # adjust cfg.weight and cgf.savepath
    if exp_id:  # set 'model_weight_final.pth' from 'exp_id'
        cfg.savedir = os.path.join(hydra.utils.get_original_cwd(), '../logs/ids/{exp_id}/transfer'.format(exp_id=exp_id))
        os.makedirs(cfg.savedir, exist_ok=True)
        if not cfg.weight:
            cfg.weight = os.path.join(hydra.utils.get_original_cwd(), '../logs/ids/{exp_id}/train/checkpoint/model_weight_final.pth'.format(exp_id=exp_id))

    if not cfg.weight.startswith('/'):  # fix relative path because hydra automatically change the current working dirctory.
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
