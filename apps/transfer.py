import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import copy
import hydra
import omegaconf
import torch
import pytorch_lightning

from libs.io import save_model
from libs.io import load_model
from libs.litmodel import LitModel
from libs.utils import check_required_keys
from libs.utils import replace_final_fc
from libs.utils import freeze_params
from libs.utils import UNFREEZE_PARAMS

from submodules.ModelBuilder.model_builder import ModelBuilder


@hydra.main(config_path='../conf/transfer.yaml', strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
    # check config
    required_keys = 'weight original_num_classes'.split()
    check_required_keys(required_keys, cfg)

    # fix weight bacause hydra change the current working dir
    cfg.weight = os.path.join(hydra.utils.get_original_cwd(), cfg.weight)
    cfg.unfreeze_params = UNFREEZE_PARAMS[cfg.arch][cfg.unfreeze_level]

    print(cfg.pretty())  # show config

    logger = pytorch_lightning.loggers.mlflow.MLFlowLogger(
        experiment_name='mlflow_output',
        tags=None
    )

    # log hyperparams
    _cfg = copy.deepcopy(cfg)
    for key, val in cfg.items():
        if type(val) is omegaconf.dictconfig.DictConfig:
            dict_for_log = {'.'.join([key, k]): v for k, v in val.items()}  # because cfg is nested dict, the nest info is added to keys.
            logger.log_hyperparams(dict_for_log)
            _cfg.pop(key)
    logger.log_hyperparams(dict(_cfg))

    # this function is called when saving checkpoint
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoint', '{epoch}-{val_loss_avg:.2f}'),
        monitor='val_loss_avg',
        save_top_k=1,
        verbose=True,
        mode='min',
        save_weights_only=False,
        prefix=cfg.prefix
    )

    trainer = pytorch_lightning.trainer.Trainer(
        deterministic=False,  # set True when you need reproductivity.
        benchmark=True,  # this will accerarate training.
        gpus=cfg.gpus,
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        default_save_path='.',
        weights_save_path='.'
    )

    # build model
    model = ModelBuilder(num_classes=cfg.original_num_classes, pretrained=False)[cfg.arch]
    if 'weight' in cfg:
        print('loading weight from {weight}'.format(weight=cfg.weight))
        load_model(model, cfg.weight)  # load model weight
    replace_final_fc(cfg.arch, model, cfg.dataset.num_classes)  # replace fc
    freeze_params(model, cfg.unfreeze_params)  # freeze some params

    # build lighting model
    litmodel = LitModel(model, cfg)

    # train for transfer
    trainer.fit(litmodel)
    save_model(litmodel.model, os.path.join(os.getcwd(), 'checkpoint', 'model_weight_transfered_final.pth'))  # manual backup of final model weight.
    # test
    trainer.test()


if __name__ == '__main__':
    main()