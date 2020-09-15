import os
import logging
import re
import copy
import torch
import omegaconf
from collections import OrderedDict


def save_model(model, path):
    torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), path)


def load_model(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError('path "{path}" does not exist.'.format(path=path))
    logging.info('loading model weight from {path}'.format(path=path))

    # load weight from .pth file.
    if path.endswith('.pth'):
        weight = torch.load(path)
        statedict = OrderedDict([(re.sub('^module.', '', k), v) for k, v in weight.items()])
        # statedict = OrderedDict()
        # for k, v in torch.load(path).items():
        #     if k.startswith('model.'):
        #         k = '.'.join(k.split('.')[1:])

        #     statedict[k] = v

        # model.load_state_dict(torch.load(path))
        model.load_state_dict(statedict)
    # load weight from checkpoint.
    elif path.endswith('.ckpt'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise ValueError('this checkponint do not inculdes state_dict')
    else:
        raise ValueError('path is not supported type of extension.')


# def log_hyperparams(logger, cfg: omegaconf.DictConfig):
#     """
#     log hyperparameters from omegaconf.
#     because omegaconf is nested, this function is flatten them.
#     """
#     _cfg = copy.deepcopy(cfg)
#     for key, val in cfg.items():
#         if type(val) is omegaconf.dictconfig.DictConfig:
#             dict_for_log = {'.'.join([key, k]): v for k, v in val.items()}  # because cfg is nested dict, the nest info is added to keys.
#             logger.log_hyperparams(dict_for_log)
#             _cfg.pop(key)
#     logger.log_hyperparams(dict(_cfg))
#     del _cfg
