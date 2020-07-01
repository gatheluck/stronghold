import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import math
import tqdm
import logging
import hydra
import omegaconf
import collections

import torch
import torchvision

from libs.io import load_model
from libs.metric import accuracy
from libs.logger import Logger

from submodules.ModelBuilder.model_builder import ModelBuilder
from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
from submodules.AttackBuilder.attack_builder import AttackBuilder
from submodules.AttackBuilder.attacks.utils import Denormalizer

SUPPORTED_MODE = 'acc fourier spacial'.split()
# - acc: test standard and robust acc
# - fourier: generate fourier heatmap
# - spacial: test spacial sensitity


def eval_accuracy(model, hydra_logger, cfg):
    """
    evaluate satandard and robust accuracy of the model
    """
    # build
    dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)
    val_dataset = dataset_builder(train=False, normalize=cfg.normalize)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False)

    denormalizer = Denormalizer(cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, to_pixel_space=False)

    metrics = dict()
    metrics['stdacc1'] = list()
    metrics['stdacc5'] = list()
    metrics['advacc1'] = list()
    metrics['advacc5'] = list()

    with tqdm.tqdm(enumerate(val_loader)) as pbar:
        for i, (x, y) in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)

            attacker = AttackBuilder('pgd')(input_size=cfg.dataset.input_size, mean=cfg.dataset.mean, std=cfg.dataset.std, **cfg.attack)
            x_adv = attacker(model, x, target=y, avoid_target=True, scale_eps=cfg.attack.scale_eps)
            with torch.autograd.no_grad():
                y_predict_std = model(x)
                y_predict_adv = model(x_adv)

            stdacc1, stdacc5 = accuracy(y_predict_std, y, topk=(1, 5))
            advacc1, advacc5 = accuracy(y_predict_adv, y, topk=(1, 5))

            metrics['stdacc1'].append(stdacc1.item())
            metrics['stdacc5'].append(stdacc5.item())
            metrics['advacc1'].append(advacc1.item())
            metrics['advacc5'].append(advacc5.item())

            pbar.set_postfix(collections.OrderedDict(std='{}'.format(stdacc1), adv='{}'.format(advacc1)))

            # save first 8 samples
            if i == 0:
                x_for_save = torch.cat([denormalizer(x[0:8, :, :, :]), denormalizer(x_adv[0:8, :, :, :])], dim=2)
                torchvision.utils.save_image(x_for_save, 'pgd_test.png')

    # take average over metrics
    for k, v_list in metrics.items():
        metrics[k] = sum(v_list) / float(len(v_list))

    # local logger for csv file
    local_logger = Logger(cfg.logger_path, mode='test')
    local_logger.log(metrics)

    # hydra logger
    for k, v in metrics.items():
        hydra_logger.info('{k}: {v}'.format(k=k, v=v))


@hydra.main(config_path='../conf/test.yaml')
def main(cfg: omegaconf.DictConfig):
    if cfg.mode not in SUPPORTED_MODE:
        raise ValueError

    # fix weight bacause hydra change the current working dir
    cfg.weight = os.path.join(hydra.utils.get_original_cwd(), cfg.weight)

    # logging hyper parameter
    hydra_logger = logging.getLogger(__name__)
    hydra_logger.info(' '.join(sys.argv))
    hydra_logger.info(cfg.pretty())
    cfg.attack.step_size = eval(cfg.attack.step_size)  # eval actual value of step_size

    # build model
    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[cfg.arch]
    try:
        hydra_logger.info('loading weight from {weight}'.format(weight=cfg.weight))
        load_model(model, cfg.weight)  # load model weight
        model = model.to(cfg.device)
        model.eval()
    except ValueError:
        hydra_logger.info('can not load weight from {weight}'.format(weight=cfg.weight))
        raise ValueError('model loading is failed')

    # eval
    if cfg.mode == 'acc':
        eval_accuracy(model, hydra_logger, cfg)
    elif cfg.mode == 'fourier':
        raise NotImplementedError
    elif cfg.mode == 'spacial':
        raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
