import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

import math
import glob
import shutil
import hydra
import omegaconf
import logging

from libs.io import load_model
from apps.test import eval_accuracy, eval_corruption_accuracy
from submodules.FourierHeatmap.fhmap.fourier_heatmap import create_fourier_heatmap
from submodules.ModelBuilder.model_builder import ModelBuilder
from submodules.DatasetBuilder.dataset_builder import DatasetBuilder


def eval_fourier_heatmap(model, cfg, online_logger=None, savedir=None):
    """
    create fourier heatmap.
    main algorithm is written in https://github.com/gatheluck/FourierHeatmap
    """
    # create savedir
    if savedir is None:
        savedir = cfg.savedir

    if os.path.exists(savedir):
        return None
    else:
        os.makedirs(savedir, exist_ok=True)

    dataset_builder = DatasetBuilder(
        root_path=os.path.join(hydra.utils.get_original_cwd(), "../data"), **cfg.dataset
    )
    create_fourier_heatmap(model, dataset_builder, norm_type='linf', log_dir=savedir, eps=4.0, **cfg)


def check_target_mode(targetpath, targetdir, mode_candidates: list = ['train', 'transfer']) -> str:
    """
    this function check weather target path is train or transfer or unknown.
    targetpath should be like, 'hoge/logs/train/fbdb_train_runtime_index_0815/2020-08-15_10-13-39_mono_fba_weight-1.0_eps-4.0_index/model_weight_final.pth'
    """

    # remove final separator and adjust targetpath
    targetdir = targetdir[:-1] if targetdir.endswith(os.sep) else targetdir
    targetdir = os.sep.join(targetdir.split(os.sep)[:-1])
    targetpath = targetpath.replace(targetdir, '')

    # check mode from targetpath
    modes = list()
    for dirname in targetpath.split(os.sep):
        modes.extend([mode for mode in mode_candidates if mode in dirname])

    # remove redanduncy
    modes = list(set(modes))
    if len(modes) == 1:
        mode = modes[0]
    else:
        mode = 'unknown'

    return mode, targetpath.lstrip(os.sep).split(os.sep)[1]


def find_needles(targetdir: str, needle='model_weight_final.pth') -> list:
    """
    find needle form targetdir.
    """
    return glob.glob(os.path.join(targetdir, '**', needle), recursive=True)

# def get 


@hydra.main(config_path="../conf/test_mul.yaml")
def main(cfg: omegaconf.DictConfig):
    if cfg.targetdir is None:
        raise FileNotFoundError('Please specify targetdir.')
    elif not cfg.targetdir.startswith('/'):
        raise FileNotFoundError('Please use abspath to specify targetdir.')

    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info(cfg.pretty())

    # find targets
    needles = find_needles(cfg.targetdir)
    if len(needles) == 0:
        logger.error('No targets are found.')
    else:
        logger.info('{} needles are found.'.format(len(needles)))

    # eval actual value of step_size
    cfg.attack.step_size = eval(cfg.attack.step_size)

    for needle in needles:
        logger.info('---------------------------')
        logger.info('run test: {}'.format(needle))

        # create output dir and copy
        mode, dirname = check_target_mode(needle, cfg.targetdir)
        test_savedir = os.path.join(dirname, 'test')
        os.makedirs(test_savedir, exist_ok=True)
        if cfg.copy_needle:
            shutil.copytree(os.path.join(cfg.targetdir, dirname), os.path.join(dirname, mode))

        # dump omegaconf for viewer
        omegaconf.OmegaConf.save(cfg, os.path.join(test_savedir, 'config.yaml'))

        # build and load model
        model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[cfg.arch]
        try:
            logger.info("loading weight from [{weight}]".format(weight=needle))
            load_model(model, needle)  # load model weight
            model = model.to(cfg.device)
            model.eval()
        except ValueError:
            logger.info("can not load weight from {weight}".format(weight=needle))
            raise ValueError("model loading is failed")

        # run test
        # eval_accuracy(model, logger, cfg, online_logger=None, savedir=os.path.join(test_savedir, 'acc'))
        eval_corruption_accuracy(model, cfg, savedir=os.path.join(test_savedir, 'rob'))
        # eval_fourier_heatmap(model, cfg, savedir=os.path.join(test_savedir, 'fourier'))


if __name__ == '__main__':
    main()
