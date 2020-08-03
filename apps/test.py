import collections
import logging
import math
import os
import sys

import glob
import hydra
import omegaconf
import torch
import torchvision
import comet_ml
import tqdm

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base)

from libs.io import load_model
from libs.logger import Logger
from libs.metric import accuracy
from libs.utils import cfg_to_tags
from submodules.AttackBuilder.attack_builder import AttackBuilder
from submodules.AttackBuilder.attacks.utils import Denormalizer
from submodules.CnnSpacialSensitivity.spatial_sensitivity.patch_shuffle import (
    eval_patch_shuffle,
)
from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
from submodules.DatasetBuilder.libs.eval import evaluate_corruption_accuracy
from submodules.FeatureVisualizer.libs.first_layer import save_first_layer_weight
from submodules.FeatureVisualizer.libs.sensitivity_map import visualize_sensitivity_map
from submodules.FourierHeatmap.fhmap.fourier_heatmap import create_fourier_heatmap
from submodules.ModelBuilder.model_builder import ModelBuilder


SUPPORTED_TESTER = "acc fourier spacial corruption sensitivity layer".split()
# - acc: test standard and robust acc
# - fourier: generate fourier heatmap
# - spacial: test spacial sensitity
# - corruption: evaluate corruption accuracy


def eval_accuracy(model, hydra_logger, cfg, online_logger=None):
    """
    evaluate satandard and robust accuracy of the model
    """
    # build
    dataset_builder = DatasetBuilder(
        root_path=os.path.join(hydra.utils.get_original_cwd(), "../data"), **cfg.dataset
    )
    val_dataset = dataset_builder(train=False, normalize=cfg.normalize)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    denormalizer = Denormalizer(
        cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, to_pixel_space=False
    )

    metrics = dict()
    metrics["stdacc1"] = list()
    metrics["stdacc5"] = list()
    metrics["advacc1"] = list()
    metrics["advacc5"] = list()

    with tqdm.tqdm(enumerate(val_loader)) as pbar:
        for i, (x, y) in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)

            attacker = AttackBuilder("pgd")(
                input_size=cfg.dataset.input_size,
                mean=cfg.dataset.mean,
                std=cfg.dataset.std,
                **cfg.attack
            )
            x_adv = attacker(
                model, x, target=y, avoid_target=True, scale_eps=cfg.attack.scale_eps
            )
            with torch.autograd.no_grad():
                y_predict_std = model(x)
                y_predict_adv = model(x_adv)

            stdacc1, stdacc5 = accuracy(y_predict_std, y, topk=(1, 5))
            advacc1, advacc5 = accuracy(y_predict_adv, y, topk=(1, 5))

            metrics["stdacc1"].append(stdacc1.item())
            metrics["stdacc5"].append(stdacc5.item())
            metrics["advacc1"].append(advacc1.item())
            metrics["advacc5"].append(advacc5.item())

            pbar.set_postfix(
                collections.OrderedDict(
                    std="{}".format(stdacc1), adv="{}".format(advacc1)
                )
            )

            # save first 8 samples
            if i == 0:
                x_for_save = torch.cat(
                    [denormalizer(x[0:8, :, :, :]), denormalizer(x_adv[0:8, :, :, :])],
                    dim=2,
                )
                torchvision.utils.save_image(x_for_save, os.path.join(cfg.savedir, "pgd_test.png"))

    # take average over metrics
    for k, v_list in metrics.items():
        metrics[k] = sum(v_list) / float(len(v_list))

    # local logger for csv file
    local_logger = Logger(os.path.join(cfg.savedir, cfg.logger_path), mode="test")
    local_logger.log(metrics)

    # hydra logger
    for k, v in metrics.items():
        hydra_logger.info("{k}: {v}".format(k=k, v=v))


def eval_corruption_accuracy(model, cfg, online_logger=None):
    """
    evaluate corruption accuracy.
    currntly only valid for cifar10c.
    """
    assert (
        cfg.dataset.name in "cifar10c".split()
    ), "this dataset is not supported to evaluate corruption."

    dataset_builder = DatasetBuilder(
        root_path=os.path.join(hydra.utils.get_original_cwd(), "../data"), **cfg.dataset
    )
    evaluate_corruption_accuracy(
        model, dataset_builder, log_dir=cfg.savedir, corruptions=cfg.dataset.corruptions, **cfg
    )


def eval_fourier_heatmap(model, cfg, online_logger=None):
    """
    create fourier heatmap.
    main algorithm is written in https://github.com/gatheluck/FourierHeatmap
    """
    dataset_builder = DatasetBuilder(
        root_path=os.path.join(hydra.utils.get_original_cwd(), "../data"), **cfg.dataset
    )
    create_fourier_heatmap(model, dataset_builder, norm_type='linf', log_dir=cfg.savedir, **cfg.tester)


def eval_spacial_sensitivity(model, cfg, online_logger=None):
    """
    eval spacial sensitivity.
    main algorithm is written in https://github.com/gatheluck/CnnSpacialSensitivity
    """
    dataset_builder = DatasetBuilder(
        root_path=os.path.join(hydra.utils.get_original_cwd(), "../data"), **cfg.dataset
    )
    eval_patch_shuffle(model, dataset_builder, log_dir=cfg.savedir, **cfg.tester)


def visualize_sensitity_map(model, cfg, online_logger=None):
    """
    visualize sensitivity map
    main algorithm is written in https://github.com/gatheluck/FeatureVisualizer
    """
    dataset_builder = DatasetBuilder(
        root_path=os.path.join(hydra.utils.get_original_cwd(), "../data"), **cfg.dataset
    )
    visualize_sensitivity_map(
        model,
        dataset_builder,
        num_samples=cfg.tester.num_samples,
        num_classes=cfg.dataset.num_classes,
        ratio=cfg.tester.ratio,
        method=cfg.tester.method,
        log_dir=cfg.savedir,
        device=cfg.device
    )


def visualize_first_layer_weight(model, cfg, online_logger=None):
    """
    visualize first layer weight.
    main algrithm is written in https://github.com/gatheluck/FeatureVisualizer
    """
    save_first_layer_weight(model, log_path=os.path.join(cfg.savedir, "./first_layer_weight.png"), bias=cfg.tester.bias, **cfg)


@hydra.main(config_path="../conf/test.yaml")
def main(cfg: omegaconf.DictConfig):
    exp_id = os.environ.get('EXP_ID')  # get exp_id

    assert cfg.weight or exp_id, 'please specify [weight] option or environmental variable [EXP_ID].'
    assert (
        cfg.tester.name in SUPPORTED_TESTER
    ), "specified [tester] option is not suppored type."

    # replace cfg.savedir
    if exp_id:
        cfg.savedir = os.path.join(hydra.utils.get_original_cwd(), '../logs/ids/{exp_id}/test/{tester}'.format(exp_id=exp_id, tester=cfg.tester.name))
        os.makedirs(cfg.savedir, exist_ok=True)

    # fix weight bacause hydra change the current working dir
    if cfg.weight:
        if not cfg.weight.startswith('/'):
            cfg.weight = os.path.join(hydra.utils.get_original_cwd(), cfg.weight)
    else:
        targetpath = os.path.join(hydra.utils.get_original_cwd(), '../logs/ids/{exp_id}/**/model_weight_final.pth'.format(exp_id=exp_id))
        candidates = glob.glob(targetpath, recursive=True)
        if len(candidates) != 1:
            raise ValueError('correct weight file not found')
        else:
            cfg.weight = candidates[0]

    # logging hyper parameter
    hydra_logger = logging.getLogger(__name__)
    hydra_logger.info(" ".join(sys.argv))
    hydra_logger.info(cfg.pretty())

    # online logger
    # api_key = os.environ.get('ONLINE_LOGGER_API_KEY')
    # if api_key and cfg.online_logger.activate:
    #     online_logger = comet_ml.Experiment(api_key=api_key, project_name=cfg.project_name)
    #     online_logger.add_tags(cfg_to_tags(cfg))
    #     online_logger.log_parameters(omegaconf.OmegaConf.to_container(cfg))

    cfg.attack.step_size = eval(cfg.attack.step_size)  # eval actual value of step_size

    if cfg.hydra_id is None:
        cfg.hydra_id = os.path.basename(os.getcwd())

    # dump omegaconf
    omegaconf.OmegaConf.save(cfg, os.path.join(cfg.savedir, 'config.yaml'))

    # build model
    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[
        cfg.arch
    ]
    try:
        hydra_logger.info("loading weight from [{weight}]".format(weight=cfg.weight))
        load_model(model, cfg.weight)  # load model weight
        model = model.to(cfg.device)
        model.eval()
    except ValueError:
        hydra_logger.info("can not load weight from {weight}".format(weight=cfg.weight))
        raise ValueError("model loading is failed")

    # eval
    if cfg.tester.name == "acc":
        eval_accuracy(model, hydra_logger, cfg)
    elif cfg.tester.name == "corruption":
        eval_corruption_accuracy(model, cfg)
    elif cfg.tester.name == "fourier":
        eval_fourier_heatmap(model, cfg)
    elif cfg.tester.name == "spacial":
        eval_spacial_sensitivity(model, cfg)
    elif cfg.tester.name == "sensitivity":
        visualize_sensitity_map(model, cfg)
    elif cfg.tester.name == "layer":
        visualize_first_layer_weight(model, cfg)
    else:
        raise NotImplementedError

    logging.info('[test.py] is successfully ended.')


if __name__ == "__main__":
    main()
