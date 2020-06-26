import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import hydra
import omegaconf

from libs.utils import check_required_keys
from libs.io import load_model

from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
from submodules.ModelBuilder.model_builder import ModelBuilder
from submodules.FourierHeatmap.fhmap.fourier_heatmap import create_fourier_heatmap


@hydra.main(config_path='../conf/fourier_heatmap.yaml', strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
    print(cfg.pretty())

    # check config
    required_keys = 'weight'.split()
    check_required_keys(required_keys, cfg)
    print(cfg.pretty())

    # fix weight bacause hydra change the current working dir
    cfg.weight = os.path.join(hydra.utils.get_original_cwd(), cfg.weight)

    # build model
    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[cfg.arch].cuda()
    if 'weight' in cfg:
        print('loading weight from {weight}'.format(weight=cfg.weight))
        load_model(model, cfg.weight)  # load model weight

    # prepare dataset builder
    dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)

    create_fourier_heatmap(model, dataset_builder, log_dir='.', **cfg)


if __name__ == '__main__':
    main()
