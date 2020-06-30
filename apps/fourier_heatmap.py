import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import copy
import hydra
import logging
import omegaconf
import pytorch_lightning

from libs.utils import check_required_keys
from libs.io import load_model
from libs.io import log_hyperparams

from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
from submodules.ModelBuilder.model_builder import ModelBuilder
from submodules.FourierHeatmap.fhmap.fourier_heatmap import create_fourier_heatmap


@hydra.main(config_path='../conf/fourier_heatmap.yaml', strict=False)
def main(cfg: omegaconf.DictConfig) -> None:
    # check config
    required_keys = 'weight'.split()
    check_required_keys(required_keys, cfg)

    # fix weight bacause hydra change the current working dir
    cfg.weight = os.path.join(hydra.utils.get_original_cwd(), cfg.weight)

    hydra_logger = logging.getLogger(__name__)
    hydra_logger.info(' '.join(sys.argv))
    hydra_logger.info(cfg.pretty())

    loggers = [
        pytorch_lightning.loggers.mlflow.MLFlowLogger(
            experiment_name='mlflow_output',
            tags=None)
    ]
    api_key = os.environ.get('ONLINE_LOGGER_API_KEY')
    if api_key and cfg.online_logger.activate:
        loggers.append(
            pytorch_lightning.loggers.CometLogger(api_key=api_key)
        )

    # log hyperparams
    for logger in loggers:
        log_hyperparams(logger, cfg)

    # build model
    model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[cfg.arch].cuda()
    if 'weight' in cfg:
        hydra_logger.info('loading weight from {weight}'.format(weight=cfg.weight))
        load_model(model, cfg.weight)  # load model weight

    # prepare dataset builder
    dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)

    create_fourier_heatmap(model, dataset_builder, log_dir='.', **cfg)


if __name__ == '__main__':
    main()
