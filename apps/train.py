import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import hydra
import omegaconf
import itertools
import torch
import torchvision
import pytorch_lightning

from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
from submodules.ModelBuilder.model_builder import ModelBuilder


def parse_args(required_keys: set, input_args: dict) -> dict:
    """
    parse input args
    Args
    - required_keys (set) : set of required keys for input_args
    - input_args (dict)   : dict of input arugments
    """
    parsed_args = dict()

    for k in required_keys:
        if k not in input_args.keys():
            raise ValueError('initial args are invalid.')
        else:
            parsed_args[k] = input_args[k]

    return parsed_args


def get_epoch_end_log(outputs: list) -> dict:
    """
    form of outputs is List[Dict[str, Tensor]] or List[List[Dict[str, Tensor]]]
    """
    log = dict()

    # if list is nested, flatten them.
    if type(outputs[0]) is list:
        outputs = [x for x in itertools.chain(*outputs)]

    for key in outputs[0].keys():
        val = torch.stack([x[key] for x in outputs]).mean().cpu().item()
        log[key] = val

    return log


class LitModel(pytorch_lightning.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # parse misc options
        required_keys = set('arch normalize batch_size epochs'.split())
        _parsed_args = parse_args(required_keys, cfg)
        for k, v in _parsed_args.items():
            print('{}:{}'.format(k, v))
            setattr(self, k, v)

        # build
        self.dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)
        self.model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[self.arch]

        # variables
        self.train_dataset = None
        self.normalize = self.normalize
        self.dataset_root = os.path.join(hydra.utils.get_original_cwd(), '../data')
        self.log_path = '.'  # hydra automatically change the log place. for detail, please check 'conf/train.yaml'.
        self.cfg_optimizer = cfg.optimizer

        # initialize dir
        os.makedirs(self.dataset_root, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer_name = self.cfg_optimizer.pop('name')

        if optimizer_name == 'sgd':
            optimizer_class = torch.optim.SGD
        else:
            raise ValueError

        return optimizer_class(self.model.parameters(), **self.cfg_optimizer)

    def prepare_data(self):
        """
        download and prepare data. In distributed (GPU, TPU), this will only be called once this is called before requesting the dataloaders.
        """
        # train_optional_transform = torchvision.transforms.Compose([])
        self.train_dataset = self.dataset_builder(train=True, normalize=self.normalize)

        # val_optional_transform = torchvision.transforms.Compose([])
        self.val_dataset = self.dataset_builder(train=False, normalize=self.normalize)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        """
        return loss, dict with metrics for tqdm. this function must be overided.
        """
        x, y = batch
        y_predict = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_predict, y)
        self.logger.log_metrics(dict(loss=loss.detach().cpu().item()))
        return dict(loss=loss)

    def training_epoch_end(self, outputs):
        log_dict = get_epoch_end_log(outputs)
        log_dict['step'] = self.current_epoch

        return {'log': log_dict}


@hydra.main(config_path='../conf/train.yaml')
def main(cfg: omegaconf.DictConfig) -> None:
    print(cfg)

    logger = pytorch_lightning.loggers.mlflow.MLFlowLogger(
        experiment_name='mlflow_output',
        tags=None
    )
    trainer = pytorch_lightning.trainer.Trainer(
        gpus=1,
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=logger,
        default_save_path='.',
        weights_save_path='.'
    )

    model = LitModel(cfg)
    trainer.fit(model)


if __name__ == '__main__':
    main()
