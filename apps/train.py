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

from libs.metric import accuracy
from libs.io import save_model

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
    this function fill the gap between single theread and data.parallel.
    form of outputs is List[Dict[str, Tensor]] or List[List[Dict[str, Tensor]]]
    """
    log = dict()

    # if list is nested, flatten them.
    if type(outputs[0]) is list:
        outputs = [x for x in itertools.chain(*outputs)]

    if 'log' in outputs[0].keys():
        print(outputs[0]['log'].keys())
        for key in outputs[0]['log'].keys():
            val = torch.stack([x['log'][key] for x in outputs]).mean().cpu().item()
            log[key + '_avg'] = val
    else:
        for key in outputs[0].keys():
            val = torch.stack([x[key] for x in outputs]).mean().cpu().item()
            log[key + '_avg'] = val

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

        # build dataset and model
        self.dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)
        self.model = ModelBuilder(num_classes=cfg.dataset.num_classes, pretrained=False)[self.arch]

        # variables
        self.train_dataset = None
        self.val_dataset = None
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

    # train related methods
    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        """
        return loss, dict with metrics for tqdm. this function must be overided.
        """
        x, y = batch
        y_predict = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_predict, y)
        stdacc1, stdacc5 = accuracy(y_predict, y, topk=(1, 5))

        log = {'train_loss': loss,
               'train_std_acc1': stdacc1,
               'train_std_acc5': stdacc5}
        return {'loss': loss, 'log': log}

    def training_epoch_end(self, outputs):
        log_dict = get_epoch_end_log(outputs)
        log_dict['step'] = self.current_epoch
        return {'log': log_dict}

    # validation related methods
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_predict = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_predict, y)
        stdacc1, stdacc5 = accuracy(y_predict, y, topk=(1, 5))

        log = {'val_loss': loss,
               'val_std_acc1': stdacc1,
               'val_std_acc5': stdacc5}
        return log

    def validation_epoch_end(self, outputs):
        log_dict = get_epoch_end_log(outputs)
        log_dict['step'] = self.current_epoch
        return {'log': log_dict}

    # test related methods
    def test_dataloader(self):
        # IMPORTANT: now just same as validataion dataset.
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_predict = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_predict, y)
        stdacc1, stdacc5 = accuracy(y_predict, y, topk=(1, 5))

        log = {'val_loss': loss,
               'val_std_acc1': stdacc1,
               'val_std_acc5': stdacc5}
        return log

    def test_epoch_end(self, outputs):
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

    # this function is called when saving checkpoint
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'checkpoint', '{epoch}-{val_loss_avg:.2f}'),
        monitor='val_loss_avg',
        save_top_k=1,
        verbose=True,
        mode='min',
        save_weights_only=True,
        prefix=cfg.prefix
    )

    trainer = pytorch_lightning.trainer.Trainer(
        deterministic=False,  # set True when you need reproductivity.
        benchmark=True,  # this will accerarate training.
        gpus=1,
        max_epochs=cfg.epochs,
        min_epochs=cfg.epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        default_save_path='.',
        weights_save_path='.'
    )

    litmodel = LitModel(cfg)
    # train
    trainer.fit(litmodel)
    save_model(litmodel.model, os.path.join(os.getcwd(), 'checkpoint', 'model_weight_final.pth'))
    # test
    trainer.test()


if __name__ == '__main__':
    main()
