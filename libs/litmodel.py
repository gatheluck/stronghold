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
from libs.io import load_model
from libs.utils import parse_args
from libs.utils import get_epoch_end_log

from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
from submodules.ModelBuilder.model_builder import ModelBuilder
from submodules.PatchGaussian.patch_gaussian import AddPatchGaussian


class LitModel(pytorch_lightning.LightningModule):
    def __init__(self, model, cfg):
        """
        Args
        - 
        """
        super().__init__()

        # parse misc options
        required_keys = set('arch normalize batch_size epochs'.split())
        _parsed_args = parse_args(required_keys, cfg, strict=False)
        for k, v in _parsed_args.items():
            if k in required_keys:
                setattr(self, k, v)

        # build dataset and model
        self.dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)
        self.model = model

        # variables
        self.train_dataset = None
        self.val_dataset = None
        self.normalize = self.normalize
        self.dataset_root = os.path.join(hydra.utils.get_original_cwd(), '../data')
        self.log_path = '.'  # hydra automatically change the log place. for detail, please check 'conf/train.yaml'.
        self.cfg_optimizer = cfg.optimizer
        self.cfg_scheduler = cfg.scheduler
        self.cfg_augmentation = cfg.augmentation

        # initialize dir
        os.makedirs(self.dataset_root, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # optimizer part
        optimizer_name = self.cfg_optimizer.pop('name')
        if optimizer_name == 'sgd':
            optimizer_class = torch.optim.SGD
        else:
            raise ValueError

        optimizer = optimizer_class(self.model.parameters(), **self.cfg_optimizer)

        # scheduler part
        scheduler_name = self.cfg_scheduler.pop('name')
        if scheduler_name == 'steplr':
            scheduler_class = torch.optim.lr_scheduler.StepLR
        elif scheduler_name == 'multisteplr':
            scheduler_class = torch.optim.lr_scheduler.MultiStepLR
        else:
            raise ValueError

        scheduler = scheduler_class(optimizer, **self.cfg_scheduler)

        return [optimizer], [scheduler]

    def prepare_data(self):
        """
        download and prepare data. In distributed (GPU, TPU), this will only be called once this is called before requesting the dataloaders.
        """

        if self.cfg_augmentation.name == 'standard':
            train_optional_transform = []
        elif self.cfg_augmentation.name == 'patch_gaussian':
            train_optional_transform = [AddPatchGaussian(**self.cfg_augmentation)]
        else:
            raise NotImplementedError

        self.train_dataset = self.dataset_builder(train=True, normalize=self.normalize, optional_transform=train_optional_transform)

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