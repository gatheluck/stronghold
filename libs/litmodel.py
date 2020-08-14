import os
import sys
import math

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import shutil
import hydra
import omegaconf
from omegaconf import OmegaConf
import itertools
import logging
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
from submodules.FourierHeatmap.fhmap.fourier_basis_augmented_dataset import FourierBasisAugmentedDataset


class LitCallback(pytorch_lightning.callbacks.Callback):
    """
    Callback class used in [pytorch_lightning.trainer.Trainer] class.
    For detail, please check following docs:
    - https://pytorch-lightning.readthedocs.io/en/stable/callbacks.html
    - https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#callbacks
    """
    def on_train_start(self, trainer, pl_module):
        logging.info('Training is started!')

    def on_train_end(self, trainer, pl_module):
        logging.info('Training is successfully ended!')

        # save state dict to local.
        local_save_path = os.path.join(trainer.weights_save_path, 'model_weight_final.pth')
        save_model(trainer.model.module.model, local_save_path)  # trainer.model.module.model is model in LitModel class
        logging.info('Trained model is successfully saved to [{path}]'.format(path=local_save_path))

        # copy log info to 'local_save_path'
        # if trainer.default_root_dir != '.':
        #     shutil.copytree('.', os.path.join(trainer.default_root_dir, 'log'))

        # logging to online logger
        for logger in trainer.logger:
            if isinstance(logger, pytorch_lightning.loggers.comet.CometLogger):
                # log local log to comet: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_asset_folder
                logger.experiment.log_asset_folder(trainer.default_root_dir, log_file_name=True, recursive=True)

                # log model to comet: https://www.comet.ml/docs/python-sdk/Experiment/#experimentlog_model
                if trainer.model is None:
                    logging.info('There is no model to log because [trainer.model] is None.')
                    pass
                else:
                    # model_state_dict = trainer.model.module.state_dict() if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model.state_dict()
                    logger.experiment.log_model('checkpoint', local_save_path)
                    logging.info('Trained model is successfully saved to comet as state dict.')


class LitModel(pytorch_lightning.LightningModule):
    def __init__(self, model, cfg):
        """
        Args
        - 
        """
        super().__init__()

        # this attribute is for saving hyper harams with model weight by pytorch lightning
        self.hparams = omegaconf.OmegaConf.to_container(cfg)  # convert DictConfig to dict. please check following page: https://omegaconf.readthedocs.io/en/latest/usage.html

        # parse misc options
        required_keys = set('arch normalize batch_size epochs num_workers'.split())
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
        self.cfg_dataset = cfg.dataset
        self.cfg_optimizer = cfg.optimizer
        self.cfg_scheduler = cfg.scheduler
        self.cfg_augmentation = cfg.augmentation

        # initialize dir
        os.makedirs(self.dataset_root, exist_ok=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # optimizer part
        _cfg_optimizer = OmegaConf.to_container(self.cfg_optimizer)
        optimizer_name = _cfg_optimizer.pop('name')

        if optimizer_name == 'sgd':
            optimizer_class = torch.optim.SGD
        else:
            raise ValueError

        optimizer = optimizer_class(self.model.parameters(), **_cfg_optimizer)

        # scheduler part
        _cfg_scheduler = OmegaConf.to_container(self.cfg_scheduler)
        scheduler_name = _cfg_scheduler.pop('name')

        if scheduler_name == 'steplr':
            scheduler_class = torch.optim.lr_scheduler.StepLR
        elif scheduler_name == 'multisteplr':
            scheduler_class = torch.optim.lr_scheduler.MultiStepLR
        else:
            raise ValueError

        scheduler = scheduler_class(optimizer, **_cfg_scheduler)

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
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

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
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

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
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

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


class FourierBasisAugmentedLitModel(LitModel):
    """
    Wrapper class for Lightning Model. Add Fourier basis to dataset at runtime.
    """
    def __init__(self, model, cfg, **kwargs):
        super().__init__(model, cfg)

        assert 0.0 <= kwargs['weight_fba'] <= 1.0
        assert kwargs['dim_rep'] > 0
        assert kwargs['dim_fba'] > 0
        assert kwargs['eps_fba'] > 0

        self.weight_fba = kwargs['weight_fba']
        self.fc_fba = torch.nn.Linear(kwargs['dim_rep'], kwargs['dim_fba'])
        self.eps_fba = kwargs['eps_fba']

        torch.nn.init.kaiming_normal(self.fc_fba.weight)  # init weight

    def prepare_data(self):
        if self.cfg_augmentation.name == 'standard':
            pass
        elif self.cfg_augmentation.name == 'patch_gaussian':
            raise ValueError('Fourier basis augmentation should not be used with Patch Gaussian')
        else:
            raise NotImplementedError

        train_dataset = self.dataset_builder(train=True, normalize=False)
        self.train_dataset = FourierBasisAugmentedDataset(train_dataset,
                                                          input_size=self.cfg_dataset.input_size,
                                                          mean=self.cfg_dataset.mean,
                                                          std=self.cfg_dataset.std,
                                                          h_index=-(math.ceil(self.cfg_dataset.input_size / 2.0) - 1),
                                                          w_index=-(math.ceil(self.cfg_dataset.input_size / 2.0) - 1),
                                                          eps=self.eps_fba,
                                                          randomize_index=True,
                                                          normalize=self.normalize,
                                                          mode='index')

        val_dataset = self.dataset_builder(train=False, normalize=False)
        self.val_dataset = FourierBasisAugmentedDataset(val_dataset,
                                                        input_size=self.cfg_dataset.input_size,
                                                        mean=self.cfg_dataset.mean,
                                                        std=self.cfg_dataset.std,
                                                        h_index=-(math.ceil(self.cfg_dataset.input_size / 2.0) - 1),
                                                        w_index=-(math.ceil(self.cfg_dataset.input_size / 2.0) - 1),
                                                        eps=self.eps_fba,
                                                        randomize_index=True,
                                                        normalize=self.normalize,
                                                        mode='index')

    def forward(self, x):
        return self.model(x, return_rep=True)

    def training_step(self, batch, batch_idx):
        """
        return loss, dict with metrics for tqdm. this function must be overided.
        """
        x, y, y_fba = batch

        # predict class label
        y_predict, rep = self.forward(x)
        loss_cls = torch.nn.functional.cross_entropy(y_predict, y) * (1.0 - self.weight_fba)

        # predict fourier basis index
        y_predict_fba = self.fc_fba(rep)
        loss_fba = torch.nn.functional.cross_entropy(y_predict_fba, y_fba) * self.weight_fba

        loss = loss_cls + loss_fba

        stdacc1_cls, stdacc5_cls = accuracy(y_predict, y, topk=(1, 5))
        stdacc1_fba, stdacc5_fba = accuracy(y_predict_fba, y_fba, topk=(1, 5))

        log = {'train_loss': loss,
               'train_loss_cls': loss_cls,
               'train_loss_fba': loss_fba,
               'train_std_acc1': stdacc1_cls,
               'train_std_acc5': stdacc5_cls,
               'train_std_acc1_fba': stdacc1_fba,
               'train_std_acc5_fba': stdacc5_fba}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        """
        return loss, dict with metrics for tqdm. this function must be overided.
        """
        x, y, y_fba = batch

        # predict class label
        y_predict, rep = self.forward(x)
        loss_cls = torch.nn.functional.cross_entropy(y_predict, y) * (1.0 - self.weight_fba)

        # predict fourier basis index
        y_predict_fba = self.fc_fba(rep)
        loss_fba = torch.nn.functional.cross_entropy(y_predict_fba, y_fba) * self.weight_fba

        loss = loss_cls + loss_fba

        stdacc1_cls, stdacc5_cls = accuracy(y_predict, y, topk=(1, 5))
        stdacc1_fba, stdacc5_fba = accuracy(y_predict_fba, y_fba, topk=(1, 5))

        log = {'val_loss': loss,
               'val_loss_cls': loss_cls,
               'val_loss_fba': loss_fba,
               'val_std_acc1': stdacc1_cls,
               'val_std_acc5': stdacc5_cls,
               'val_std_acc1_fba': stdacc1_fba,
               'val_std_acc5_fba': stdacc5_fba}
        return log

    def test_step(self, batch, batch_idx):
        """
        return loss, dict with metrics for tqdm. this function must be overided.
        """
        x, y, y_fba = batch

        # predict class label
        y_predict, rep = self.forward(x)
        loss_cls = torch.nn.functional.cross_entropy(y_predict, y) * (1.0 - self.weight_fba)

        # predict fourier basis index
        y_predict_fba = self.fc_fba(rep)
        loss_fba = torch.nn.functional.cross_entropy(y_predict_fba, y_fba) * self.weight_fba

        loss = loss_cls + loss_fba

        stdacc1_cls, stdacc5_cls = accuracy(y_predict, y, topk=(1, 5))
        stdacc1_fba, stdacc5_fba = accuracy(y_predict_fba, y_fba, topk=(1, 5))

        log = {'val_loss': loss,
               'val_loss_cls': loss_cls,
               'val_loss_fba': loss_fba,
               'val_std_acc1': stdacc1_cls,
               'val_std_acc5': stdacc5_cls,
               'val_std_acc1_fba': stdacc1_fba,
               'val_std_acc5_fba': stdacc5_fba}

        return log
