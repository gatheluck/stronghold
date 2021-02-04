"""moudle for 2D image datasets.

Todo:
    * Add more datasets (CIFAR-100, ImageNet etc.)

"""

import pathlib
from dataclasses import dataclass
from typing import Final, Tuple

import pytorch_lightning as pl
import torchvision
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DatasetStats:
    num_classes: int = MISSING
    input_size: int = MISSING
    mean: Tuple[float, float, float] = MISSING
    std: Tuple[float, float, float] = MISSING


@dataclass(frozen=True)
class Cifar10Stats(DatasetStats):
    num_classes: int = 10
    input_size: int = 32
    mean: Tuple[float, float, float] = (0.49139968, 0.48215841, 0.44653091)
    std: Tuple[float, float, float] = (0.24703223, 0.24348513, 0.26158784)


def get_transform(
    input_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    train: bool,
    normalize: bool = True,
) -> torchvision.transforms.transforms.Compose:
    """return composed tranforms for PyTorch 2D image dataset.

    Args:
        input_size (int): The size of input image.
        mean (Tuple[float]): The means of dataset.
        std (Tuple[float]): The standard diviation of dataset.
        train (bool): If True, data augmentations are composed.
        normalize (bool, optional): If True, normalization is composed. Defaults to True.

    Returns:
        torchvision.transforms.transforms.Compose: Composed transforms.

    """
    transform = list()

    # apply standard data augmentation
    if input_size == 32:
        if train:
            transform.extend(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                ]
            )
        else:
            pass
    elif input_size == 224:
        if train:
            transform.extend(
                [
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            transform.extend(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(224),
                ]
            )
    else:
        raise NotImplementedError

        transform.extend(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    # Convert to tensor
    transform.extend([torchvision.transforms.ToTensor()])

    # normalize
    if normalize:
        transform.extend([torchvision.transforms.Normalize(mean=mean, std=std)])

    return torchvision.transforms.Compose(transform)


class BaseDataModule(pl.LightningDataModule):
    """Base class for all 2d image LightningDataModule.

    A datamodule encapsulates the five steps involved in data processing in PyTorch:
    - Download / tokenize / process.
    - Clean and (maybe) save to disk.
    - Load inside Dataset.
    - Apply transforms (rotate, tokenize, etc…).
    - Wrap inside a DataLoader.
    For more detail, please check official docs: https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html#what-is-a-datamodule

    Attributes:
        batch_size (int): The size of input image.
        num_workers (int): The number of workers.
        dataset_stats (DatasetStats): The dataclass which holds dataset statistics.
        train_dataset: (Dataset): The dataset for training.
        val_dataset: (Dataset): The dataset for validation

    """

    def __init__(self, batch_size: int, num_workers: int, root: pathlib.Path) -> None:
        super().__init__()
        self.batch_size: Final[int] = batch_size
        self.num_workers: Final[int] = num_workers
        self.dataset_stats: DatasetStats
        self.train_dataset: Dataset
        self.val_dataset: Dataset

    def prepare_data(self, *args, **kwargs) -> None:
        """Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings."""
        raise NotImplementedError()

    def setup(self, stage=None) -> None:
        """There are also data operations you might want to perform on every GPU."""
        raise NotImplementedError()

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _get_transform(self, train: bool) -> torchvision.transforms.transforms.Compose:
        return get_transform(
            input_size=self.dataset_stats.input_size,
            mean=self.dataset_stats.mean,
            std=self.dataset_stats.std,
            train=train,
        )


class Cifar10DataModule(BaseDataModule):
    """The LightningDataModule for CIFAR-10 dataset.

    Attributes:
        dataset_stats (DatasetStats): The dataclass which holds dataset statistics.
        root (pathlib.Path): The root path which dataset exists. If not, try to download.

    """

    def __init__(self, batch_size: int, num_workers: int, root: pathlib.Path) -> None:
        super().__init__(batch_size, num_workers, root)
        self.dataset_stats: DatasetStats = Cifar10Stats()
        self.root: Final[pathlib.Path] = root / "cifar10"

    def prepare_data(self, *args, **kwargs) -> None:
        """Try to download dataset (DO NOT assign train/val here)."""
        self.root.mkdir(exist_ok=True, parents=True)
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=True)

    def setup(self, stage=None) -> None:
        """Assign dataset train and val """
        self.train_dataset: Dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self._get_transform(train=True),
        )
        self.val_dataset: Dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self._get_transform(train=False),
        )
