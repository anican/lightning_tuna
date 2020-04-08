from abc import ABC, abstractmethod
from argparse import Namespace
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms


class LightningBase(pl.LightningModule, ABC):
    @abstractmethod
    def __init__(self, hparams: Namespace, paths):
        super(LightningBase, self).__init__()
        self.hparams = hparams
        self.paths = paths

    def prepare_data(self):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset_path = str(self.paths.DATASET_PATH)
        cifar_train = datasets.CIFAR10(root=dataset_path, train=True,
                download=True, transform=transform_train)
        print(len(cifar_train))
        partition = [45000, 5000]
        self.cifar_train, self.cifar_val = random_split(cifar_train, partition)
        self.cifar_test = datasets.CIFAR10(root=dataset_path, train=False,
                download=False, transform=transform_test)

    def training_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        train_loss = F.cross_entropy(logits, label)
        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        val_loss = F.cross_entropy(logits, label)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        data, label = batch
        logits = self.forward(data)
        test_loss = F.cross_entropy(logits, label)
        return {'test_loss': test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return DataLoader(self.cifar_train, self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, self.hparams.test_batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, self.hparams.test_batch_size)

    def configure_optimizers(self):
        lr = self.hparams.lr
        momentum = self.hparams.momentum
        weight_decay = self.hparams.weight_decay

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum,
                weight_decay=weight_decay)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer
