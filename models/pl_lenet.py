import argparse
from argparse import Namespace
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.multiprocessing import cpu_count
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms


class LeNet(pl.LightningModule):
    """
    LeNet was the first convolutional neural networks (CNN) model designed for
    image recognition on the MNIST database. This particular architecture was
    first introduced in the 1990's by Yann LeCun at New York University.

    This particular model is adjusted for use on the CIFAR10 database.
    """
    def __init__(self, hparams: Namespace, paths, num_classes: int = 10):
        super(LeNet, self).__init__()
        self.hparams = hparams
        self.paths = paths
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs):
        features = self.layers(inputs)
        features = features.view(features.size(0), -1)
        outputs = self.classifier(features)
        return outputs

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
        return DataLoader(self.cifar_train, self.hparams.batch_size,
                num_workers=cpu_count())

    def val_dataloader(self):
        return DataLoader(self.cifar_val, self.hparams.test_batch_size,
                num_workers=cpu_count())

    def test_dataloader(self):
        return DataLoader(self.cifar_test, self.hparams.test_batch_size,
                num_workers=cpu_count())

    def configure_optimizers(self):
        lr = self.hparams.lr
        momentum = self.hparams.momentum
        weight_decay = self.hparams.weight_decay

        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum,
                weight_decay=weight_decay)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=1)
        return optimizer

def _test():
    parser = argparse.ArgumentParser(prog='lightning_tuna')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    args = parser.parse_args()

    inputs = torch.randn(50, 3, 32, 32)
    print("inputs shape", inputs.shape)
    model = LeNet(args, paths=None)
    outputs = model(inputs)
    print(outputs.shape)

if __name__ == '__main__':
    _test()

