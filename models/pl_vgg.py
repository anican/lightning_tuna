import argparse
from argparse import Namespace
from pl_base import LightningBase
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
        512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
        'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
        512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(LightningBase):
    """
    Deep convolutional neural network from Oxford University. Achieved second
    place in the ImageNet competition 2014. Four varieties of VGG are provided.

    This particular model is adjusted for use on the CIFAR10 dataset.
    """
    def __init__(self, vgg_name, hparams, paths, batch_norm=True, num_classes=10):
        super(VGG, self).__init__(hparams, paths)
        self.features = self._make_layers(cfg[vgg_name],batch_norm)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, get_features=False):
        out = self.features(x)
        if get_features:
            out_flat = out.view(out.size(0), -1)
            out = self.classifier(out_flat)
            return out, out_flat
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                # nn.init.xavier_uniform_(layers[-1].weight.data) # self added
                # nn.init.xavier_uniform_(layers[-1].bias.data) # self added
                if batch_norm:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def _test():
    parser = argparse.ArgumentParser(prog='lightning_tuna')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    args = parser.parse_args()
    net = VGG(vgg_name='VGG11', hparams=args, paths=None)
    x = torch.randn(50,3,32,32)
    y = net(x)
    print(y.size())
    print(net)

if __name__ == '__main__':
    _test()

