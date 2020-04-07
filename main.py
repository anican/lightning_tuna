#!/usr/bin/env python
import argparse
from callbacks import PrintCallback
from models import LeNet
from paths import Paths
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.cuda import device_count as gpu_count


def parse_args():
    parser = argparse.ArgumentParser(prog='lightning_tuna',
            description='<Description of Experiment>')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--levy_alpha', type=float, default=-1.0,
            help='tail index of added levy motion')
    parser.add_argument('--levy_sigma', type=float, default=-1.0,
            help='scale parameter of added levy noise')
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--neurons', type=int, default=1024,
            help='number of neurons in hidden layer')
    parser.add_argument('--log_save_interval', type=int, default=100)
    parser.add_argument('--criterion', type=str, default="cross_entropy",
            help='cross_entropy or linear_hinge')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--architecture', type=str, default='mlp')
    return parser.parse_args()


def main():
    args = parse_args()

    paths = Paths()
    checkpoints_path = str(paths.CHECKPOINTS_PATH)
    logging_path = str(paths.LOG_PATH)

    callbacks = [PrintCallback()]
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoints_path + '/{epoch}-{val_acc:.3f}', save_top_k=True,
        verbose=True, monitor='val_acc', mode='max', prefix=''
    )
    early_stop_callback = EarlyStopping(
        monitor='val_acc', mode='max', verbose=False, strict=False,
        min_delta=0.0, patience=2
    )
    gpus = gpu_count()
    log_save_interval = args.log_save_interval
    logger = TensorBoardLogger(save_dir=logging_path, name='tuna-log')
    max_epochs = args.epochs

    model = LeNet(hparams=args, paths=paths)
    trainer = Trainer(
        callbacks=callbacks, checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback, fast_dev_run=True, gpus=gpus,
        log_save_interval=log_save_interval, logger=logger,
        max_epochs=max_epochs, min_epochs=1, show_progress_bar=True,
        weights_summary='full',
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()

