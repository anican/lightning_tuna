from pytorch_lightning import Callback


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("\nModel training has begun...")

    def on_train_end(self, trainer, pl_module):
        print("\nModel training has finished...")

    def on_validation_start(self, trainer, pl_module):
        print("\nModel validation has begun...")

    def on_validation_end(self, trainer, pl_module):
        print("\nModel validation has finished...")

