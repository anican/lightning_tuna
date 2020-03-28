from pytorch_lightning import Callback


class PrintCallback(Callback):
    def on_train_start(self):
        print("Model training has begun...")

    def on_train_end(self):
        print(f"Training has finished...\nThe logs are: {self.trainer.logs}")

    def on_validation_start(self):
        print("Model validation has begun...")

    def on validation_end(self):
        print("Model validation has finished...")

