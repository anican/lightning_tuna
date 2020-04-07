from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    """This class is used to keep track of all the relevant directories for
    cleanly managing a deep learning exeeriment.

    Attributes:
        BASE_PATH: Path, base directory for all project activities.
        CHECKPOINS_PATH: Path, directory for storing model checkpoints
        DATASET_PATH: Path, directory for storing all datasets related to
            project experiments.
        IMAGES_PATH: Path, directory for storing images from the experiments.
            This is the place to store any samples or pictures not related to
            logging info pertaining to train/val/test metrics.
        LOG_PATH: Path, directory for storing all logger information (Comet.ml,
            tensorboard, neptune, etc.)

    """
    BASE_PATH: Path = Path(__file__).parents[0]
    CHECKPOINTS_PATH: Path = BASE_PATH / 'checkpoints'
    DATASET_PATH: Path = BASE_PATH / 'dataset'
    LOG_PATH: Path = BASE_PATH / 'logging'
    IMAGES_PATH: Path = BASE_PATH / 'images'

    def __post_init__(self):
        self.CHECKPOINTS_PATH.mkdir(exist_ok=True)
        self.DATASET_PATH.mkdir(exist_ok=True)
        self.LOG_PATH.mkdir(exist_ok=True)
        self.IMAGES_PATH.mkdir(exist_ok=True)
