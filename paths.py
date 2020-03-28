from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    """This class is used to keep track of all the relevant directories for
    cleanly managing a deep learning exeeriment.

    Attributes:
        BASE_PATH: Path, base directory for all project activities.
        DATA_PATH: Path, directory for storing model checkpoints, and logging
            information.
        DATASET_PATH: Path, directory for storing all datasets related to
            project experiments.
        IMAGES_PATH: Path, directory for storing images from the experiments.

    """
    BASE_PATH: Path = Path(__file__).parents[0]
    DATA_PATH: Path = BASE_PATH / 'data'
    DATASET_PATH: Path = BASE_PATH / 'dataset'

    def __post_init__(self):
        self.DATA_PATH.mkdir(exist_ok=True)
