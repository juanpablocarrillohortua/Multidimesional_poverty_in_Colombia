"""
General settings of the project.
"""

from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        self.WORKDIR = Path(__file__).resolve().parent

        self.DATADIR = self.WORKDIR / "data"

        self.MODEL_DIR = self.WORKDIR / "models"

        self.EXTRACTED = self.DATADIR / "extracted"

config = Configuration()
