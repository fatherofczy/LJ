"""
Spatio-temporal model
"""

__all__ = ["data", "model", "pipeline", "utils"]

import torch
import torch.nn as nn

from .data import MmapDataset, Preprocessor
from .pipeline import Pipeline
from .utils import Directory


class EncoderPipeline(Pipeline):
    """Pipeline for encoder model"""

    def calc_loss(self, data: tuple) -> torch.Tensor:
        return super().calc_loss(data)

    def prep_eval(self, data: tuple) -> torch.Tensor:
        return super().prep_eval(data)
