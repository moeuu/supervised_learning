from logging import getLogger

import torch.nn as nn

from .cutmix import cutmix
from .focal import FocalLoss
from .metric_learning import ArcFace, CosFace
from .mixup import mixup
from .cutblur import cutblur

__all__ = [
    "get_criterion",
    "ArcFace",
    "CosFace",
    "mixup",
    "cutmix",
    "cutblur",
    "MixupCrossEntropy",
]

logger = getLogger(__name__)

def get_criterion(name="cross_entropy") -> nn.Module:
    if name == "focal":
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


