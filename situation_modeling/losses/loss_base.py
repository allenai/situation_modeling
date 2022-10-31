import os
import torch
from torchtyping import TensorType
from torch import nn,Tensor
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type
from ..base import ConfigurableClass

class ScoreLoss(nn.Module,ConfigurableClass):
    def __init__(self, config):
        super(ScoreLoss, self).__init__()
        self.config = config

    def forward(self,features: Dict[str, Tensor], labels: Tensor = None):
        """Takes a set of input features, `features`, and produces an output 
        space and/or loss.
        :rtype: dictionary
        """
        raise NotImplementedError
        
    @classmethod
    def from_config(cls,config):
        return cls(
            config
        )