import os
import torch
from torch import nn,Tensor
import transformers
from torch.utils.data import DataLoader
from ..base import ConfigurableClass
from ..base_modules import SequenceClassifier
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type
from collections import OrderedDict
from .. import initialize_config
from dataclasses import dataclass
from tqdm.autonotebook import trange
from .model_base import Aggregator,BasicAggregator

class ClassificationModel(BasicAggregator):

    def __init__(self,model,device=None):
        super().__init__()
        self.model = model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

    def forward(self, batch):
        features,labels = batch
        return self.model(features,labels)

    @classmethod
    def from_config(cls,config):
        """Builds a classifier instance from config 

        :param config: the global configuration 
        """
        ### build model
        encoder = SequenceClassifier.from_config(config)
        device = None if not config.device else config.device
        return cls(encoder,device=device)
