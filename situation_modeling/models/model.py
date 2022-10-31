import logging
import os
import torch
from torch import nn,Tensor
import transformers
from torch.utils.data import DataLoader
from ..base import ConfigurableClass
from ..base_modules import BuildModule,BuildPooler
from ..losses import BuildLoss
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type
from collections import OrderedDict
from .. import initialize_config
from dataclasses import dataclass
from .classifier_model import ClassificationModel
from .text_to_text_model import Text2TextSequenceGenerationModel

from .situation_model import (
    SituationEncoderAggregator,
    SituationModel,
    SituationEncoderDecoder,
)

_AGGREGATORS = {
    #"situation" : (SituationModel,situation_settings),
    "situation"        : SituationModel,
    "situation_model"  : SituationModel,
    "classifier"       : ClassificationModel,
    "text2text"        : Text2TextSequenceGenerationModel,
    "situation_encoder_decoder"  : SituationEncoderDecoder,
}
    
util_logger = logging.getLogger('situation_modeling.models')


def BuildModel(config):
    """Factory method for building aggregators

    :param config: the global configuration 
    """
    util_logger.info(f'Building model with {config.model_type}')
    
    aclass = _AGGREGATORS.get(config.model_type,None)
    if aclass is None:
        raise ValueError(
            'Unknown aggregator type: %s, available: %s' %\
            (config.model_type,', '.join(_AGGREGATORS.keys()))
        )
    return aclass.from_config(config)


def params(config):
    """Main configuration settings for this module 

    :param config: a global configuration instance
    :rtype: None 
    """
    group = OptionGroup(config,"situation_modeling.models",
                            "Settings for model building")

    from ..base_modules import params as m_params
    m_params(config)
    from ..losses.loss import params as l_params
    l_params(config)
    from .situation_model import params as s_params
    s_params(config)
    from ..readers import params as r_params
    r_params(config)

    group.add_option("--aggregator_type",
                         dest="aggregator_type",
                         default="situation_model",
                         type=str,
                         help="the type of loss function to use [default='bpe']")
    group.add_option("--model_type",
                         dest="model_type",
                         default="situation_model",
                         type=str,
                         help="the type of encoder model to use [default='transformer_model']")

    config.add_option_group(group)


def main(argv):
    config = initialize_config(argv,params)

    ## load model
    util_logger.info('Loading model')
    model = BuildAggregator(config)
