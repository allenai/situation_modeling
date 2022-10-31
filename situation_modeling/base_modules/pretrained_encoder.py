import os
import logging
import json
import torch
import itertools
from torch.utils.data import DataLoader
from ..base import ConfigurableClass
from ..readers import TextPropositionInput
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Callable
from .module_base import Model
from ..readers.situation_reader import SituationReader
from torch.utils.data import DataLoader
from torch import Tensor
from torchtyping import TensorType
from .transformer_model import TransformerModel

class PretrainedEncoder(TransformerModel):
    """Vanilla pre-trained encoder, following the original forward 
    implementation from sentence transformers.
    
    """
    def forward(self, features):
        """Returns token_embeddings, cls_token

        :note: this is repeated here from `SentenceTransformers.models.Transformer`
        :param features: the input encodings 
        :rtype: features: dict 
        :rtype: dict
        """
        trans_features = {
            'input_ids'     : features['input_ids'],
            'attention_mask': features['attention_mask']
        }
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.run_encoder(features) 
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]
        features.update({
            'token_embeddings'    : output_tokens,
            'cls_token_embeddings': cls_tokens,
            'attention_mask'      : features['attention_mask']
        })

        #Some models only output last_hidden_states and all_hidden_states
        if self.model.config.output_hidden_states:

            all_layer_idx = 2
            if len(output_states) < 3: all_layer_idx = 1
            hidden_states = output_states[all_layer_idx]

            features.update({
                'all_layer_embeddings': hidden_states
            })

        ### 
        
            
        return features
    
def params(config):
    from .module_base import params as m_params
    m_params(config)

    group = OptionGroup(config,"situation_modeling.models.pretrained_encoder",
                            "Parameters for building (vanilla) pretrained encoders")
    group.set_conflict_handler("resolve")
    config.add_option_group(group)
