import json
import os
import logging
import torch
from optparse import OptionParser,OptionGroup
from ..base import ConfigurableClass
from typing import List, Dict, Optional, Union, Tuple, Iterable
from torch import Tensor,nn
from torchtyping import TensorType

logger = logging.getLogger(__name__)

class BaseSITAggregator(nn.Module, ConfigurableClass):
    """
    Handles aggregation TODO
    """
    def __init__(self, config):
        super(BaseSITAggregator, self).__init__()
        self.config = config
        self.logger.info(
            f'Base situation aggregator loaded...'
        )

    @classmethod
    def from_config(cls,config):
        return cls(
            config
        )
class DefaultSITAggregator(BaseSITAggregator):
    """
    Default SIT aggregator module, passes through inputs.
    """


    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        return features, labels


class NaiveSelfAttSITAggregator(BaseSITAggregator):
    """
    Compute simple (non-parametereized) self-attention over [SIT] token embeddings,
    with softmax scaling and future token masking - tokens can only attend on
    themselves and past tokens.                                                   
    Based on http://peterbloem.nl/blog/transformers
    """

    def __init__(self, config):
        super(NaiveSelfAttSITAggregator, self).__init__(config)
        self.logger.info(
            f'Self attention aggregator loaded, concat_self={self.config.concat_self_hidden}'
        )

    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        ### encodings for text inputs and propositions 
        text_inputs: TensorType["B", "T",  "D"]        = features["sentence_embedding"]
        sit_mask: TensorType["B", "T"]  = features["sit_special_mask"]
        proposition_matrix: TensorType["B", "T", "P", "D"] = features["proposition_matrix"]


        # batch outer product to get B,T,T mask https://discuss.pytorch.org/t/batch-outer-product/4025/2
        batch_att_sit_mask: TensorType["B","T","T"] = torch.einsum('bi,bj->bij', (sit_mask, sit_mask))
        mask_inds: TensorType["B","T","T"] = (batch_att_sit_mask == 0)
        
        ## dims
        B,T,P,D = proposition_matrix.shape

        ## dot product between all [SIT] tokens, basis of dot-product attention
        raw_sit_dot_product : TensorType["B","T","T"] = torch.bmm(
            text_inputs,
            text_inputs.transpose(1,2)
        )

        # scaling softmax - multiply by 1 / sqrt(D)
        scaling = 1.0 / torch.sqrt(torch.ones_like(raw_sit_dot_product) * D)
        scaled_dot = raw_sit_dot_product * scaling

        ### masking constraint: [SIT] can only look in the past, set future entries to -inf so softmax=0
        indices = torch.triu_indices(T, T, offset=1)
        copy_scale_dot = scaled_dot.clone() # to enable inplace assignment
        copy_scale_dot[:, indices[0], indices[1]] = float('-inf')
        
        sit_attention_weights : TensorType["B","T","T"] = torch.softmax(
            copy_scale_dot,
            dim=2
        ).type_as(raw_sit_dot_product)

        # zero out all weights on masked sit tokens
        sit_attention_weights_clone = sit_attention_weights.clone()
        sit_attention_weights_clone[mask_inds] = 0


        ## compute weighted sum for each [SIT] token
        sit_self_attention_out : TensorType["B","T","D"] = torch.bmm(
            sit_attention_weights_clone,
            text_inputs
        )

        ### concat?
        if self.config.concat_self_hidden:
            sit_self_attention_out : TensorType["B","T","D*2"] = torch.cat(
                (sit_self_attention_out,text_inputs),
                -1
            ).type_as(text_inputs) #<-- might not need type_as

        features.update({
            "sentence_embedding": sit_self_attention_out,
            "sentence_embedding_pre_agg": text_inputs
            })

        return features, labels

class KeyValueSITSelfAttention(NaiveSelfAttSITAggregator):
    """situation self attention with additional key/value parameters"""
    def __init__(self, config):
        super(NaiveSelfAttSITAggregator, self).__init__(config)
        #self.config = config

        ### additional parameters
        self.key_params   = torch.nn.Linear(config.embed_dim,config.embed_dim,bias=False)
        self.query_params = torch.nn.Linear(config.embed_dim,config.embed_dim,bias=False)
        self.value_params = torch.nn.Linear(config.embed_dim,config.embed_dim,bias=False)
        self.norm = torch.nn.LayerNorm(config.embed_dim)
        self.norm2 = torch.nn.LayerNorm(config.embed_dim)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(config.embed_dim, config.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.embed_dim, config.embed_dim)
        )
        #self.do = nn.Dropout(0.0)

    def forward(self,features,labels=None):
        """Nearly same as above with additional key/query + value parameters 

        :param features: the incoming features 
        :param labels: the target labels 
        """
        
        text_inputs: TensorType["B", "T",  "D"]  = features["sentence_embedding"]
        sit_mask: TensorType["B", "T"]  = features["sit_special_mask"]
        proposition_matrix: TensorType["B", "T", "P", "D"] = features["proposition_matrix"]

        ## dims
        B,T,P,D = proposition_matrix.shape

        #### query, key values
        queries : TensorType["B","T","D"] = self.query_params(text_inputs).view(B,T,D)
        keys    : TensorType["B","T","D"] = self.key_params(text_inputs).view(B,T,D)
        values  : TensorType["B","T","D"] = self.value_params(text_inputs).view(B,T,D)

        # batch outer product to get B,T,T mask https://discuss.pytorch.org/t/batch-outer-product/4025/2
        batch_att_sit_mask: TensorType["B","T","T"] = torch.einsum('bi,bj->bij', (sit_mask, sit_mask))
        mask_inds: TensorType["B","T","T"] = (batch_att_sit_mask == 0)

        ## dot product between all [SIT] tokens, basis of dot-product attention
        raw_sit_dot_product : TensorType["B","T","T"] = torch.bmm(
            queries,
            keys.transpose(1,2)
        )
        scaling = 1.0 / torch.sqrt(torch.ones_like(raw_sit_dot_product) * D)
        scaled_dot = raw_sit_dot_product * scaling

        ### masking constraint: [SIT] can only look in the past, set future entries to -inf so softmax=0
        indices = torch.triu_indices(T, T, offset=1)
        copy_scale_dot = scaled_dot.clone() # to enable inplace assignment
        copy_scale_dot[:, indices[0], indices[1]] = float('-inf')
        
        sit_attention_weights : TensorType["B","T","T"] = torch.softmax(
            copy_scale_dot,
            dim=2
        ).type_as(raw_sit_dot_product)

        # zero out all weights on masked sit tokens
        sit_attention_weights_clone = sit_attention_weights.clone()
        sit_attention_weights_clone[mask_inds] = 0


        ## compute weighted sum for each [SIT] token
        sit_self_attention_out : TensorType["B","T","D"] = torch.bmm(
            sit_attention_weights_clone,
            values
        )

        ### layer norm with a ressidual layer, 
        sit_self_attention_out = self.norm(sit_self_attention_out + text_inputs)
        #sit_self_attention_out = self.do(sit_self_attention_out)

        feedforward = self.ff(sit_self_attention_out)
        sit_self_attention_out = self.norm2(feedforward + sit_self_attention_out)
        #sit_self_attention_out = self.do(sit_self_attention_out)

        ### concat?
        if self.config.concat_self_hidden:
            sit_self_attention_out : TensorType["B","T","D*2"] = torch.cat(
                (sit_self_attention_out,text_inputs),
                -1
            ).type_as(text_inputs) #<-- might not need type_as

        features.update({
            "sentence_embedding": sit_self_attention_out,
            "sentence_embedding_pre_agg": text_inputs
        })
    

def params(config):
    group = OptionGroup(config,"situation_modeling.base_modules.sit_aggregation",
                            "Settings for the situation aggregator module")
    
    group.add_option("--sit_agg_type",
                        dest="sit_agg_type",
                        default='default',
                        type=str,
                        help="Situation aggregation method to use. [default='default']")
    config.add_option_group(group) 
