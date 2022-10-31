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

class BaseScorer(nn.Module, ConfigurableClass):
    """
    Handles scoring of situation representations and propositions. Given a vector representing
    a situation (such as a [SIT] token embedding) and a vector representing a proposition (such as 
    a [PROP] token embedding), compute a score in [0,1] representing the belief that the proposition 
    is true in the current situation.
    """
    def __init__(self, config):
        super(BaseScorer, self).__init__()
        self.config = config

    
    @classmethod
    def from_config(cls,config):
        return cls(
            config
        )

class DotProductScorer(BaseScorer):
    """
    DotProductScorer. Computes score as simple dot product between situation vector and 
    corresponding propsition vector.
    """
    def __init__(self,config):
        super(DotProductScorer, self).__init__(config)
        if self.config.class_output:
            raise ValueError("Cannot use dot product scoring with `class_output == True`!")


    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):

        ### encodings for text inputs and propositions 
        text_inputs = features["sentence_embedding"]
        proposition_matrix = features["proposition_matrix"]

        ## dims
        B,T,P,D = proposition_matrix.shape

        rep_enc_input : TensorType["B", "T", "P", "D"]  = text_inputs.unsqueeze(2).repeat(1,1,P,1)
        
        # normalize dot product by hidden dimension
        situation_tensor : TensorType["B", "T", "P"] = torch.sum(rep_enc_input * proposition_matrix,dim=-1) / D
        
        # unsqueeze last dimension to conform to loss following loss layer
        scores: TensorType["B", "T", "P", 1] =  situation_tensor.unsqueeze(-1)
        

        features.update({
            "scores": scores
            })

        return features


class BilinearScorer(BaseScorer):
    """
    BilinearScorer. Computes score as bilinear transformation between situation vector and 
    corresponding propsition vector, optionally with bias.

    https://pytorch.org/docs/1.9.0/generated/torch.nn.functional.bilinear.html
    """
    def __init__(self, config):
        super().__init__(config)
        D = config.embed_dim

        if not hasattr(config, "class_output"):
            # compatibility with older versions that don't have this flag in config
            self.out_dim = 1
        else:
            self.out_dim = 1 if not self.config.class_output else 3
        O = self.out_dim

        if hasattr(self.config,"concat_self_hidden") and self.config.concat_self_hidden:
            self.W = nn.Bilinear(D*2, D, O, bias=config.bi_scorer_bias)
            self.logger.info(f'concat_self_attention={str(self.config.concat_self_hidden)},D={str(D)}')
        else:
            self.W = nn.Bilinear(D, D, O, bias=config.bi_scorer_bias)

        ### O indices are determined by respective dataset reader
    
    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        ### encodings for text inputs and propositions 
        text_inputs = features["sentence_embedding"]
        proposition_matrix = features["proposition_matrix"]

        ## dims
        B,T,P,D = proposition_matrix.shape

        rep_enc_input : TensorType["B", "T", "P", "D"]  = text_inputs.unsqueeze(2).repeat(1,1,P,1)

        scores: TensorType["B", "T", "P", "O"] = self.W(rep_enc_input, proposition_matrix)

        features.update({
            "scores": scores
        })

        return features


class ConcatScorer(BaseScorer):
    """
    Based on NLI-style embeddings from Sentence-BERT (https://arxiv.org/pdf/1908.10084.pdf).
    https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/SoftmaxLoss.py

    """
    def __init__(self, config):
        super(ConcatScorer, self).__init__(config)
        self.D = config.embed_dim



        if not hasattr(config, "class_output"):
            # compatibility with older versions that don't have this flag in config
            self.out_dim = 1
        else:
            self.out_dim = 1 if not self.config.class_output else 3

        O = self.out_dim

        self.concat_sent_rep: bool = True # currently enabled by default
        self.concat_sent_diff: bool = config.cat_scorer_diff
        self.concat_sent_mult: bool = config.cat_scorer_mult

        self.num_vectors_concatenated = 0
        if self.concat_sent_rep:
            self.num_vectors_concatenated += 2
        if self.concat_sent_diff:
            self.num_vectors_concatenated += 1
        if self.concat_sent_mult:
            self.num_vectors_concatenated += 1

        self.W = nn.Linear(self.num_vectors_concatenated * self.D, O)

        logger.info(f"Set up linear layer with dimension {self.W.weight.shape}.")


    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        ### encodings for text inputs and propositions 
        text_inputs = features["sentence_embedding"]
        proposition_matrix = features["proposition_matrix"]

        ## dims
        B,T,P,D = proposition_matrix.shape

        rep_enc_input : TensorType["B", "T", "P", "D"]  = text_inputs.unsqueeze(2).repeat(1,1,P,1)

        vectors_concat = []

        if self.concat_sent_rep:
            vectors_concat.append(rep_enc_input)
            vectors_concat.append(proposition_matrix)

        if self.concat_sent_diff:
            vectors_concat.append(torch.abs(rep_enc_input - proposition_matrix))

        if self.concat_sent_mult:
            vectors_concat.append(rep_enc_input * proposition_matrix)

        # n = num vectors concatenated
        concat_rep_tensor: TensorType["B", "T", "P", "n*D"] = torch.cat(vectors_concat, 3)

        scores: TensorType["B", "T", "P", "O"] = self.W(concat_rep_tensor)

        features.update({
            "scores": scores
            })

        return features



def params(config):
    group = OptionGroup(config,"situation_modeling.base_modules.scoring",
                            "Settings for the scoring module")
    
    group.add_option("--scorer_type",
                        dest="scorer_type",
                        default='dot_prod_scorer',
                        type=str,
                        help="Scoring method to use. [default='dot_prod_scorer']")

    group.add_option("--bi_scorer_bias",
                        dest="bi_scorer_bias",
                        action='store_true',
                        default=False,
                        help="Whether to use bias for bilinear scorer. [default=False]")

    
    group.add_option("--cat_scorer_diff",
                        dest="cat_scorer_diff",
                        action='store_true',
                        default=True,
                        help="Whether to use diffs for concat scorer. [default=True]")

    
    group.add_option("--cat_scorer_mult",
                        dest="cat_scorer_mult",
                        action='store_true',
                        default=True,
                        help="Whether to use element-wise mult for concat scorer. [default=True]")


    group.add_option("--class_output",
                        dest="class_output",
                        action='store_true',
                        default=False,
                        help="Whether to use 3-way classification for prediction outputs. [default=False]")

    config.add_option_group(group) 
