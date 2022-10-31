### pooling module

from optparse import OptionParser,OptionGroup


# from sentence_transformers.models.Pooling import Pooling as OrigPooling
from typing import List, Dict, Optional, Union, Tuple, Iterable
from collections import Counter
import numpy as np
import torch
from torch import Tensor, Size
from torch import nn
from torchtyping import TensorType

from ..base import ConfigurableClass
from .pooling import BasePooler


__all__ = [
    "SituationPropositionPooler",
]


def get_id_str(idxs: List[int], pop_idx: int = None) -> str:
    """
    Helper function to convert list representing a tensor entry to string notation.
    [d1, d2, ..., dn] -> "d1_d2_..._dn". If `pop_idx` specified, it will be omitted
    from the string representation.
    """
    idxs_copy = idxs.copy()
    if pop_idx:
        idxs_copy.pop(pop_idx)
    return "_".join([str(x) for x in idxs_copy])

def create_gather_idx_tensor(gather_idxs: Tensor, embs_shape: Size, max_gather: int,
                             device: str = "cpu") -> Tuple[Tensor, Tensor]:
    """Create an index tensor for use with the `torch.gather` function, for collecting 
    embeddings with the indices in `gather_idxs`.

    Given an embeddings tensor of shape (d1,... dn-1, dn), dn is the embedding dim
    and dn-1 is the index along which we want to gather embeddings. In our case, dn-1 is
    the sequence length dimension L, and we would like to gather T embeddings corresponding
    to T special [SIT] tokens in the sequence.

    :param gather_idxs: A 2D tensor where each row is the index of an embedding vector
    of dimension [d1,..,dn-1]  
    :type gather_idxs: Tensor
    :param embs_shape: The size (d1,... dn-1, dn) of the embeddings tensor.
    :type embs: Size
    :param max_gather: The maximum number of indices to gather, T, s.t. T <= L == dn-1.
    :type max_gather: int
    :return: Index tensor of size (d1,... dn-1, dn)
    :rtype: Tensor
    :return: Mask tensor of size (d1,... dn-1)
    :rtype: Tensor
    """
    assert(gather_idxs.shape[1] == len(embs_shape) -1), (f"gather_idxs.shape[1]={gather_idxs.shape[1]}" 
    f", should be {len(embs_shape) -1}")

    

    emb_dim_axis = len(embs_shape) - 1
    emb_dim = embs_shape[emb_dim_axis]
    gather_idx_dim = len(embs_shape) - 2


    # create index tensor of from embs shape, where if embs is shaped [d1, d2, d_{n-1}, d_n]
    #  index tensor is of shape [d1, d2,.., d_{n-2}, max_gather]
    idx_tensor_shape = list(embs_shape)[:-1]
    assert(embs_shape[gather_idx_dim] >= max_gather)
    idx_tensor_shape[gather_idx_dim] = max_gather
    index_tensor = torch.zeros(idx_tensor_shape, dtype=torch.int64).to(device)
    
    # count number of special token appearances per sequence
    c = Counter()

    for gather_idx in gather_idxs.tolist():
        # get sequence id
        loc_id_str = get_id_str(gather_idx, pop_idx=gather_idx_dim)
        
        # get position of special token in sequence
        pos = gather_idx[gather_idx_dim]

        # set correct index along gather dimension the special token
        # count thus far for current sequence 
        gather_idx[gather_idx_dim] = c[loc_id_str]
        index_tensor[tuple(gather_idx)] = pos
        c[loc_id_str] += 1

    # add back dimension dim
    repeat_dims = tuple([1] * len(index_tensor.shape) + [emb_dim])
    res = index_tensor.unsqueeze(emb_dim_axis).repeat(*repeat_dims)
    
    # calculate mask
    mask = (res[..., 0] != 0)
    
    return res, mask.int().to(device)

    

class SituationPropositionPooler(BasePooler):
    """
    Pooler for collecting encoded representations corresponding to special 
    situation and propostion tokens. 
    """

    def __init__(self,
                word_embedding_dimension: int,
                sit_token_id: int,
                prop_token_id: int
                ):
        super(SituationPropositionPooler, self).__init__(word_embedding_dimension)

        self.config_keys = ['sit_token_id', 'prop_token_id']

        self.embed_dim = word_embedding_dimension
        self.prop_token_id = prop_token_id
        self.sit_token_id = sit_token_id

        ### linear layer
        self.linear_layer = torch.nn.Linear(
            word_embedding_dimension,
            word_embedding_dimension,
            bias=False
        )
        
    
    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):

        D = self.embed_dim

        ### the standard embeddings to expect input (can be changed as needed) 
        token_embeddings: TensorType["B","1","L","D"] = features['token_embeddings']
        input_ids: TensorType["B","1","L"] = features['input_ids']

        ### proposition stuff
        prop_token_embeddings: TensorType["B","T","P","L2","D"] = features["prop_token_embeddings"]
        prop_ids: TensorType["B","T","P","L2"] = features["prop_input_ids"]
        
        _, T, _, _ = prop_ids.shape
        
        if "sentence_embedding" not in features:

            # gather the special situation token embeddings
            sit_gather_ids = (input_ids == self.sit_token_id).nonzero()
            sit_gather_idxs, sit_sp_mask = create_gather_idx_tensor(sit_gather_ids, 
                                                        token_embeddings.shape,
                                                        max_gather=T,
                                                    device=token_embeddings.device)

            sit_sp_embs = torch.gather(token_embeddings,dim=2,index=sit_gather_idxs)
            sit_sp_embs: TensorType["B","T","D"] = sit_sp_embs.squeeze(1)
            sit_sp_mask: TensorType["B","T"] = sit_sp_mask.squeeze(1)

            ## resize mask and mask out inactive [SIT] token embeddings
            sit_mask_resized: TensorType["B","T","D"] = sit_sp_mask.unsqueeze(-1).repeat(1,1,D)
            masked_sit_sp_embs: TensorType["B","T","D"] = sit_sp_embs * sit_mask_resized
        
            features.update({
                "sentence_embedding": masked_sit_sp_embs,
                "sit_special_mask": sit_sp_mask
            })

        ## in case proposition shaven't already been pooled (e.g., via `decoder pooling`) 
        if "proposition_matrix" not in features:
            # gather the special proposition token embeddings
            prop_gather_ids = (prop_ids == self.prop_token_id).nonzero()
            prop_gather_idxs, _ = create_gather_idx_tensor(prop_gather_ids, 
                                                               prop_token_embeddings.shape,
                                                               max_gather=1,
                                                               device=prop_token_embeddings.device)

            prop_sp_embs = torch.gather(prop_token_embeddings,
                                            dim=3,
                                            index=prop_gather_idxs)
            prop_sp_embs: TensorType["B","T","P","D"] = prop_sp_embs.squeeze(3)
        
            features.update({
                "proposition_matrix": prop_sp_embs
            })

        ### linear layer and norm
        sit_reps = self.linear_layer(features["sentence_embedding"])
        prop_reps = self.linear_layer(features["proposition_matrix"])
        features["sentence_embedding"] = torch.nn.functional.normalize(sit_reps,p=2,dim=2)
        features["proposition_matrix"] = torch.nn.functional.normalize(prop_reps,p=2,dim=3)

        return features

    @classmethod
    def from_config(cls,config):
        return cls(
            config.embed_dim,
            config.sit_token_id,
            config.prop_token_id
        )

    
