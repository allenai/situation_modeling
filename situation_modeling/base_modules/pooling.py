### pooling module
import json
import os
import torch
from optparse import OptionParser,OptionGroup
from ..base import ConfigurableClass
#from sentence_transformers.models.Pooling import Pooling as OrigPooling
from typing import List, Dict, Optional, Union, Tuple, Iterable
from torch import Tensor,nn

### from sentence transformers
    
class BasePooler(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. 
    This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
        ):
        super(BasePooler, self).__init__()

        self.config_keys = [
            'word_embedding_dimension',
            'pooling_mode_cls_token',
            'pooling_mode_mean_tokens',
            'pooling_mode_max_tokens',
            'pooling_mode_mean_sqrt_len_tokens'
        ]

        if pooling_mode is not None:
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token   = (pooling_mode == 'cls')
            pooling_mode_max_tokens  = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([
            pooling_mode_cls_token,
            pooling_mode_max_tokens,
            pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens
        ])
        
        self.pooling_output_dimension =\
          (pooling_mode_multiplier * word_embedding_dimension)

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append('cls')
        if self.pooling_mode_mean_tokens:
            modes.append('mean')
        if self.pooling_mode_max_tokens:
            modes.append('max')
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append('mean_sqrt_len_tokens')

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        raise NotImplementedError('Not to be used directly')

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)

class BasicPooler(BasePooler):
    """The original sentence transformer pooler
    """
    
    def forward(self, features: Dict[str, Tensor]):
        """Implementation of a basic sentence pooler 

        :param features: the input encodings containing token and cls
        token embeddings, attention masks tensors, etc..
        """
        token_embeddings = features['token_embeddings']
        cls_token        = features['cls_token_embeddings']
        attention_mask   = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        ## cls pooling 
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        else:
            raise ValueError(
                'Only implemented now for cls token pooling, please implement others!'
            )

        ## max token pooling 
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)

        ## mean token pooling 
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({
            'sentence_embedding': output_vector
        })
        return features
    

#class SentencePooler(OrigPooling):

#class SentencePooler(BasePooler):
class SituationPooler(BasePooler):
    ## repeated here just to show default implementation
    
    def forward(self, features: Dict[str, Tensor],labels: Tensor = None):

        ### the standard embeddings to expect input (can be changed as needed) 
        token_embeddings = features['token_embeddings']
        cls_token        = features['cls_token_embeddings']
        attention_mask   = features['attention_mask']

        ### proposition stuff
        prop_token_embeddings = features["prop_token_embeddings"]
        #prop_cls_token        = features["prop_cls_tokens"]
        prop_attention_mask   = features["prop_attention_mask"]

        ##e.g., sit_tokens = some_func_over(features['token_embeddings'])

        ## Pooling strategy
        output_vectors   = []
        prop_out_vectors = []


        ## classifier token pooling 
        # if self.pooling_mode_cls_token:
        #     output_vectors.append(cls_token)
        #     ## applied to propositions 
        #     prop_out_vectors.append(prop_cls_token)

        ## max token pooling 
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]

            ### propositions 
            prop_input_mask_expanded = prop_attention_mask.unsqueeze(-1).expand(
                prop_token_embeddings.size()).float()
            prop_token_embeddings[prop_input_mask_expanded == 0] = -1e9
            prop_max_over_time = torch.max(prop_token_embeddings, 1)[0]
            
            output_vectors.append(max_over_time)
            prop_out_vectors.append(prop_max_over_time)
            
        ## mean token pooling 
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

            ### proposition vectors
            prop_input_mask_expanded = prop_attention_mask.unsqueeze(-1).expand(
                prop_token_embeddings.size()).float()
            prop_sum_embeddings = torch.sum(prop_token_embeddings * prop_input_mask_expanded, 1)
            prop_sum_mask = prop_input_mask_expanded.sum(1)
            prop_sum_mask = torch.clamp(prop_sum_mask, min=1e-9)
            prop_out_vectors.append(prop_sum_embeddings / prop_sum_mask)
                        

        output_vector = torch.cat(output_vectors, 1)
        prop_output_vector = torch.cat(prop_out_vectors,1)
        
        ## updates the input representations to have `sentence_embedding`
        features.update({
            'sentence_embedding': output_vector,
            "proposition_matrix": prop_output_vector,
        })

        return features

    @classmethod
    def from_config(cls,config):
        """Loads a pooler with a set pooler method (via `--pooler_method`)
        from configuration. 

        :param config: the global configuration 
        """
        return cls(
            config.embed_dim,
            pooling_mode_mean_tokens=True if config.pooler_method == 'mean' else False,
            pooling_mode_cls_token=True if config.pooler_method == 'cls' else False,
            pooling_mode_max_tokens=True if config.pooler_method == 'max' else False
        )

    
    

### can implement custom poolers that build representations from special tokens, implement more complex
## pooling strategies, etc..

def params(config):
    """Main configuration settings for this module 

    :param config: a global configuration instance
    :rtype: None 
    """
    group = OptionGroup(config,"situation_modeling.models.pooling",
                            "Standard settings for pooler models")

    group.set_conflict_handler("resolve")

    group.add_option("--pooler_type",
                         dest="pooler_type",
                         default="situation_pooler",
                         type=str,
                         help="the type of pooler to use [default='situation_pooler']")
    group.add_option("--pooler_method",
                         dest="pooler_method",
                         default="mean",
                         type=str,
                         help="the type of pooler method to use [default='mean']")
    
    config.add_option_group(group)      
