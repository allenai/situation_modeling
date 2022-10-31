import os
import json
from ..base import ConfigurableClass
from ..readers import TextPropositionInput
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple
from torch import Tensor,nn

## consider making just ordinary nn.Module (

class Model(nn.Module,ConfigurableClass):
    """Base model class for building models (or individual pytorch `modules`)"""
    
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """Tokenize texts 

        """
        raise NotImplementedError('tokenizer not implemented')

    def encode_data(self,data_instance):
        """Encodes data in needed format 

        :param data_instance: the instance to be encoded
        :raises: ValueError
        """
        raise NotImplementedError('data instance encoding functionality not implemented')

    def _add_special_tokens(self,special_tokens : List):
        """Optional method for adding special tokens; passed by default

        :param special_tokens: the list of special tokens to add to internal tokenizer 
        """
        pass

    def forward(self, features : Dict[str, Tensor], labels: Tensor = None):
        """Returns embeddings for provided input in `features`. Technically, will add 
        tokens embeddings, CLS token embeddings, etc, to the `features` input to be used with
        other modules. 

        """
        raise NotImplementedError('Not to be used directly as a model!')
    
    def load_data(self,data_path : str, split : str,evaluate=False):
        """Load and encode data for the model 

        :param data_path: the location of the target data 
        :param split: the target split 
        :param evaluate: load in evaluation model, optional
        """
        raise NotImplementedError('data loading not implemented for this class')

    def load_instance(self,query_input,**kwargs):
        """high-level interface for querying models

        :param query_input: the query input 
        """
        raise NotImplementedError('instance loader not implemented for this class')

    def collate_fn(self,batch):
        """Function for creating batches 

        :param batch: the batch to encode
        """
        raise NotImplementedError('data collate functionality not implemented here')

    @property
    def custom_validation(self):
        """Determines where a custom validation step needs to be implemented
        
        """
        return False

    def evaluate_output(self,outputs):
        raise NotImplementedError('evaluator not implemented for this class')

    def evaluate_validation_output(self,output):
        """Evaluates the output generated from a validation run; passes
        by default. 

        :param output: the output generated from validation run 
        :rtype: None 
        """
        pass

    def sanity_check_data(self,dataloader):
        """Sanity check a given dataloader object

        :param dataloader: the target torch dataloader
        """
        self.logger.warning(
            'Sanity check not implemented, skipping'
        )

    def save(self, output_path : str):
        raise NotImplementedError

    @classmethod
    def load(input_path : str):
        raise NotImplementedError

    def modify_config(self,key,value):
        """Modify configuration settings in underlying model

        :param key: the key identifying the target value
        :param value: the new value 
        """
        self.logger.warning('Modifications for this class are not implemented')

    def posthoc_analysis(self,config):
        self.logger.info('No posthoc analysis to peform, passing...')

def params(config):
    group = OptionGroup(config,"situation_modeling.base_modules","Generic module settings")
    group.set_conflict_handler("resolve")

    group.add_option("--max_seq_length",
                         dest="max_seq_length",
                         default=120,
                         type=int,
                         help="the type of model to use [default=120]")

    group.add_option("--max_qa_length",
                         dest="max_qa_length",
                         default=50,
                         type=int,
                         help="the output to allow for qa [default=50]")
    group.add_option("--max_out_length",
                         dest="max_out_length",
                         default=128,
                         type=int,
                         help="the maximum length of output [default=128]")
    group.add_option("--max_data",
                         dest="max_data",
                         default=100000000,
                         type=int,
                         help="the maximum training data [default=100000000]")
    group.add_option("--max_eval_data",
                         dest="max_eval_data",
                         default=100000000,
                         type=int,
                         help="the maximum training data [default=100000000]")    
    group.add_option("--data_dir",
                         dest="data_dir",
                         default='',
                         type=str,
                         help="The directory where the data sits [default='']")
    group.add_option("--data_builder",
                         dest="data_builder",
                         default="simple_situation",
                         type=str,
                         help="The type of data builder to use [default='simple_situation']")
    group.add_option("--data_collator",
                         dest="data_collator",
                         default="temporal_situation_collator",
                         type=str,
                         help="The type of data collator to use [default='simple_situation_collator']")
    group.add_option("--eval_subdir",
                         dest="eval_subdir",
                         default='',
                         type=str,
                         help="Sub directory in evaluation path [default='']")
    group.add_option("--model_dir",
                         dest="model_dir",
                         default='',
                         help="The model to use for eval [default='']")
    group.add_option("--output_dir",
                         dest="output_dir",
                         default='',
                         help="The location to put output for model [default='']")
    group.add_option("--target_model",
                         dest="target_model",
                         default='',
                         type=str,
                         help="Path to target model (for loading model) [default=T5Classification]")
    group.add_option("--data_subdir",
                         dest="data_subdir",
                         default='',
                         type=str,
                         help="The subdirectory to find the data (if needed) [default='']")
    group.add_option("--module_type",
                         dest="module_type",
                         default="transformer_model",
                         type=str,
                         help="The type of module to use [default='']")
    group.add_option("--text2text_type",
                         dest="text2text_type",
                         default="basic",
                         type=str,
                         help="The type of text2text module to use [default='']")
    group.add_option("--max_prop_length",
                         dest="max_prop_length",
                         default=120,
                         type=int,
                         help="the type of model to use [default=120]")
    group.add_option("--sen_bert_encoder",
                         dest="sen_bert_encoder",
                         default='',
                         type=str,
                         help="the type of model to use [default=120]")
    group.add_option("--concat_self_hidden",
                        dest="concat_self_hidden",
                        action='store_true',
                        default=False,
                        help="Concatenate hidden and self attention vectors (for additional self-attention layer) [default=False]")
    group.add_option("--decoder_pooling",
                        dest="decoder_pooling",
                        action='store_true',
                        default=False,
                        help="For decoder based situation models, used decoder hidden states as pooled representation [default=False]")
    group.add_option("--add_sit_token",
                        dest="add_sit_token",
                        action='store_true',
                        default=False,
                        help="For decoder based situation models, used decoder hidden states as pooled representation with situation token included [default=False]")
    group.add_option("--add_full_input_rep",
                        dest="add_full_input_rep",
                        action='store_true',
                        default=False,
                        help="For decoder based situation models, used decoder hidden states as pooled representation with situation token included [default=False]")
    
    config.add_option_group(group)
