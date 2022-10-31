import os
import logging
import json
import torch
import itertools
from torch.utils.data import DataLoader
from ..base import ConfigurableClass
from ..readers import TextPropositionInput
from transformers import AutoModel, AutoTokenizer, AutoConfig
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Callable
from .module_base import Model
from ..readers.situation_reader import SituationReader
from torch.utils.data import DataLoader
from torch import Tensor
from torchtyping import TensorType

util_logger = logging.getLogger('situation_modeling.models.transformer_model')

# ##########################
# # DATA READER FUNCTIONS  #
# ##########################

## than inherit from it

class BaseTransformer(Model):

    def __init__(
            self,
            model,
            tokenizer,
            config,
            special_tokens : str = '',
            reader_function : Callable = '',
            collator_function : Callable = '',
            instance_function : Callable  = '',
            max_seq_length : Optional[int] = None,
            max_output_length : Optional[int] = None,
            max_prop_length : Optional[int] = None,
            do_lower_case: bool = False,
            cache_dir: Optional[str] = None,
            global_config  = None,
    ):
        """Initializes a pre-trained model for sequence classification

        :param model_type: the type of pre-trained model to use
        :param num_labels: the number of labels
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.global_config = global_config #<--- useful for models that need additional settings
        self.max_seq_length = max_seq_length
        self.max_prop_length = max_prop_length
        self.max_output_length = max_output_length

        self.config_keys = [
            'max_seq_length',
            'do_lower_case',
            'max_output_length',
            'max_prop_length',
        ]

        ## special readers
        self.logger.info('adding special tokens')
        self._add_special_tokens(special_tokens)
        self._reader = reader_function
        self._collator = collator_function
        self._instance = instance_function
        if self.global_config.decoder_pooling and not hasattr(self.model,"decoder"):
            self.logger.warning(
                'Decoder pooling set though this model doesnt have a decoder, turning off'
            )
            self.global_config.decoder_pooling = False 

        self.logger.info(
            'Model loaded, (new) special_tokens=%s,max_seq_length=%s,max_prop_lengths=%s,max_output_length=%s,decoder_poooling=%s' %\
            (special_tokens,self.max_seq_length,self.max_prop_length,self.max_output_length,self.global_config.decoder_pooling)
        )

        ### freezing?
        if global_config.freeze_transformer:
            self.logger.info('Freezing encoder weights...')
            self._freeze_layers()
            
    def _freeze_layers(self):
        """Freezes some encoder layers 

        :param: https://stackoverflow.com/questions/71048521/how-to-freeze-parts-of-t5-transformer-model
        """
        modules_to_freeze = [
            self.model.encoder.block[i].layer[0] for i in range(len(self.model.encoder.block))
        ]
        self.logger.info(f'Number of modules frozen: {len(modules_to_freeze)}')
            
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False  # Actual freezing operation
        
    def load_data(
            self,
            data_path : str,
            split : str,
            evaluate=False
        ):
        """Load and encode data for the model

        :param data_path: the location of the target data
        :param split: the target split
        :raises: NotImplementedError
        """
        if not self._reader:
            raise NotImplementedError(
                'Reader is not implemented or built during object creation!'
            )
        return self._reader(
            data_path,
            split,
            config=self.global_config,
            evaluate=evaluate,
            tokenizer=self.tokenizer
        )

    def run_encoder(self,features):
        """Return the forward method of the underlying encoder model
        :param features: the target input features
        """
        if hasattr(self.model,"decoder"):
            return self.model.encoder(**features)
        return self.model(**features,return_dict=False)

    def run_decoder(self,features):
        return self.model.decoder(**feature_dict)

    def collate_fn(self,batch : List):
        """Function for creating batches and turning them into features
        and tensors.

        :param batch: the model batch
        :rtype batch: list
        :raises: NotImplementedError
        """
        if not self._collator:
            raise NotImplementedError(
                'Collator is not implemented or built during object creation!'
            )
        return self._collator(
            batch,
            self.tokenizer,
            self.max_seq_length,
            self.max_prop_length
        )

    def _add_special_tokens(self,special_tokens):
        """Method for adding any special tokens specified during initialization.

        :rtype: None
        :see: https://github.com/yakazimir/situation_modeling/issues/93
        """
        if special_tokens:
            token_list = [t.strip() for t in special_tokens.split(";")]
            # special_tokens_dict = {'additional_special_tokens': token_list}
            num_added_toks = self.tokenizer.add_tokens(token_list, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.logger.info('Added new tokens: %s' % token_list)

    def get_word_embedding_dimension(self) -> int:
        return self.model.config.hidden_size

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def encode_data(self,data_instance):
        """Encodes data in needed format

        :param data_instance: the instance to be encoded
        :raises: ValueError
        """
        return self._instance(data_instance)

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids

        :param texts: the texts to tokenizer
        """
        raise NotImplementedError(
            'Not implemented for this class, use `encode_data` instead'
        )

    def evaluate_output(self,output, out_file: str = None):
        """Method for generating output produced during training and/or evaluation.
        Passes by default.

        :param output: the output generated by runner
        """
        return {}

    def save(self, output_path: str):
        """Saves model to file

        :param output_path: the target path to put model
        """
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'transformer_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @classmethod
    def load(input_path : str):
        """Load models from file

        :param input_path: the location of model
        :returns: `TransformerModel` instance
        """
        ## load config
        config_name = os.path.join(input_path,'transformer_config.json')
        with open(sbert_config_path) as fIn:
            config = json.load(fIn)

        ## do auto loading with transformers
        model_config = AutoConfig.from_pretrained(
            input_path,
        )
        auto_model = AutoModel.from_pretrained(
            input_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            input_path,
        )
        ### doesn't save functions, can be looked up manually in actually classes
        return cls(
            auto_model,
            tokenizer,
            model_config,
            max_output_length=config['max_output_length'],
            max_prop_length=config['max_prop_length'],
            max_input_length=config['max_input_length'],
            do_lower_case=False, ##<--- remove
        )

    def modify_config(self,key,value):
        """Modify configuration settings in underlying model

        :param key: the key identifying the target value
        :param value: the new value 
        """
        hparams = self.global_config
        if key not in hparams.__dict__:
            self.logger.error('Key not found in configuration values: {}'.format(key))
            return
        hparams.__dict__[key] = value

class TransformerModel(BaseTransformer):
    pass

class GeneratorModel:
    """Convenient class for implementing a standard generator"""

    def generate(
            self,
            features,
            max_length=None,
            no_repeat_ngram_size=None,
            num_beams=None,
            do_sample=None,
            top_p=None,
            min_length=None,
            top_k=None,
            num_return_sequences=None
        ):
        """Calls the underlying model generator. Low-level function 
        used during training and validation. 

        :param features: the 
        """
        hparams = self.global_config

        no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size is not None else hparams.no_repeat_ngram_size
        max_length = max_length if max_length is not None else hparams.max_output_length
        num_beams = num_beams if num_beams is not None else hparams.num_beams
        do_sample = do_sample if do_sample else hparams.do_sample
        top_p = top_p if top_p is not None else hparams.top_p
        top_k = top_k if top_k is not None else hparams.top_k
        if do_sample and top_p: top_k = 0
        elif do_sample and top_k: top_p = None

        ## note : doesn't require outputs
        ### this is good to avoid issues with relative attention
        ### https://github.com/huggingface/transformers/issues/10484

        
        outs = self.model.generate(
            input_ids=features["source_ids"],
            attention_mask=features["source_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True, ## <---- look at this, eos_
            no_repeat_ngram_size=no_repeat_ngram_size,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            use_cache=True
        )
        return outs

def params(config):
    """Main configuration settings for this module

    :param config: a global configuration instance
    :rtype: None
    """
    group = OptionGroup(config,"situation_modeling.models.transformer_model",
                            "Standard settings for transformer based encoder models")

    group.set_conflict_handler("resolve")

    from .module_base import params as m_params
    m_params(config)

    group.add_option("--model_name_or_path",
                         dest="model_name_or_path",
                         default="roberta-base",
                         type=str,
                         help="the type of model to use [default='roberta-base']")
    group.add_option("--tokenizer_name_or_path",
                         dest="tokenizer_name_or_path",
                         default='roberta-base',
                         type=str,
                         help="the name of the target tokenizer [default='roberta-base']")
    group.add_option("--special_tokens",
                         dest="special_tokens",
                         default='',
                         type=str,
                         help="The type of special tokens to add (delimited by `;`) [default='']")
    group.add_option("--cache_dir",
                         dest="cache_dir",
                         default=None,
                         type=str,
                         help="the target cache directory for huggingface [default='']")
    group.add_option("--base_model",
                         dest="base_model",
                         default='roberta',
                         type=str,
                         help="the base model [default='roberta']")
    group.add_option("--freeze_transformer",
                        dest="freeze_transformer",
                        action='store_true',
                        default=False,
                        help="Freeze transformer weights [default=False]")


    config.add_option_group(group)
