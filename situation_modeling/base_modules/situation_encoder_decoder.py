import os
import logging
import json
import torch
import numpy as np
import itertools
from torch.utils.data import DataLoader
from sklearn import metrics as sklearn_metrics
from ..base import ConfigurableClass
from ..readers import TextPropositionInput, MultiTextPropositionInput
from sentence_transformers import SentenceTransformer
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Callable
from .module_base import Model
from ..readers.situation_reader import SituationReader
from ..readers.mult_sent_sit_dataset import MultiSentSituationsDataset
from torch.utils.data import DataLoader
from torch import Tensor
from torchtyping import TensorType
from .transformer_model import (
    TransformerModel,
    GeneratorModel,
)
from .situation_encoder import (
    SituationEncoder,
    temporal_situation_collator,
    temporal_situation_loader
)
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
)

util_logger = logging.getLogger(
    'situation_modeling.base_modules.situation_encoder_decoder'
)

def simple_encoder_decoder_loader(
        data_path : str,
        split : str, 
        config=None,
        evaluate=False,
        tokenizer=None,
    ) -> List:
    """A data loader for the simple `simple` (i.e., non-time related) situation
    datasets.

    :param data_path: the target data path or directory
    :param split: the target split, e.g., `train`, `dev`, `test`
    """
    return temporal_situation_loader(
        data_path,
        split,
        config,
        evaluate,
        tokenizer 
    )

def temporal_encoder_decoder_situation_collator(
        batch : List,
        tokenizer,
        config
    ):
    """[summary]
    :param batch: Batch of data instances
    :type batch: List
    :param tokenizer: Model tokenizer
    :type tokenizer: [type]
    :param max_seq_length: Maximum length of input sequences.
    :type max_seq_length: int
    :param max_seq_len_prop: [description]
    :type max_seq_len_prop: int
    :param sit_token: [description]
    :type sit_token: str
    :param prop_token: [description]
    :type prop_token: str
    :rtype: tuple
    """
    return temporal_situation_collator(
        batch,
        tokenizer,
        config
    )

def _sanity_check(
        dataloader,
        config,
        logger,
    ):
    """Checks to if too much truncation is occurring. Can be
    turned on by adding `--sanity_check_data` to runner. 

    """
    logger.info('Still not doing any sanity checking...')


class SituationEncoderDecoder(SituationEncoder,GeneratorModel):
    """A situation encoder decoder implementation that does simultaenous 
    belief modeling and generation. 

    """
    @classmethod
    def from_config(cls,config):
        """Loads a situation encder decoder from config 

        :param config: the global configuration 
        """
        util_logger.info('Building the SituationEncoderDecoder')

        add_toks = [tok for tok in [config.sit_token, config.prop_token] if not tok in config.special_tokens]
        config.special_tokens += ";".join(add_toks)
        
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
            config=model_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path if config.tokenizer_name_or_path is not None else \
            config.model_name_or_path,
            cache_dir=config.cache_dir
        )

        return cls(
            model,
            tokenizer,
            model_config,
            ## should be removed below in favor of using global config
            max_seq_length=config.max_seq_length,
            max_prop_length=config.max_prop_length,
            max_output_length=config.max_output_length,
            reader_function=simple_encoder_decoder_loader,
            collator_function=temporal_encoder_decoder_situation_collator,
            global_config=config,
            special_tokens=config.special_tokens,
        )

    def run_decoder(self,features):
        """Sub-routine for running the decoder 

        :param features: the target features, which includes `source_ids` and
           `source_mask`
        """
        orig_labels = features["target_ids"].clone()
        
        lm_labels = features["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        ### run the text2text forward method
        outputs = self.model(
            input_ids=features["source_ids"],
            attention_mask=features["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=features['target_mask'],
            return_dict=True
        )
        if "qa_loss" in features:
            features["qa_loss"] += outputs.loss
        else:
            features["qa_loss"] = outputs.loss

        ### store generation probabilities
        ### requires another softmax, might be already stored somewhere
        if self.global_config.model_agreement:
            token_probs_batch = outputs.logits.softmax(-1)
            token_probs_batch = torch.prod(
                torch.gather(
                    token_probs_batch,
                    -1,
                    orig_labels.unsqueeze(-1),
                ).squeeze(-1)+((features["target_mask"] == 0)*1),
                dim=-1
            ).unsqueeze(-1)

            features["gen_probs"] = token_probs_batch

        return features

    def run_generator(self,features):
        """Runs the model generator

        :param features: the target features
        """
        raw_out = self.generate(features)
        features["print_out"]["gen_out"] = [
            self.tokenizer.decode(ids.detach().cpu(),skip_special_tokens=True) \
            for ids in raw_out
        ]
        return features

    def forward(
            self,
            features : Dict[str, Tensor],
            labels: Tensor = None
        ):
        """Returns embeddings for provided input in `features`. Technically, will add
        tokens embeddings, CLS token embeddings, etc, to the `features` input to be used with
        other modules.

        :note: this is repeated here from `SentenceTransformers.models.Transformer` just to see how it is
        implemented. Different variants here can be created as needed (e.g., a version that returns decoder
        output in the case of a transformer with decoder).
        """
        ### run ordinary forward
        features = super().forward(features,labels=labels)

        ### extra bit for generation 
        if "source_ids" in features and "target_ids" in features:
            features = self.run_decoder(features)
            
        ## generation output?
        if features.get("gen",False) is True and "source_ids" in features:
            features = self.run_generator(features)

        return features

def params(config):
    """Module-level configuration values 

    :param config: the global config 
    :rtype: None 
    """
    group = OptionGroup(config,"situation_modeling.models.situation_encoder_decoder",
                            "Standard settings for the situation encoder_decoder modules")

    config.add_option_group(group)
