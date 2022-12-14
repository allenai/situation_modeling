import os
import logging
import itertools
from optparse import OptionParser,OptionGroup
import torch
from typing import List, Dict, Optional, Union, Tuple, Callable
from torchmetrics import Accuracy, Precision, Recall

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from .transformer_model import TransformerModel
from ..readers.classification_reader import ClassificationReader

util_logger = logging.getLogger('situation_modeling.base_modules.pretrained_sequence_classifier')

def simple_classifier_loader(data_path : str, split : str, config=None):
    """A data loader for the simple `simple` (i.e., non-time related) situation
    datasets. 

    :param data_path: the target data path or directory 
    :param split: the target split, e.g., `train`, `dev`, `test`
    :rtype: list 
    """
    target_data = os.path.join(data_path,split+".jsonl")
    util_logger.info('Reading data from {}'.format(target_data))

    data_container = ClassificationReader.from_file(target_data)
    return data_container.data

def simple_classifier_collator(
        batch : List, tokenizer,
        max_seq_length,
        max_prop_length : int = None
    ):

    main_inputs = list(itertools.chain(*[t.texts for t in batch]))
    labels = list(itertools.chain(*[[int(o) for o in t.output] for t in batch]))
    
    text_encodings = tokenizer(
        main_inputs,
        return_tensors='pt',
        truncation=True,
        max_length=max_seq_length,
        padding='max_length'
    )

    out_labels = torch.tensor(labels)

    features = {
        "input_ids"      : text_encodings["input_ids"],
        "attention_mask" : text_encodings["attention_mask"],
        "text_in"        : main_inputs,  #<--- storing later for printing, evaluation
        "labels_out"     : labels,       #<--- same 
    }

    #text_encodings["text_inputs"] = main_inputs
    #text_encodings["raw_labels"] = labels

    return (text_encodings,out_labels)

def simple_classifier_evaluator(
        output,
        out_file=None
    ):
    """A simple evaluator for measuring accuracy and optionally 
    printing output 

    :param output: the output produced by the model 
    :param out_file: the path to print output (if set)
    """
    ### the part that computes metrics, prints output, etc..
    pass 
    
class PretrainedSequenceClassifier(TransformerModel):

    @classmethod
    def from_config(cls,config):
        """Loads a pre-trained sequence classifier from configuration 

        :param config: the global configuration 
        :raises: ValueError 
        """
        model_config = AutoConfig.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir,
        )
        model_config.num_labels = config.num_labels
        
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            cache_dir=config.cache_dir
        )

        return cls(
            model,
            tokenizer,
            model_config,
            reader_function=simple_classifier_loader,
            collator_function=simple_classifier_collator,
            max_seq_length=config.max_seq_length,
            global_config=config
        )

    def forward(self,features,labels):
        outputs = self.model(**features,labels=labels,return_dict=True)
        preds = torch.argmax(outputs.logits, dim=1).type_as(labels)

        output = {
            "loss"    : None,
            "outputs" : preds,
        }

        if labels is not None:
            output["loss"] = outputs.loss
            ## still having issues with device here
            acc = Accuracy()(preds.to("cpu"),labels.to("cpu")) 
            outputs["acc"] = acc

        return output
        
        # if labels is not None: 
        #     loss, logits = outputs[:2]
        #     ### accuracy
        #     preds = torch.argmax(logits, dim=1)
        #     acc = Accuracy()(preds.to("cpu"),labels.to("cpu")) #<-- complained before about devices, might slow things down
        #     return {"loss" : loss,"outputs" : logits, "acc" : acc}
        # else:
        #     return {"outputs" : outputs}

    def evaluate_output(self,output,out_file=None):
        """Method for generating output produced during training and/or evaluation. 
        Passes by default. 

        :param output: the output generated by runner
        :raises: ValueError
        """
        return simple_classifier_evaluator(
            output,
            out_file=out_file
        )

def params(config):
    from .transformer_model import params as t_params
    t_params(config)

    group = OptionGroup(config,"situation_modeling.models.pretrained_sequence_classification",
                            "Settings for pre-trained sequence classifier models")

    group.set_conflict_handler("resolve")
    
    group.add_option("--num_labels",
                         dest="num_labels",
                         default=2,
                         type=int,
                         help="the number of labels [default=2]")
    
    config.add_option_group(group)
