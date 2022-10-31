from .transformer_model import TransformerModel
import os
import logging
import itertools
import numpy as np
from sklearn import metrics as sklearn_metrics
from optparse import OptionParser,OptionGroup
import torch
from typing import List, Dict, Optional, Union, Tuple, Callable
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    BartTokenizer
)    
from .transformer_model import (
    TransformerModel,
    GeneratorModel
)
from ..readers.text_to_text_reader import Text2TextReader
from ..readers.input_example import Text2TextInput
from ..utils.model_analysis import text_to_text_analysis

util_logger = logging.getLogger('situation_modeling.base_modules.pretrained_encoder_decoder')

def simple_text_to_text_loader(
        data_path : str,
        split : str,
        config=None,
        evaluate=False,
        tokenizer=None
    ):
    """Loads a text2text dataset in the json format. Assumes 
    data in the format `data_path/{train,dev,test}.jsonl`.

    :param data_path: the target directory
    :param split: the target split, in {train,dev,test}.jsonl
    :param config: the global configuration 
    :param evaluate: whether to run in evaluation model
    """
    target_data = os.path.join(data_path,split+".jsonl")
    util_logger.info('Reading data from {}, evaluate={}'.format(target_data,str(evaluate)))

    data_container = Text2TextReader.from_file(
        target_data,
        config=config,
        evaluate=evaluate
    )

    #### sanity check here
    if tokenizer is not None and config is not None:
        util_logger.info(f'Sanity checking the data and counting truncation for {split}...')
        input_lengths = []
        output_lengths = []
        total_inputs = 0
        total_outputs = 0

        ### https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        for instance in data_container.data:
            for dinput in instance.texts:
                total_inputs += 1
                input_lengths.append(len(tokenizer.tokenize(dinput)))
            for oinput in instance.output:
                total_outputs += 1
                output_lengths.append(len(tokenizer.tokenize(oinput)))

        ### log truncation
        num_over = len([l for l in input_lengths if l > config.max_seq_length])
        util_logger.info(
            f"input stats: max={np.max(input_lengths)},mean={np.mean(input_lengths)}, truncate={num_over} / {total_inputs}"
        )
        out_over = len([l for l in output_lengths if l > config.max_output_length])
        util_logger.info(
            f"output stats: max={np.max(output_lengths)},mean={np.mean(output_lengths)}, truncate={out_over} / {total_outputs}"
        )
        del os.environ["TOKENIZERS_PARALLELISM"]

    return data_container.data

def simple_text_to_text_collator(
        batch : List,
        tokenizer,
        max_seq_length,
        max_prop_length : int = None
    ):
    """This function is called when the torch dataloader batches are enumerated. Main 
    function is turn batches into features 

    :param batch: the incoming batch 
    :param tokenizer: the model tokenizer 
    :param max_seq_length: the maximum sequence allowed 
    :param max_prop_length: the maximum output length 
    """
    main_inputs = list(itertools.chain(*[t.texts for t in batch]))
    outputs = list(itertools.chain(*[[str(o) for o in t.output] for t in batch]))
    assert len(main_inputs) == len(outputs)
    evaluate = [e.evaluation for e in batch][0]

    ## special for Bart
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
      
    tokenizer_inputs = tokenizer(
        main_inputs,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
        **extra_kw,
    )

    # for minput in main_inputs:
    #     if tokenizer.tokenize(minput) > 

    tokenizer_targets = tokenizer(
        outputs,
        max_length=max_prop_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt",
        **extra_kw,
    )

    source_ids  = tokenizer_inputs["input_ids"].squeeze(-1)
    target_ids  = tokenizer_targets["input_ids"].squeeze(-1)
    src_mask    = tokenizer_inputs["attention_mask"].squeeze(-1)
    target_mask = tokenizer_targets["attention_mask"].squeeze(-1)

    features = {
        "source_ids"  : source_ids,
        "source_mask" : src_mask,
        "target_ids"  : target_ids,
        "target_mask" : target_mask,
        "evaluate"    : evaluate,
        "print_out"   : {
            "text_out"  : outputs,
            "text_in"   : main_inputs,
            "guid"      : [i.guid if i.guid else str(k) for k,i in enumerate(batch)],
            "prefix"    : [i.prefix for i in batch]
        },
    }

    return (features,None)

class PretrainedEncoderDecoder(TransformerModel,GeneratorModel):
    """Generic transformer-based pretrained encoder decoder (e.g., T5, BART, etc..)
    which has the added feature of doing on-the-fly generation during training and evaluation. 
    """

    @classmethod
    def from_config(cls,config):
        """Loads a pretrained encoder decoder from configuration 

        :param config: the global configuration 
        """
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)

        return cls(
            model,
            tokenizer,
            model_config,
            max_seq_length=config.max_seq_length,
            max_prop_length=config.max_output_length,
            max_output_length=config.max_output_length,
            reader_function=simple_text_to_text_loader,
            collator_function=simple_text_to_text_collator,
            global_config=config,
        )

    def forward(self,features,labels=None):
        """This is a somewhat non-standard (and potentially problematic) version 
        of the forward method for the underlying `ConditionalSequenceGeneration`
        model. In addition to measuring loss, it can also be used to generate arbitrary 
        raw text output (e.g., for evaluation purposes during training)
        
        :param features: the target inputs
        :param labels: the target output labels (optional)
        """
        main_out = {"output" : None, "print_out" : {}}
        main_out["print_out"].update(features["print_out"])

        ## if used during training 
        if "target_ids" in features:

            ## output labels
            lm_labels = features["target_ids"]
            lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        
            outputs =  self.model(
                input_ids=features["source_ids"],
                attention_mask=features["source_mask"],
                labels=lm_labels,
                #decoder_attention_mask=features['target_mask'], # not strictly needed
                return_dict=True
            )

            main_out["loss"] = outputs.loss

        else:
            outputs = self.model(
                input_ids=features["source_ids"],
                attention_mask=features["source_mask"],
                return_dict=True 
            )
            ### getting out some target stuff 
            ## loss.logits[:,0,torch.tensor([0,1])]

        ## does on-the-fly generation if evaluation is set and
        if "evaluate" in features and features["evaluate"] is True:
            
            ## run generator 
            raw_out = self.generate(features)
            
            main_out["print_out"]["gen_out"] = [
                self.tokenizer.decode(ids.detach().cpu(),skip_special_tokens=True) for ids in raw_out
            ]

            ## label scores, need to reconsider how to implement this properly 
            # if "target_ids" in features and self.global_config.get_confidence:
            #     lm_labels = features["target_ids"]
            #     lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
            #     logits = outputs.logits
            #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100,reduction='none')
            #     scores = loss_fct(logits.view(-1, logits.size(-1)), lm_labels.view(-1))
            #     main_out["print_out"]["label_scores"] = [torch.sum(scores).cpu().tolist()]

        ### special masked generation

        return main_out

    def evaluate_output(self,output,out_file=None):
        """Method for generating output produced during training and/or evaluation. 
        Passes by default. 

        :param output: the output generated by runner
        :raises: ValueError
        """
        raise NotImplementedError('Deprecated, please use `situation_model` evaluation')

    def load_instance(self,query_input,**kwargs):
        """High-level interface for querying models. This model just 
        takes text input. 

        :param query_input: the query input 
        """
        output = "nil" if "output" not in kwargs else kwargs["output"]
        prefix = "" if "prefix" not in kwargs else kwargs["prefix"]
        guid = "-1" if "guid" not in kwargs else kwargs["guid"]
        
        instance = Text2TextInput(
            guid=guid,
            texts=[query_input.strip()],
            output=[output],
            prefix=prefix,
        )
        instance.evaluation = True
        return instance

    def posthoc_analysis(self,config):
        """Posthoc analysis code for performing different types of analysis on expeirments 

        :param config: the global configuration 
        """
        return text_to_text_analysis(config)

    
def params(config):
    from .transformer_model import params as t_params
    t_params(config)

    group = OptionGroup(config,"situation_modeling.models.pretrained_sequence_classification",
                            "Settings for pre-trained sequence classifier models")

    group.set_conflict_handler("resolve")

    group.add_option("--max_output_length",
                         dest="max_output_length",
                         default=120,
                         type=int,
                         help="the number output length [default=120]")
    group.add_option("--max_qa_input",
                         dest="max_qa_input",
                         default=60,
                         type=int,
                         help="the maximum length allowed for qa models [default=120]")    
    group.add_option("--retrain_batch",
                         dest="retrain_batch",
                         default=16,
                         type=int,
                         help="The batch for retraining [default=16]")
    group.add_option("--num_beams",
                         dest="num_beams",
                         default=3,
                         type=int,
                         help="The number of beams to use during search/full generation [default=3]")
    group.add_option("--do_sample",
                         dest="do_sample",
                         action='store_true',
                         default=False,
                         help="Use sampling instead of beam search [default=False]")
    group.add_option("--skip_em",
                         dest="skip_em",
                         action='store_true',
                         default=False,
                         help="Skip doing exact match [default=False]")
    group.add_option("--sort_output",
                         dest="sort_output",
                         action='store_true',
                         default=False,
                         help="Sort output for EM scoring [default=False]")
    group.add_option("--no_repeat_ngram_size",
                         dest="no_repeat_ngram_size",
                         default=0,
                         type=int,
                         help="Do not repeat ngrams of size greater than this [default=0]")
    group.add_option("--early_stop_decoding",
                         dest="early_stop_decoding",
                         action='store_true',
                         default=True,
                         help="Early stopping during decoding [default=True]")
    group.add_option("--get_confidence",
                         dest="get_confidence",
                         action='store_true',
                         default=False,
                         help="Get the output confidence score [default=False]")
    group.add_option("--top_p",
                         dest="top_p",
                         default=None,
                         help="Nucleaus sampling parameter [default=None]")
    group.add_option("--force_prefix",
                         dest="force_prefix",
                         default='',
                         help="run the model in a target mode [default='']")
    group.add_option("--top_k",
                         dest="top_k",
                         default=None,
                         help="Another sampling parameter [default=None]")
    group.add_option("--min_length",
                         dest="min_length",
                         default=None,
                         help="Minimal length for the generation [default=None]")
    group.add_option("--regenerate_eval",
                         dest="regenerate_eval",
                         action='store_true',
                         default=False,
                         help="Use generation at eval time [default=False]")
    group.add_option("--regen_k",
                         dest="regen_k",
                         default=3,
                         type=int,
                         help="The number of sentences to sample/generate for generative training [default=5]")
    
    config.add_option_group(group)
