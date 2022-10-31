import os
import pandas as pd
import torch
from pathlib import Path
import logging
import itertools
import json
from torch import nn,Tensor
import transformers
from torch.utils.data import DataLoader
from ..base_modules import BuildModule, BuildPooler, BuildScorer, BuildSITAggregator
from ..readers.input_example import MultiTextPropositionInput
from ..losses import BuildLoss
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type
from collections import OrderedDict
from .. import initialize_config
from dataclasses import dataclass
from .model_base import (
    EncodePoolScoreModel,
    TranslationOutput
)
from transformers.training_args import TrainingArguments
from dataclasses import dataclass, asdict
import inspect
from sklearn import metrics as sklearn_metrics
from torch.utils.data import DataLoader


util_logger = logging.getLogger('situation_modeling.models.situation_model')

@dataclass
class MultiSituationOutput:
    """Helper class for printing situation output"""
    
    inputs : List[str]
    prop_lists : List[List[str]]
    gold_output : List[List[float]]
    pred_output: Optional[List[List[float]]]
    prop_times: Optional[List[List[int]]]
    guid: Optional[str] = None


    @classmethod
    def from_dict(cls, env): 
        """
        Construct from dict while ignoring extra args
        https://stackoverflow.com/a/55096964/2882125
        """     
        return cls(**{
            k: v for k, v in env.items() 
            if k in inspect.signature(cls).parameters
        })
    
    @classmethod
    def from_preds(cls, input_inst: MultiTextPropositionInput, preds: List[float]):
        """
        

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        input_inst : MultiTextPropositionInput
            DESCRIPTION.
        pred : List[float]
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # may be shorter due to padding
        n = len(input_inst.outputs)
        assert(n <= len(preds))
        
        actual_preds = preds[:n]
        actual_preds = [s[:len(input_inst.outputs[i])] for i,s in enumerate(actual_preds)]
        return cls(inputs=input_inst.texts,
                   prop_lists=input_inst.prop_lists,
                   gold_output=input_inst.outputs,
                   pred_output=actual_preds,
                   guid=input_inst.guid,
                   prop_times=input_inst.prop_times)
        
        


class SituationEncoderAggregator(EncodePoolScoreModel):

    @classmethod
    def from_config(cls,config):
        """Loads an aggregator from configuration

        :param config: the global configuration 
        """

        ### encoder module
        util_logger.info('Building encoder model...')
        encoder = BuildModule(config)
        embed_dim = encoder.get_word_embedding_dimension()
        
        
        # get token ids from encoder tokenizer
        prop_tok_id = encoder.tokenizer.vocab.get(config.prop_token)
        sit_tok_id = encoder.tokenizer.vocab.get(config.sit_token)
        

        ## pooler module 
        config.embed_dim = embed_dim
        config.sit_token_id = sit_tok_id
        config.prop_token_id = prop_tok_id
        pooler = BuildPooler(config)

        # situation aggregator module
        situation_aggregator = BuildSITAggregator(config)

        # scorer module
        scorer = BuildScorer(config)

        ## loss module
        loss = BuildLoss(config)

        device = None if not config.device else config.device
        return cls(
            modules=[encoder,pooler,situation_aggregator,scorer,loss],
            device=device, 
            global_config=config
        )

    def forward(self,batch):
        """Implements the forward method for this sequential module. It assumes 
        that `batch` (which is interpreted via the encoder model's collate function)
        expands out to `features` and `labels`
        
        :param batch: the batch to run
        """
        features,labels = batch
        for module in self:
            input = module(features,labels)
        return input



class SituationModel(SituationEncoderAggregator):
    """A better name"""


    def evaluate_output(self, output, out_file=None):

        ### single instances 
        if isinstance(output,dict):
            return SituationOutput.from_output(self.global_config,[output])
        
        eval_method = self.global_config.eval_method
        
        self.logger.info(f"Logging to {out_file}, eval_method={eval_method}.")


        if eval_method == "standard_multi":
            res = self.eval_standard_multi(output, out_file)
        elif eval_method == "standard":
            res = self.eval_standard(output, out_file)

        else:
            raise NotImplementedError(f"Unrecognized eval method: {eval_method}")
        
        return res

    def eval_standard(self, output, out_file=None):
        """Evaluates outputs

        :param output: the output generated from runner
        :param out_file: the file to print to 
        """
        self.logger.info(f'Evaluating model output, skip_em={self.global_config.skip_em}')

        metrics = {}
        sout = SituationOutput.from_output(self.global_config,output)

        if not self.global_config.skip_em:
            self.logger.info('Computing Exact Match (EM)')
            gen_em = sout.gen_em()
            self.logger.info(f'EM result: {gen_em}')
            
            if gen_em is not None:
                metrics["gen_em_acc"] = gen_em
        
        # we want to compute the global accuracy over all instances, in addition to average accuracy
        # over batches
        total = None
        correct = None
        for col in sout.metrics.columns:
            if "total" in col:
                total = sout.metrics[col].sum().item()
                metrics[col] = total
            elif "correct" in col:
                correct = sout.metrics[col].sum().item()
                metrics[col] = correct
            elif "acc" in col:
                # average accuracy over batches
                metrics[col] = sout.metrics[col].mean().item()
        
        if total and correct:
            # global accuracy
            metrics["global_acc"] = correct / total

        if out_file:
            with open(out_file,'w') as my_out:
                for instance in sout:
                    my_out.write(json.dumps(instance))
                    my_out.write("\n")

        return metrics

    def eval_standard_multi(self, output, out_file=None):
        """ 
        Simple evaluation method that just prints all predicted instances to file. Only difference with
        `eval_standard` is that we load the dataset at eval time which isn't efficient but was useful
        to grab information from the inputs (used for taking proposition distance info). Should probably
        be deprecated.
        """
        # default, and if no outfile provided, just assume dev or test eval
        split_name = "test" if self.global_config.test_eval else "dev"
        
        if out_file:
            out_path = Path(out_file)
            # somewhat hacky, but go by name of split from file, to also catch train eval
            for split in ["train", "test", "dev"]:
                if split in out_path.name:
                    split_name = split
                    break

            
        dset = self.model.load_data(self.global_config.data_dir, split_name)
        
        # get length N list of outputs for eval dataset of size N.
        unbatched_outs = []
        for out_d in output:
            unbatched_outs += out_d.get("outputs").tolist()
        
        assert(len(unbatched_outs) == len(dset))
        
        # create output instances for each eval example containing both
        # original input + gold as well as model prediction
        output_insts = []
        for out, instance in zip(unbatched_outs, dset):
            out_inst = MultiSituationOutput.from_preds(instance, out)
            output_insts.append(out_inst)
            
        # just save predictions to file
        data = [asdict(inst) for inst in output_insts]
        df = pd.DataFrame(data)
        df.to_csv(out_file, sep="\t")
        
        # compute some evaluation metrics to log
        metrics = {}
        
        return metrics


### helper class for parsing output

@dataclass
class SituationOutput:
    """Helper class for printing situation output. 

    """
    config     : Dict
    print_data : Dict
    metrics    : Dict

    @classmethod
    def from_output(cls,config,output):
        """Loads from outputs

        :param outputs: the outputs produced by the model
        """
        ## retrieve data needed for printing 
        print_data = {}
        print_out_keys = set(
            itertools.chain(*[list(i["print_out"].keys()) for i in output])
        )

        for key_name in print_out_keys:
            raw_data = [t["print_out"][key_name] for t in output]
            print_data[key_name] = [t for t in itertools.chain(*raw_data)]

        ### retrieve the situation data
        predicted_outputs = [o["outputs"].squeeze(0).cpu().tolist() for o in output]
        gold_labels = [
            o["labels"].squeeze(0).cpu().tolist() if o["labels"] is not None else None for o in output
        ]

        pred_probs = [
            o["pred_probs"].squeeze(0).cpu().tolist() for o in output
        ]
        print_data["pred_probs"] = pred_probs
        print_data["labels"] = gold_labels
        print_data["outputs"] = predicted_outputs

        if "metrics" in output:
            metrics = output["metrics"]
        else:
            try:
                metrics = pd.DataFrame([o["metrics"] for o in output])
            except KeyError:
                metrics = {}
                 
        return cls(config=config,print_data=print_data,metrics=metrics)

    def __repr__(self):
        return(f"SituationOutput()")

    @property
    def targets(self):
        return self.print_data.get("text_out",[])

    @property
    def outputs(self):
        return self.print_data.get("gen_out",[])

    @property
    def generative(self):
        return self.targets and self.outputs

    def gen_em(self):
        """Returns an exact match accuracy for generation
        
        :rtype: float or None
        """
        targets = [t.strip() for t in self.targets]
        outputs = [t.strip() for t in self.outputs]

        if targets and outputs and len(targets) == len(outputs):
            return sklearn_metrics.accuracy_score(targets,outputs)

    def __iter__(self):
        for instance in self.generate_situation_map():
            yield instance
            
    def generate_situation_map(self):
        """Generates a dictionary representation of the situation output 

        :rtype: dict 
        """
        situation_out = []

        ## situation input (make into properties)
        situation_texts   = self.print_data["sit_input"]
        proposition_texts = self.print_data["prop_lists"]
        pred_probs        = self.print_data["pred_probs"] \
          if "pred_probs" in self.print_data else []
        gold_labels       = self.print_data["labels"]
        predicted_outputs = self.print_data["outputs"]
        guids             = self.print_data["guid"]
        questions         = self.print_data["text_in"] \
          if "text_in" in self.print_data else []
        answers           = self.targets
        gen_answers       = self.outputs

        for k,(situation,prop_map) in enumerate(
                zip(situation_texts,proposition_texts)
            ):
            
            instance_dict = {}
            instance_dict["guid"] = guids[k]
            instance_dict["texts"] = situation.replace(self.config.sit_token,". ") 
            instance_dict["events"] = []

            subevents = [e.strip() for e in situation.split(self.config.sit_token) if e.strip()]
            prop_vals = prop_map[:len(subevents)]
            predicted = predicted_outputs[k][:len(subevents)]
            gold = gold_labels[k][:len(subevents)]
            if pred_probs: 
                lprobs = pred_probs[k][:len(subevents)]

            for w,subevent in enumerate(subevents):

                props = [p for p in prop_vals[w] if p.strip() != "empty"]
                fpredicted = predicted[w][:len(props)]
                fgold = gold[w][:len(props)]
                if pred_probs: 
                    probs = [p[2] for p in lprobs[w][:len(props)]]
                    instance_dict["events"].append((subevent,props,fpredicted,fgold,probs))
                else: 
                    instance_dict["events"].append((subevent,props,fpredicted,fgold))

            situation_out.append(instance_dict)

            ## only handles files with single questions

            ## hacky
            if questions:
                ## a bit of hack, the last `$question` part
                instance_dict["questions"] = [questions[k].split("$question")[-1].strip()]
            if answers:
                instance_dict["answers"] = [answers[k]]
            if gen_answers:
                instance_dict["gen_answers"] = [gen_answers[k]]

        return situation_out 

class SituationEncoderDecoder(SituationModel):

    @classmethod
    def from_config(cls,config):
        """Loads an aggregator from configuration

        :param config: the global configuration 
        """
        #config.module_type = 'situation_encoder_decoder'
        if config.loss_type not in set(["bce_multi","mcce_multi","logic_loss"]):
            raise ValueError(
                'Wrong loss function for this class, must be: `bce_multi`,`mcce_multi`, `logic_loss`'
            )

        return SituationModel.from_config(config)

    # def evaluate_output(self, output, out_file=None):
    #     """Evaluates outputs

    #     :param output: the output generated from runner
    #     :param out_file: the file to print to 
    #     """
    #     ### bad design, needs to be redone 
    #     metrics = SituationModel.evaluate_output(
    #         self,
    #         output,
    #         out_file=out_file
    #     )

    #     sout = TranslationOutput.from_output(
    #         self.global_config,
    #         output
    #     )
    #     if not self.global_config.skip_em:
    #         gen_em = sout.gen_em(sort=False)
    #         if gen_em: 
    #             metrics["gen_acc"] = sout.gen_em(sort=False)
    #     if out_file:
    #         out_dir = Path(out_file).parent
    #         out_dir.mkdir(parents=True, exist_ok=True)
    #         out_file = out_file.replace('.jsonl','_gen.jsonl')
    #         with open(out_file,'w') as my_out:
    #             for instance in sout:
    #                 my_out.write(json.dumps(instance))
    #                 my_out.write("\n")
    #     return metrics

def params(config):
    group = OptionGroup(config,"situation_modeling.models.situation_model",
                            "Standard settings for the situation model")
    
    
    group.add_option("--prop_token",
                        dest="prop_token",
                        default='[PROP]',
                        type=str,
                        help="Special proposition token [default='[PROP]']")
    
    group.add_option("--sit_token",
                        dest="sit_token",
                        default='[SIT]',
                        type=str,
                        help="Special situation token [default='[SIT]']")
    
    group.add_option("--eval_method",
                        dest="eval_method",
                        default='standard',
                        type=str,
                        help="Evaluation method to use. [default='standard']")

    config.add_option_group(group) 
