import logging
import os
import itertools
from sklearn import metrics as sklearn_metrics
import torch
from torch import nn,Tensor
import transformers
from torch.utils.data import DataLoader
from ..base import ConfigurableClass
from ..base_modules import BuildModule,BuildPooler,SequenceClassifier
from ..losses import BuildLoss
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type
from collections import OrderedDict
from .. import initialize_config
from dataclasses import dataclass
from tqdm.autonotebook import trange
from dataclasses import dataclass

util_logger = logging.getLogger('situation_modeling.models.model_base')

class BasicAggregator(nn.Module,ConfigurableClass):
    def forward(self, batch):
        raise NotImplementedError('Must implement for each new aggregator!')

    def generate_output(self,instance):
        """Generates output from a simple string instance

        :param instance: string 
        """
        raise NotImplementedError

    def load_data(self,data_dir,split,evaluate=False):
        """Main method for loading a dataset using a target directory
        and split. 

        :param data_dir: the target directory 
        :param split: the target split 
        :param evaluate: run in evaluation mode?, optional
        """
        return self.model.load_data(data_dir,split,evaluate=evaluate)

    def load_instance(self,query_input,**kwargs):
        """high-level interface for querying models

        :param query_input: the query input 
        """
        return self.model.load_instance(query_input,**kwargs)

    def evaluate_output(self,output,out_file=None):
        raise NotImplementedError('Evaluation not implemented for this class')

    @property
    def device(self):
        return self.model.device

    def modify_config(self,key,value):
        """Modify configuration settings in underlying model

        :param key: the key identifying the target value
        :param value: the new value 
        """
        self.model.modify_config(key,value)

    def posthoc_analysis(self,config):
        self.logger.info('Posthoc analysis, not implemented, passing...')

class Aggregator(nn.Sequential,ConfigurableClass):
    """Base aggregator class"""

    def __init__(self,modules: Optional[Iterable[nn.Module]] = None,
                     device: Optional[str] = None,
                     train_settings : dict = {},
                     global_config = None
        ):
        """Create an `Aggregator` instance

        :param modules: the modules that form the basis of the `nn.Sequential` instance
        :param device: the device to run computations on (will default to `cuda` if available unless
        specified otherwise)
        """
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        ### auto-select gpu if available 
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info("Use pytorch device: {}".format(device))
        self.device = device

        self._target_device = torch.device(device)
        self.train_settings = train_settings 
        self.global_config = global_config
        
        super().__init__(modules)

    def generate_output(self,**instance):
        """Generates output from a simple string instance

        :param instance: string 
        """
        raise NotImplementedError

    def _first_module(self):
        """Returns the first module of this sequential embedder"""
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        """Returns the last module of this sequential embedder"""
        return self._modules[next(reversed(self._modules))]

    def forward(self, batch):
        raise NotImplementedError('Must implement for each new aggregator!')

    def fit(self,train_settings : dict):
        """The main training method 

        :param train_settings: the trainer settings
         """
        raise NotImplementedError(
            'Deprecated, please use pytorch-lightning trainer in `situation_modeling.runner`'
        )
    
    def evaluate_output(self,output,out_file=None):
        return self.model.evaluate_output(output,out_file=out_file)

    def load_data(self,data_dir,split,evaluate=False):
        return self.model.load_data(data_dir,split,evaluate=evaluate)

    def load_instance(self,query_input,**kwargs):
        """high-level interface for querying models

        :param query_input: the query input 
        """
        return self.model.load_instance(query_input,**kwargs)

    def sanity_check_data(self,dataloader):
        """Runs a sanity check of a target dataset and dataloader
        
        :param dataloader: the target dataloader 
        """
        self.model.sanity_check_data(dataloader)

    def collate_fn(self,batch):
        """Main method for featurizing incoming batches 

        :param batch: the incoming batch 
        """
        return self.model.collate_fn(batch)

    def modify_config(self,key,value):
        """Modify configuration settings in underlying model

        :param key: the key identifying the target value
        :param value: the new value 
        """
        self.model.modify_config(key,value)

class EncodePoolScoreModel(Aggregator):
    """Models that encode, pool and score"""
    
    @property
    def model(self):
        """The first module, which is always an encoder model 

        :rtype: torch.nn.Module
        """
        return self._first_module()

    @property
    def scorer(self):
        """The last module, which is always a scoring and loss function 

        :rtype: torch.nn.Module 
        """
        return self._last_module()

    def posthoc_analysis(self,config):
        return self.model.posthoc_analysis(config)
    


### helper classes for managing output

@dataclass
class TranslationOutput:
    """Helper class for translation output"""
    config : Dict
    print_data : Dict

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

        return cls(config=config,print_data=print_data)

    @property
    def prefixes(self):
        return self.print_data.get("prefix",[])

    @property
    def label_scores(self):
        return self.print_data.get("label_scores",[])

    @property
    def targets(self):
        return self.print_data.get("text_out",[])

    @property
    def outputs(self):
        return self.print_data.get("gen_out",[])

    def gen_em(self,sort=False):
        """Returns an exact match accuracy for generation
        
        :rtype: float or None
        """
        prefixes = self.prefixes
        targets = self.targets
        outputs = self.outputs
        
        if prefixes: 
            targets = [t.strip() for k,t in enumerate(targets) if not prefixes[k] or prefixes[k] == "answer:"]
            outputs = [t.strip() for k,t in enumerate(outputs) if not prefixes[k] or prefixes[k] == "answer:"]

        ### special sorting functionality for set processing
        ### assumes set items are delimited by `+`
        if sort is True:
            util_logger.info('Sorting output...')
            targets = ["+".join(sorted(t.split("+"))) for t in targets]
            outputs = ["+".join(sorted(o.split("+"))) for o in outputs]
                
        if targets and outputs and len(targets) == len(outputs):
            util_logger.info(
                'First few inputs: %s' % ', '.join(targets[:4])
            )
            util_logger.info(
                'First few outputs: %s' % ', '.join(outputs[:4])
            )
            return sklearn_metrics.accuracy_score(targets,outputs)

    @property
    def generative(self):
        return True

    def enumerate_instances(self):
        """Enumerate through instances for printing

        """
        guids    = self.print_data["guid"]
        text_in  = self.print_data["text_in"]
        prefixes = self.prefixes
        targets  = self.targets
        outputs  = self.outputs
        label_scores = self.label_scores

        total_outputs = []

        for k,identifier in enumerate(guids):
            instance_dict = {}
            instance_dict["id"] = identifier
            instance_dict["context"] = text_in[k]
            instance_dict["gen_out"] = outputs[k]
            if targets: 
                instance_dict["answer"]  = targets[k]
            if prefixes:
                instance_dict["meta"] = {}
                instance_dict["meta"]["prefix"] = prefixes[k]
            if label_scores:
                instance_dict["label_scores"] = label_scores[k]

            total_outputs.append(instance_dict)

        return total_outputs

    def __iter__(self):
        for item in self.enumerate_instances():
            yield item
