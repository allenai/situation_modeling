from pprint import pformat
import argparse
import sys
import os
import resource
import json
import torch
import itertools
import logging
from pathlib import Path
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from . import initialize_config,get_config
from optparse import OptionParser,OptionGroup,Values
from torch import nn
from .base import ConfigurableClass,LoggableClass
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning import seed_everything

from typing import List, Dict, Optional, Union, Tuple, Iterable,Type,Any
from .utils.runner_utils import update_config
from .utils import (
    setup_wandb,
    set_seed,
    save_wandb,
)
from .models import (
    BuildModel,
)
import pytorch_lightning as pl
from .trainer import (
    setup_trainer
)
from transformers import (
    AdamW,
    Adafactor,
    get_linear_schedule_with_warmup ### might want to consider other things here
)
from .utils.wandb_util import download_wandb_models

util_logger = logging.getLogger(
    'situation_modeling.runner'
)

def get_cpu_mem():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

class ModelRunner(pl.LightningModule):
    """Pytorch module for running models (i.e., training, evaluation,..).
    This is intended to be general enough for any `aggregator` model that needs to be built
    in this library.
    """

    def __init__(self,model,config):
        """Creates model runner instance

        :param model: the underlying aggregator model (see
           details about construction in `cls.from_config`)
        :param config: the global configuration and set of hyper-parameters
        """
        super().__init__()
        self.model = model

        self.hparams.update(config) #<-- same as above, but `.update` works
        self.global_epoch_counter = 0
        
        self.model_logger.info(
            f'Loaded runner instance, global_epoch_counter={self.global_epoch_counter}'
        )

        
    def configure_optimizers(self):
        """Setup the main optimizer

        :returns: the main optimizer
        """
        model = self.model

        ### set up optimizer parameters
        no_decay = ["bias", "LayerNorm.weight"]
        parameters_first = [
            p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        parameters_sec = [
            p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]

        optimizer_grouped_parameters = [
            {
                "params"       : parameters_first,
                "weight_decay" : self.hparams.weight_decay
            },
            {
                "params"       : parameters_sec,
                "weight_decay" : 0.0
            },
        ]

        ### adafactor doesn't seem to completely work yet
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                relative_step=False
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def _step(self,batch):
        """Runs a single batch

        :param batch: the incoming batch
        """
        return self(batch)

    def forward(self,batch):
        return self.model(batch)

    @torch.no_grad()
    def evaluate_model(
            self,
            data_dir,
            name,
            out_path=None,
            evaluate=False
        ):
        """General evaluate loop

        :returns: dictionary of metrics collected during evaluation
        """
        self.model_logger.info('Building target dataloader, evaluate={}'.format(str(evaluate)))

        dataloader = self.generic_loader(
            data_dir,
            name,
            False,
            False,
            batch_size=self.hparams.eval_batch_size,
            evaluate=evaluate
        )

        self.model_logger.info(
            'Enumerating through data, model device=%s' % self._device
        )
        self.eval()
        outputs = []

        for batch in tqdm(dataloader):
            ## map to device batch to correct device
            batch = move_data_to_device(batch,device=self._device)

            output = self.common_step("val",batch)
            outputs.append(output)

        self.model_logger.info(
            f'Evaluating output, out_path={out_path}.'
        )
        return self.model.evaluate_output(outputs,out_file=out_path)

    def posthoc_analysis(self,config):
        """Run after model is done running 

        """
        return self.model.posthoc_analysis(config)

    def common_step(self, prefix: str, batch: Any):
        """Optimizer step common to both test and validation steps (just different prefixes)

        :param prefix: the target step type
        :param batch: the target batch
        :rtype: dict
        """
        output_dict = self._step(batch)

        ## losses
        if "loss" in output_dict:
            output_dict[f"{prefix}_loss"] = output_dict["loss"]
            del output_dict["loss"]

        ## other losses
        full_losses = {}
        for lname,lvalue in output_dict.get("full_losses",{}).items():
             full_losses[f"{prefix}_{lname}"] = lvalue
        output_dict["full_losses"] = full_losses
        
        ### metrics
        out_metrics = {}
        for mname,mvalue in output_dict.get("metrics",{}).items():
            out_metrics[f"{prefix}_{mname}"] = mvalue
        output_dict["metrics"] = out_metrics

        ## data stats
        

        return output_dict

    def test_step(self, batch, batch_idx):
        """Runs a single step over the validation data

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        return self.common_step("test",batch)

    def validation_step(self, batch, batch_idx):
        """Runs a single step over the validation data

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        return self.common_step("val",batch)

    def log_metrics_epoch_end(self,key,outputs,prog_bar=False):
        """Method for logging different metrics for either training 
        or validation runs after epoch. 
        
        :param key: the target key in the output to look for 
        :param outputs: the step outputs 
        :rtype: None 
        """
        metric_keys = set(
            itertools.chain(*[list(o[key].keys()) \
                                  if key in o else [] for o in outputs])
        )
        mkeys = set()
        for mkey in metric_keys:
            # ignore these as they are longs and raise exception if .mean() called on them
            if "_total" in mkey or "_correct" in mkey:
                continue

            ## average metric over full outputs 
            avg_metric = torch.stack(
                [x[key][mkey] for x in outputs if mkey in x[key]]
            ).mean()

            ### log metric 
            self.log(
                mkey.replace("batch_","avg_"),
                avg_metric,
                on_epoch=True,
                prog_bar=True if prog_bar and "acc" in mkey  else False
            )
            mkeys.add(mkey)
        return mkeys

    def validation_epoch_end(self,outputs):
        """End of validation epoch

        :param outputs: the output of the validation step
        :rtype: None
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss",avg_loss,on_epoch=True,prog_bar=True)

        ## compute alternative losses
        self.log_metrics_epoch_end("full_losses",outputs)
        mkeys = self.log_metrics_epoch_end("metrics",outputs,prog_bar=True)

        ### run model specific evaluator (if specified)
        metrics_out = self.model.evaluate_output(outputs)

        for metric_name,metric_value in metrics_out.items():
            ### avoid double printing metrics  
            if metric_name in mkeys: continue

            self.log(
                "val_%s" % metric_name,
                metric_value,
                on_epoch=True,
                prog_bar=True \
                if "_correct" not in metric_name and "_total" not in metric_name \
                else False
            )

    def training_step(self,batch,batch_idx):
        """Runs a single training step

        :param batch: the target batch
        :param batch_idx: the path id
        :rtype: dict
        :returns: dictionary that includes loss
        """
        try: 
            if isinstance(batch[0],dict):
                if "meta" not in batch[0]: batch[0]["meta"] = {}
                batch[0]["meta"]["epoch"] = self.global_epoch_counter
        except:
            pass

        output_dict = self._step(batch)
        loss = output_dict["loss"]

        out_dict = {
            "loss"        : loss,
            "outputs"     : None, #outputs #<--- remove this, too much memory
            "full_losses" : {},
            "metrics"     : {},
            "data_stats"  : {},
        }

        ## log loss
        self.log('batch_train_loss',loss)
        ### mem
        #self.log('Cpu-mem-usg',get_cpu_mem())

        ## log other losses (if provided)
        for lname,lvalue in output_dict.get("full_losses",{}).items():
            self.log("batch_train_%s" % lname,lvalue)
            out_dict["full_losses"][f"train_{lname}"] = lvalue

        ## store metrics (if provided) 
        for mname,mvalue in output_dict.get("metrics",{}).items():
            out_dict["metrics"][f"train_{mname}"] = mvalue

        ## store data stats (if provided)
        for dstat,dval in output_dict.get("data_stats",{}).items():
            out_dict["data_stats"][dstat] = dval

        return out_dict

    def training_epoch_end(self, outputs):
        """Called at the end of the training epoch

        :param outputs: the outputs of the train step
        :rtype: None 
        """
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()

        ## log loss 
        self.log(
            "avg_train_loss",
            avg_train_loss,
            on_step=False,
            on_epoch=True
        )

        ### log other losses and metrics (if provided)
        self.log_metrics_epoch_end(
            "full_losses",
            outputs
        )
        self.log_metrics_epoch_end(
            "metrics",
            outputs
        )

        self.global_epoch_counter += 1
        self.model_logger.info(
            f'updating global epoch counter: {self.global_epoch_counter}'
        )

    def get_lr_scheduler(self):
        """Sets up the optimizer learning rate scheduler

        """
        ## compute total steps
        #num_devices = max(1, self.hparams.n_gpu)
        num_devices = self.hparams.n_gpu if torch.cuda.is_available() else 1
        effective_batch_size = self.hparams.train_batch_size * self.hparams.gradient_accumulation_steps * num_devices
        total_steps = (len(self.train_dataloader().dataset) / effective_batch_size) * self.hparams.num_train_epochs

        self.model_logger.info(
            'total_steps computed for scheduler: %s, warmup step: %s' % (total_steps,str(self.hparams.warmup_steps))
        )

        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval" : "step",
            "frequency": 1
        }
        return scheduler

    def train_dataloader(self):
        """Loader to building training data, will also
        initialize the `lr_scheduler`.

        :rtype: None
        """
        dataloader = self.generic_loader(
            self.hparams.data_dir,
            self.hparams.train_name,
            final_eval=False,
            shuffle=not self.hparams.no_shuffle,
            batch_size=self.hparams.train_batch_size,
        )
        self.model_logger.info(
            'Length of training data loader %d' % len(dataloader)
        )
        return dataloader

    def common_loader(self,prefix):
        """Common loader for evaluat sets during training

        :param prefix: the particular evaluation split
        """
        dataloader = self.generic_loader(
            self.hparams.data_dir,
            prefix,
            final_eval=False,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
        )
        self.model_logger.info(
            'Length of %s data loader %d' % (prefix,len(dataloader))
        )
        
        return dataloader

    def val_dataloader(self):
        """Loader to building dev data

        :rtype: None
        """
        return self.common_loader(self.hparams.dev_name)

    def test_dataloader(self):
        """Loader to building dev data

        :rtype: None
        """
        return self.common_loader(self.hparams.test_name)

    @classmethod
    def from_config(cls,config):
        """Loads model instance from configuration

        :param config: the global configuration
        """
        if config.load_existing:
            return cls.load(config)

        model = BuildModel(config)
        config = config.__dict__
        return cls(model,config)

    @classmethod
    def load(cls,config,keep_old_config=False):
        

        ### wandb?
        if not os.path.isfile(config.load_existing):
            candidate_checkpoints = [os.path.join(config.load_existing,f) for f in os.listdir(config.load_existing) if '.ckpt' in f]
            if candidate_checkpoints:
                if len(candidate_checkpoints) > 1:
                    util_logger.warning(f'More than one checkpoint found, taking first: {candidate_checkpoints}')
                config.load_existing = candidate_checkpoints[0]
            else: 
                download_wandb_models(config)

        util_logger.info(
            f'Found an existing checkpoint, trying to load: {config.load_existing}, keep_old_config={keep_old_config}'
        )

        device = "cuda" if torch.cuda.is_available() and not config.no_gpu else "cpu"
        if not config.device: config.device = device

        ## load raw checkpoint
        raw_checkpoint = torch.load(
            config.load_existing,
            map_location=config.device
        )
        
        hparams = raw_checkpoint["hyper_parameters"]
        util_logger.info(
            'Loaded raw checkpoint, now re-building model'
        )
        
        ## resurrect the old config
        old_config = Values(hparams)
        old_config.load_existing = ''
        config,old_config = update_config(config,old_config)

        # re-build from factory using updated settings
        model = BuildModel(old_config)

        util_logger.info('Now building from checkpoint again..')
        ## load from checkpoint
        model = cls.load_from_checkpoint(
            model=model,
            config=old_config.__dict__,
            checkpoint_path=config.load_existing
        )

        ## switch back to device (is this in the checkpoint somewhere?)
        device = "cuda" if torch.cuda.is_available() and not config.no_gpu else "cpu"
        model = model.to(device)
        return model

    @property
    def model_logger(self):
        """Returns a logger instance

        :returns: logger instance
        """
        level = '.'.join([
            __name__,type(self).__name__
        ])
        return logging.getLogger(level)

    def modify_config(self,key,value):
        """Modify configuration settings in underlying model

        :param key: the key identifying the target value
        :param value: the new value 
        """
        self.model.modify_config(key,value)

    ### loader

    def generic_loader(
            self,
            data_dir,
            name,
            final_eval=False,
            shuffle=False,
            batch_size=1,
            evaluate=False,
        ):
        """Generic method for building a dataloader

        :param name: the particular split to load
        :param final_eval: switch to indicate running a final evaluation
        :param shuffle: whether or not to shuffle the data
        :param batch_size: the batch size to use
        """
        dset = self.model.load_data(data_dir,name,evaluate=evaluate)

        dataloader = DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.model.collate_fn
        )

        return dataloader

    ### high-level query interface

    @torch.no_grad()
    def query(self,query_input,**kwargs):
        """High-level interface for querying models

        :param query_input: the input
        """
        instance = self.model.load_instance(query_input,**kwargs)

        dloader = DataLoader(
            [instance],
            batch_size=1,
            shuffle=False,
            drop_last=False,
            collate_fn = self.model.collate_fn
        )

        for batch in dloader:
            batch = move_data_to_device(batch,device=self._device)
            output = self(batch)
            return self.model.evaluate_output(output)
        
def main(argv):
    """The main execution method

    :param argv: the cli arguments
    """
    ## intialize config and any models/data needed from wandb
    util_logger.info('Setting up configuration for model runner...')
    config = initialize_config(argv,params)

    util_logger.info(
         "\n===========\n"+pformat(config.__dict__)+"\n==========="
    )

    setup_wandb(config)
    wandb_runner = None
    seed_everything(config.seed, workers=True)

    ## create runner instance and device
    model = ModelRunner.from_config(config)
    metrics = {}

    ## run training
    if not config.no_training:

        trainer = setup_trainer(config)
        trainer.fit(model)
        if config.wandb_project:
            wandb_runner = trainer.logger.experiment

        ### log all of the trainer metrics 
        for key,value in trainer.callback_metrics.items():
            if key in ["log", "progress_bar"] or "acc" not in key: continue
            try:
                metrics[key] = value.detach().item()
            except:
                pass

        ## re-load best model if model is backed up
        if config.dev_eval or\
          config.test_eval and\
          trainer.checkpoint_callback.best_model_path:

            best_path = trainer.checkpoint_callback.best_model_path
            config.load_existing = best_path

            ### log best score
            metrics["best_%s" % config.callback_monitor] =\
              trainer.checkpoint_callback.best_model_score.detach().item()

            ## load best model
            util_logger.info('Trying to load best model, loc=%s' % best_path)
            model = ModelRunner.load(config,keep_old_config=True)

    util_logger.info(
        'Loaded newest model ready for evaluation, running on device=%s' %\
        model._device
    )

    ## evaluations 
    for (split,do_eval) in [
            (config.train_name,config.train_eval),
            (config.dev_name,config.dev_eval),
            (config.test_name,config.test_eval),
        ]:
        if not do_eval: continue

        out_path = f"{split}_eval.tsv" if not config.print_json else f"{split}_eval.jsonl"
        if config.print_output:
            out_dir_path = Path(config.output_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            full_path = os.path.join(config.output_dir,out_path)
        else:
            full_path = ""

        metrics_out = model.evaluate_model(
            config.data_dir,
            split,
            full_path,
            evaluate=True,
        )

        ## print metrics
        if metrics_out:
            for metric_name,metric_value in metrics_out.items():
                metrics[f"{split}_{metric_name}"] = metric_value
        
    ## optional model posthoc_analysis
    analysis_out = model.posthoc_analysis(config)
    if isinstance(analysis_out,dict):
        metrics.update(analysis_out)
    
    ## print output (if specified)
    if config.print_output:
        save_wandb(
            config,
            metrics,
            runner=wandb_runner
        )

    

    ### save metrics
    with open(os.path.join(config.output_dir,"metrics.json"),'w') as mout:
        mout.write(json.dumps(metrics,indent=4))

    
    

def params(config):
    """Params for this module

    :param config: the global configuration
    """
    group = OptionGroup(config,"situation_modeling.runner",
                            "Parameters for running and training models")

    group.set_conflict_handler("resolve")

    from .models import params as a_params
    a_params(config)

    group.add_option("--load_existing",
                         dest="load_existing",
                         default='',
                         type=str,
                         help="Path to existing model or checkpoint to run [default='']")
    group.add_option("--dev_name",
                         dest="dev_name",
                         default='dev',
                         type=str,
                         help="The standard name of dev [default='dev']")
    group.add_option("--test_name",
                         dest="test_name",
                         default='test',
                         type=str,
                         help="The standard name of test [default='test']")
    group.add_option("--dev_eval",
                         dest="dev_eval",
                         action='store_true',
                         default=False,
                         help="run an evaluation of the dev eval [default=False]")
    group.add_option("--deterministic",
                         dest="deterministic",
                         action='store_true',
                         default=False,
                         help="Run in deterministic mode [default=False]")
    group.add_option("--no_training",
                         dest="no_training",
                         action='store_true',
                         default=False,
                         help="Skip the training step [default=False]")
    group.add_option("--train_eval",
                         dest="train_eval",
                         action='store_true',
                         default=False,
                         help="run an evaluation of the train eval [default=False]")
    group.add_option("--test_eval",
                         dest="test_eval",
                         action='store_true',
                         default=False,
                         help="run an evaluation of the test [default=False]")
    group.add_option("--sanity_check_data",
                         dest="sanity_check_data",
                         action='store_true',
                         default=False,
                         help="Run a sanity check of the data after loading (e.g., check for truncation) [default=False]")
    group.add_option("--early_stopping",
                         dest="early_stopping",
                         action='store_true',
                         default=False,
                         help="Use early stopping [default=False]")
    group.add_option("--num_workers",
                         dest="num_workers",
                         default=4,
                         type=int,
                         help="number of number of processes when loading data [default=4]")
    group.add_option("--callback_monitor",
                         dest="callback_monitor",
                         default="val_loss",
                         type=str,
                         help="batch size [default='val_loss']")
    group.add_option("--callback_mode",
                         dest="callback_mode",
                         default="min",
                         type=str,
                         help="batch size [default='min']")
    group.add_option("--callback_prefix",
                         dest="callback_prefix",
                         default="checkpoint",
                         type=str,
                         help="batch size [default='checkpoint']")
    group.add_option("--log_streamlit_wandb",
                         dest="log_streamlit_wandb",
                         default=False,
                         action='store_true',
                         help="Log streamlit run output to run specified in `wandb_name` [default=False]")
    group.add_option("--period",
                         dest="period",
                         default=1,
                         type=int,
                         help="the period (number of epochs) between checkpoints [default=1]")
    group.add_option("--patience",
                         dest="patience",
                         default=5,
                         type=int,
                         help="Patient level (when early stopping is used) [default=5]")
    group.add_option("--n_gpu",
                         dest="n_gpu",
                         default=1,
                         type=int,
                         help="The number of gpus to use [default=1]")
    group.add_option("--overrides",
                         dest="overrides",
                         default='',
                         type=str,
                         help="Overrides when running new models [default='']")
    group.add_option("--opt_level",
                         dest="opt_level",
                         default='01',
                         type=str,
                         help="The optional level  [default='']")
    group.add_option("--save_top_k",
                         dest="save_top_k",
                         default=1,
                         type=int,
                         help="Number of models to save [default=1]")
    group.add_option("--drop_last",
                         dest="drop_last",
                         action='store_true',
                         default=False,
                         help="Drop the last batches [default=False]")
    group.add_option("--no_gpu",
                         dest="no_gpu",
                         action='store_true',
                         default=False,
                         help="Do not use gpu [default=False]")
    group.add_option("--verbose",
                         dest="verbose",
                         action='store_true',
                         default=False,
                         help="Verbose option [default=False]")
    group.add_option("--auto_lr_find",
                         dest="auto_lr_find",
                         action='store_true',
                         default=False,
                         help="automatic learning rate finder [default=False]")
    group.add_option("--tpu_cores",
                         dest="tpu_cores",
                         default=0,
                         type=int,
                         help="The number of TPU cores (for tpu usage) [default=0]")
    group.add_option("--special_device",
                         dest="special_device",
                         default='cuda',
                         type=str,
                         help="The special device (for loading) [default='cuda']")
    group.add_option("--print_output",
                         dest="print_output",
                         action='store_true',
                         default=False,
                         help="Print output [default=False]")
    group.add_option("--run_profiler",
                         dest="run_profiler",
                         action='store_true',
                         default=False,
                         help="Run the system pytorch profiler [default=False]")
    group.add_option("--print_json",
                         dest="print_json",
                         action='store_true',
                         default=False,
                         help="Print json output [default=False]")
    group.add_option("--remove_models",
                         dest="remove_models",
                         action='store_true',
                         default=False,
                         help="Remove models/checkpoints [default=False]")
    group.add_option("--remove_checkpoints",
                         dest="remove_checkpoints",
                         action='store_true',
                         default=False,
                         help="Remove models/checkpoints [default=False]")
    group.add_option("--keep_old_config",
                         dest="keep_old_config",
                         action='store_true',
                         default=False,
                         help="Does not update old config [default=False]")
    group.add_option("--weight_decay",
                         dest="weight_decay",
                         default=0.0,
                         type=float,
                         help="the weight decay amount [default=0.0]")
    group.add_option("--adam_epsilon",
                         dest="adam_epsilon",
                         default=1e-8,
                         type=float,
                         help="adam epsilon parameter [default=1e-8]")
    group.add_option("--warmup_steps",
                         dest="warmup_steps",
                         default=0,
                         type=int,
                         help="warmnup steps [default=0]")
    group.add_option("--max_grad_norm",
                         dest="max_grad_norm",
                         default=1.0,
                         type=float,
                         help="maximum gradient norm [default=1.0]")
    group.add_option("--checkpoint_path",
                         dest="checkpoint_path",
                         default='',
                         type=str,
                         help="Path to checkpoint (for loading model) [default=T5Classification]")
    group.add_option("--train_name",
                         dest="train_name",
                         default="train",
                         type=str,
                         help="The name of training data [default='train']")
    group.add_option("--eval_name",
                         dest="eval_name",
                         default="generic",
                         type=str,
                         help="The name of evaluation data [default='generic']")
    group.add_option("--model_name",
                         dest="model_name",
                         default='n/a',
                         type=str,
                         help="The type of model (for plotting purposes) [default='n/a']")
    group.add_option("--gradient_accumulation_steps",
                         dest="gradient_accumulation_steps",
                         default=1,
                         type=int,
                         help="number of gradient accumulations [default=1]")
    group.add_option("--learning_rate",
                         dest="learning_rate",
                         default=5e-5,
                         type=float,
                         help="learning rate [default=5e-5]")
    group.add_option("--num_train_epochs",
                         dest="num_train_epochs",
                         default=3,
                         type=int,
                         help="number of training iterations [default=3]")
    group.add_option("--no_shuffle",
                         dest="no_shuffle",
                         action='store_true',
                         default=False,
                         help="Remove shuffling [default=False]")
    group.add_option("--train_batch_size",
                         dest="train_batch_size",
                         default=16,
                         type=int,
                         help="batch size [default=16]")
    group.add_option("--eval_batch_size",
                         dest="eval_batch_size",
                         default=1,  ##<---- makes the evaluation easier
                         type=int,
                         help="the size of the eval batch size [default=1]")
    group.add_option("--adafactor",
                         dest="adafactor",
                         action='store_true',
                         default=False,
                         help="Use adafactor [default=False]")
    group.add_option("--fp_16",
                         dest="fp_16",
                         action='store_true',
                         default=False,
                         help="use fp_16 precision [default=False]")
    group.add_option("--amp_backend",
                         dest="amp_backend",
                         type='str',
                         default='native',
                         help="amp backen to use (for fp_16) [default=`native`]")    
    group.add_option("--log_frequency",
                         dest="log_frequency",
                         default=100,
                         type=int,
                         help="use fp_16 precision [default=100]")

    config.add_option_group(group)


if __name__ == "__main__":
    main(sys.argv[1:])
