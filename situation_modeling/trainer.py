import os
import argparse
import pytorch_lightning as pl
import logging
from pprint import pformat
import torch
from .base import LoggableClass
from .utils.callbacks import ModelCallback
from .utils.wandb_util import (
    init_wandb_logger,
    WandbArtifactCallback
)
from typing import Any, Callable, Dict, Optional, Tuple
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import PyTorchProfiler

## generic trainer mo
util_logger = logging.getLogger('situation_modeling.trainer')

class PlTrainer(pl.Trainer):
    """Nothing special here, add more custom features as needed
    """
    pass

class MyEarlyStopper(pl.callbacks.EarlyStopping):
    ### make compatible with logging to cpu

    def _evalute_stopping_criteria(self, current: torch.Tensor, trainer: 'pl.Trainer') -> Tuple[bool, str]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        #elif self.monitor_op(current - self.min_delta, self.best_score.to(trainer.lightning_module.device)):
        elif self.monitor_op(current - self.min_delta, self.best_score):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason
    

def setup_trainer(config) -> PlTrainer:
    """Sets up the trainer and associated call backs from configuration 

    :param configuration: the target configuration 
    :rtype: a trainer instance 
    """
    args = argparse.Namespace(**config.__dict__)
    mode = args.callback_mode #"max" if args.callback_monitor == "val_acc" else "min"
    if args.callback_monitor == "val_acc": mode = "max" #<-- default override

    util_logger.info('mode=%s via %s' % (mode,args.callback_monitor))

    if not config.cloud and not os.path.isdir(args.output_dir):
        util_logger.info('making target directory: %s' % args.output_dir)
        os.mkdir(args.output_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir,
            monitor=args.callback_monitor,
            mode=mode,
            save_top_k=1 if args.save_top_k != 0 else 0, #<-- saves either nothing or the best model 
            #filename="best",
            verbose=args.verbose,
            every_n_val_epochs=args.period
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [ModelCallback(),lr_monitor,checkpoint_callback]

    ## artifact callback
    if config.wandb_project and (config.save_wandb_results or config.save_wandb_model):

        if not config.wandb_name:
            raise ValueError(
                'TO back up artifacts, must provide name via `--wandb_name`'
            )

        artifact_callback = WandbArtifactCallback(
            config.output_dir,
            config.wandb_name,
            config.save_wandb_results,
            config.save_wandb_model,
        )
        callbacks.append(artifact_callback)

    if config.early_stopping:
        # early_stop_callback = pl.callbacks.EarlyStopping(
        #     monitor=args.callback_monitor,
        #     min_delta=0.00,
        #     patience=args.patience,
        #     verbose=args.verbose,
        #     mode=mode
        # )
        early_stop_callback = MyEarlyStopper(
            monitor=args.callback_monitor,
            min_delta=0.00,
            patience=args.patience,
            verbose=args.verbose,
            mode=mode
        )
        callbacks.append(early_stop_callback)

    reload_data = False
    n_gpus = args.n_gpu if torch.cuda.is_available() else 0

    ## train parameters
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=n_gpus,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        callbacks=callbacks,
        auto_lr_find=args.auto_lr_find,
        reload_dataloaders_every_epoch=reload_data,
        num_sanity_val_steps=0,
        log_gpu_memory='all',
        #log_every_n_steps=args.log_frequency,
        deterministic=config.deterministic,
        move_metrics_to_cpu=True, ###<--- hopefully help with memory
        amp_backend=args.amp_backend,
    )
    if config.wandb_project:
        train_params['logger'] = init_wandb_logger(config)
        train_params['logger'].log_hyperparams(vars(config)) # save config hyperparmas to wandb

    if config.run_profiler:
        filename = None #if not config.output_dir else os.path.join(config.output_dir,"pytorch_profiling.txt")
        util_logger.info(f'Running profiling, output file where to print: {filename}, output_dir={config.output_dir}')
        profiler = PyTorchProfiler(
            filename=filename,
            profiler_kwargs={"profile_memory" : True}
        )
        train_params["profiler"] = profiler


    ## log the full set of parameters
    util_logger.info(
        "\n===========\n"+pformat(train_params)+"\n==========="
    )

    trainer = PlTrainer(**train_params)
    
    return trainer
