import pytorch_lightning as pl
from ..base import LoggableClass
import copy

class ModelCallback(pl.Callback,LoggableClass):
    """Logger for T5"""

    def on_validation_end(
            self,
            trainer,
            pl_module
        ):
        """Called on validation end 

        :param trainer: the main trainer object
        :param pl_module: the lightning module running the model
        """
        self.logger.info("Validation Results...\n==================")
        metrics = trainer.callback_metrics
        for key in sorted(metrics):
            if key in ["log", "progress_bar"]: continue
            self.logger.info(
                '{} = {} '.format(key.strip(),str(metrics[key]).strip())
            )
        self.logger.info("\n=================")

    def on_init_start(self,trainer):
        self.logger.info('Starting the checkpoint')

    def on_init_end(self,trainer):
        self.logger.info('Finished setting up the logger callback')
