import os
import sys
import logging
import imp
from optparse import OptionParser,OptionGroup
from situation_modeling.utils.loader import load_module as load_module
from situation_modeling.utils.os_util import make_experiment_directory as make_wdir


USAGE = """usage: python -m situation_modeling mode [options] [--help]"""
DESCRIPTION = """Code base for situation modeling with transformers"""

_CONFIG = OptionParser(usage=USAGE,description=DESCRIPTION)

## logging

_CONFIG.add_option("--logging",dest="logging",default='info',type=str,
                      help="The logging level [default='']")
_CONFIG.add_option("--log_file",dest="log_file",default='pipeline.log',
                      help="The name of the log file (if logging to file) [default='pipeline.log']")
_CONFIG.add_option("--override",dest="override",action='store_true',default=False,
                      help="Override the current working directory and creat it again [default=False]")
_CONFIG.add_option("--cloud",dest="cloud",action='store_true',default=False,
                      help="Called when used in cloud environment [default=False]")
_CONFIG.add_option("--wdir",dest="wdir",default='',
                      help="The specific working directory to set up [default='']")
_CONFIG.add_option("--cuda_device",dest="cuda_device",default=-1,type=int,
                      help="The cuda device to run on (for GPU processes) [default=-1]")
_CONFIG.add_option("--device",dest="device",default="",type=str,
                       help="the type of device to use [default='']")

## WANDB GENERAL SETTINGS

_CONFIG.add_option("--wandb_project",
                     dest="wandb_project",
                     default=None,
                     help="The particular wandb project (if used) [default='']")
_CONFIG.add_option("--external_project",
                     dest="external_project",
                     default='',
                     help="A new piece of code to load with additional modules [default='']")
_CONFIG.add_option("--wandb_cache",
                     dest="wandb_cache",
                     default='',
                     help="The particular wandb project (if used) [default='']")
_CONFIG.add_option("--wandb_api_key",
                     dest="wandb_api_key",
                     default='',
                     type=str,
                     help="The particular wandb api key to use [default='']")
_CONFIG.add_option("--wandb_name",
                     dest="wandb_name",
                     default='new experiment (default)',
                     type=str,
                     help="The particular wandb api key to use [default='new experiment (default)']")
_CONFIG.add_option("--wandb_note",
                     dest="wandb_note",
                     default='empty',
                     type=str,
                     help="The note to use for the wandb [default='empty']")
_CONFIG.add_option("--wandb_model",
                     dest="wandb_model",
                     default='',
                     type=str,
                     help="Specifies a location to an existing wandb model [default='']")
_CONFIG.add_option("--tensorboard_dir",
                    dest="tensorboard_dir",
                    default=None,
                    help="The types of labels to use [default=None]")
_CONFIG.add_option("--save_wandb_model",
                    dest="save_wandb_model",
                    action='store_true',
                    default=False,
                    help="Backup the wandb model [default=False]")
_CONFIG.add_option("--quiet",
                    dest="quiet",
                    action='store_true',
                    default=False,
                    help="Reduce some of the debugging [default=False]")
_CONFIG.add_option("--save_wandb_results",
                    dest="save_wandb_results",
                    action='store_true',
                    default=False,
                    help="Save results to wandb [default=False]")
_CONFIG.add_option("--wandb_entity",
                    dest="wandb_entity",
                    default='',
                    type=str,
                    help="Backup the wandb model [default='']")
_CONFIG.add_option("--wandb_data",
                     dest="wandb_data",
                     default='',
                     type=str,
                     help="Link to the wandb data [default='']")
_CONFIG.add_option("--seed",
                    dest="seed",
                    default=42,
                    type=int,
                    help="random seed[default=42]")


gen_config = _CONFIG


_LEVELS = {
    "info"  : logging.INFO,
    "debug" : logging.DEBUG,
    "warning" : logging.WARNING,
    "error"   : logging.ERROR,
    "quiet"   : logging.ERROR,
}

def _logging(config):
  """Basic logging settings 

  :param config: the global configuration 
  """
  level = _LEVELS.get(config.logging,logging.INFO)
  if config.wdir and config.log_file and config.log_file != "None":
    log_out = os.path.join(config.wdir,config.log_file)
    logging.basicConfig(filename=log_out,level=level)

    ## redirect stdout to wdir (e.g., all of the tqdm stuff) 
    sys.stdout = open(os.path.join(config.wdir,"stdout.log"),'w')
    sys.stderr = open(os.path.join(config.wdir,"stderr.log"),'w')

  else:
    logging.basicConfig(level=level)

def initialize_config(argv,params=None):
    """Create a config and set up the global logging
    
    :param argv: the cli input 
    :param params: the additional parameters to add 
    """
    if params: params(_CONFIG)
    config,_ = _CONFIG.parse_args(argv)

    if config.wdir:
        wdir = make_wdir(config.wdir,config=config)
    _logging(config)
    return config    

def _load_module(module_path):
    """load a particular zubr module using format:
    zubr.module1.module12.ect.. 

    :param module_path: path of module to be loaded
    :type module_path: str
    :returns: loaded module
    :rtype: module 
    """
    try: 
        mod = __import__(module_path,level=0)
        for c in module_path.split('.')[1:]:
            mod = getattr(mod,c)
        return mod    
    except Exception as e:
        raise e

_SHORTCUTS = {
    "situation_modeling" : "situation_modeling.runner",
     "runner"            : "situation_modeling.runner"
}

def get_config(module_name,logging='info'):
    """Return back a configuration instance for a utility with default values 

    >>> from situation_modeling import get_config  
    >>> config = get_config('situation_model.model')
    >>> config.encoder_name 
    'bert-base-nli-mean-tokens'

    :param module: the name of the module to use
    """
    mod = _load_module(_SHORTCUTS.get(module_name,module_name))
    #mod = _load_module(module_name)

    if hasattr(mod,"params"):
        config = initialize_config(["--logging",logging],mod.params)
        return config
    raise ValueError('No config for this module: %s' % module_name)


### model building
from .models import BuildModel
from .runner import ModelRunner

#generic_config = get_config("runner")
from .base_modules import RegisterModule
from .base_modules import RegisterText2Text
