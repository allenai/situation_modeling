### utils for the runner code

import logging
import os
import sys
import json

__ALL__ = [
    "update_config"
]

_RUNNER_SETTINGS = [
    "wdir",
    "output_dir",
    "data_dir",
    "dev_eval",
    "test_eval",
    "train_eval",
    "train_name",
    "dev_name",
    "test_name",
    "no_training",
    "print_output",
]

util_logger = logging.getLogger(
    'situation_modeling.utils.runner_utils'
)

def parse_args_any(args):
    """Simple parser to getting overrides 

    :see: https://stackoverflow.com/questions/3356632/arbitrary-command-line-arguments-in-python

    :param args: the string of overrides
    """
    pos = []
    named = {}
    key = None
    for arg in args:
        if key:
            if arg.startswith('--'):
                named[key] = True
                key = arg[2:]
            else:
                named[key] = arg
                key = None
        elif arg.startswith('--'):
            key = arg[2:]
        else:
            pos.append(arg)
    if key:
        named[key] = True
    return named
    #return (pos, named)


def update_config(config,old_config):
    """Code for updating the configuration 

    :param config: the newest config (used to run this session) 
    :param old_config: the original model's config 
    :rtype: None 
    :raises: ValueError
    """
    ### check minimal items
    if not config.keep_old_config: 
        if not config.data_dir:
            raise ValueError(
                f"Must specify a data_dir"
            )
        if not config.dev_eval and not config.test_eval and not config.train_eval and config.no_training:
            raise ValueError(
                'Must specify a target task, `--dev_eval`, `--test_eval`, `--train_eval` or check that --no_training is False'
            )

    ### add new things
    for key,value in config.__dict__.items():
        if key not in old_config.__dict__:
            old_config.__dict__[key] = value
            util_logger.info(f"adding {key}={value} (didn't exist in old config)")

    ### mandatory updates
    if not config.keep_old_config:    
        for key in _RUNNER_SETTINGS:
            old_config.__dict__[key] = config.__dict__[key]
            util_logger.info(f"Changing mandatory setting: {key}={config.__dict__[key]}")

    ### update external project stuff
    if config.external_project and config.external_project != old_config.external_project:
        old_config.module_type = config.module_type
        old_config.external_project = config.external_project
        util_logger.info(f'updating the module type: {old_config.module_type}')

    if not config.keep_old_config and config.overrides:
        override_dict = parse_args_any(config.overrides.split())
        util_logger.info(f'parsing overrides: {config.overrides},dict={override_dict}')
        
        for key,value in override_dict.items():
            if key not in old_config.__dict__:
                raise ValueError(
                    f"Unknown configuration value: {key}"
                )
            if value == "True" or value == "true":
                target_value = True
            elif value == "False" or value == "false":
                target_value = False
            else:
                target_value = type(old_config.__dict__[key])(value)
                #target_type = type(old_config.__dict__[key])
            try: 
                old_config.__dict__[key] = target_value
                config.__dict__[key] = target_value
                util_logger.info(f'updating both configs, key={key},value={value}')
            except ValueError:
                raise ValueError(
                    f'Wrong type for update value: key={key}, value={value}'
                )
    return (config,old_config)
        
if __name__ == "__main__":
    parse_args_any(
        "--hello arg --bool True --something 3"
    )
