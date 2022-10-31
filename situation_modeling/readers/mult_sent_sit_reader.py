from typing import Dict, List
import sys
from pathlib import Path
import json
import re
import itertools
import logging
import numpy as np
import random

from torch.utils.data import Dataset

from .input_example import MultiTextPropositionInput, MultiTextTokenizedPropsInput
from .constraints import parse_basic_implications_conjunctions
from .data_reader import JsonlReader

util_logger = logging.getLogger('situation_modeling.readers.mult_sent_sit_reader')


__all__ = [
    "MultiSentSituationReader",
    "MultiSentTokPropsSituationReader"
]

LABELS = {
    "yes"   : 1.0,
    "no"    : 0.0,
    "maybe" : 0.5,
}


class SituationJsonlReader(JsonlReader):
    """Custom reader that has support for dynamically created mask examples
    """

    @classmethod
    def json_file_reader(cls,path,evaluation,config):
        total_data = []
        with_story_labels = 0
        
        with open(path) as my_json:
            for k,line in enumerate(my_json):
                line = line.strip()
                try:

                    json_line = json.loads(line)
                    if config:
                        excluded_rules = set([r.strip() for r in config.exclude_rules.split(';')])
                        line_instance = cls._read(json_line,excluded_rules)
                    else:
                        line_instance = cls._read(json_line)
                    line_instance.evaluation = evaluation

                    ### turn off qa components 
                    if config and (config.no_generation or config.no_qa):
                        line_instance.question = []
                        line_instance.answer  = []

                    ### turn off constraints 
                    if config and config.no_constraint_loss:
                        line_instance.constraints = []

                    if config and config.exclude_rules:
                        pass

                    if line_instance.story_labels:
                        with_story_labels += 1
                    total_data.append(line_instance)
                    
                except json.JSONDecodeError as e:
                    util_logger.warn(f'Line {k} JSONDecodeError: {e}, skipping...')

                ## some logging to sanity check data
                if k < 3 and config and not config.quiet:
                    util_logger.info(f"\n ========\n raw line {k}= \n \t{json_line} \n processed line= \n \t {line_instance}")
                    util_logger.info("\n ========")

        ### debugging info related to [SIT] generation 
        util_logger.info(f'Total number of instances with story labels: {with_story_labels}')
        return total_data

    @staticmethod
    def parse_constraints(constraint_list):
        pass 

class MultiSentSituationReader(SituationJsonlReader):
    """Reader for inputs with multiple situations such as stories or procedures.
    
    >>> from situation_modeling.readers import StoryReader 
    >>> dataset = SituationReader.from_file('etc/examples/multi_situation_data.jsonl')
    >>> len(dataset) 
    1 
    >>> dataset.data 
    [MultiTextPropositionInput(texts=['Julie journeyed to the kitchen.', 'Afterwards she moved to the school.'],
    prop_list=prop_lists=[['Julie at kitchen', 'Julie at hallway', 'Julie at garden'], ['Julie at school', 'Julie at park']],outputs=[[1.0, 0.0, 0.0], [1.0, 0.0]])]
    >>> dataset.size 
    1
    >>> dataset[0]
    MultiTextPropositionInput(texts=['Julie journeyed to the kitchen.', 'Afterwards she moved to the school.'],
    prop_list=[['Julie at kitchen', 'Julie at hallway', 'Julie at garden'], ['Julie at school', 'Julie at park']],outputs=[[1.0, 0.0, 0.0], [1.0, 0.0]])
    """

    @staticmethod 
    def _read(instance,excluded_rules=set()):
        """Reads multi situation modeling output

        :param instance: the target instance to be read
        :rtype: situation_modeling.readers.input_example.InputBase
        """
        texts  = instance["texts"]
        props = instance["prop_lists"]
        outs  = instance["outputs"]
        identifier = instance["guid"] if "guid" in instance else instance["id"]

        ## for multi-task model
        try: 
            question = [instance["question"]] if isinstance(instance["question"],str) else instance["question"]
            answer = [instance["answer"]] if isinstance(instance["answer"],str) else instance["answer"]
        except KeyError:
            question = []
            answer   = []
            
        prop_times = instance["prop_times"] if "prop_times" in instance else []
        constraints = [] if "constraints" not in instance else instance["constraints"]
        neg_breakpoints = [] if "neg_breakpoints" not in instance else instance["neg_breakpoints"]
        pos_breakpoints = [] if "pos_breakpoints" not in instance else instance["pos_breakpoints"]
        sit_breakpoints = [] if "sit_breakpoints" not in instance else instance["sit_breakpoints"]
        story_labels    = [] if "story_labels"    not in instance else instance["story_labels"]
        blocks = [] if "blocks" not in instance else instance["blocks"]

        if constraints:
            new_list = []
            for constraint in constraints:
                cname = constraint.split("_")[0].strip()
                if cname in excluded_rules: continue
                new_list.append(constraint)
            
            constraints = parse_basic_implications_conjunctions(new_list)

        ## convert from strings
        if isinstance(texts,str):
            texts = texts.split(";")
        if isinstance(props,str):
            prop_lists = props.split(";")
            props = [p.split(". ") for p in prop_lists]
        elif isinstance(outs,str):
            out_lists = outs.split(";")
            outs = [o.split("-") for o in out_lists]
            outs = [float(LABELS.get(v.strip(),v)) for v in outs]

        return MultiTextPropositionInput(
            guid=identifier,
            texts=texts,
            outputs=outs,
            prop_lists=props,
            question=question,
            answer=answer,
            prop_times=prop_times,
            constraints=constraints,
            neg_breakpoints=neg_breakpoints,
            pos_breakpoints=pos_breakpoints,
            sit_breakpoints=sit_breakpoints,
            story_labels=story_labels,
            blocks=blocks
        )

class MultiSentTokPropsSituationReader(SituationJsonlReader):
    """Reader for multi-situation inputs where propositions are tokenized.
    """

    @staticmethod 
    def _read(instance):
        """Reads multi situation modeling output

        :param instance: the target instance to be read
        :rtype: situation_modeling.readers.input_example.InputBase
        """
        text  = instance["texts"]
        props = instance["prop_lists"]
        outs  = instance["outputs"]
        identifier = instance["guid"] if "guid" in instance else instance["id"]
        try: 
            question = [instance["question"]] if isinstance(instance["question"],str) else instance["question"]
            answer = [instance["answer"]] if isinstance(instance["answer"],str) else instance["answer"]
        except KeyError:
            question = []
            answer   = []
        prop_times = instance["prop_times"] if "prop_times" in instance else []
        constraints = [] if "constraints" not in instance else instance["constraints"]
        neg_breakpoints = [] if "neg_breakpoints" not in instance else instance["neg_breakpoints"]
        pos_breakpoints = [] if "pos_breakpoints" not in instance else instance["pos_breakpoints"]
        sit_breakpoints = [] if "sit_breakpoints" not in instance else instance["sit_breakpoints"]
        story_labels    = [] if "story_labels"    not in instance else instance["story_labels"]
        blocks = [] if "blocks" not in instance else instance["blocks"]
        
        if constraints:
            constraints = parse_basic_implications_conjunctions(constraints)

        return MultiTextPropositionInput(
            guid=identifier,
            texts=texts,
            outputs=outs,
            prop_lists=props,
            question=question,
            answer=answer,
            prop_times=prop_times,
            constraints=constraints,
            neg_breakpoints=neg_breakpoints,
            pos_breakpoints=pos_breakpoints,
            sit_breakpoints=sit_breakpoints,
            story_labels=story_labels,
            blocks=blocks,
        )
    
