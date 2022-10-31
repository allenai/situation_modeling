import os
from typing import Dict, List
import sys
from pathlib import Path
import json
from dataclasses import asdict
from torch.utils.data import Dataset

from .input_example import MultiTextPropositionInput
from .mult_sent_sit_reader import MultiSentTokPropsSituationReader, MultiSentSituationReader


__all__ = [
    "MultiSentSituationsDataset"
]

PROB_LABEL_MAP = {
    1.0: 2,
    0.5: 1,
    0: 0,
    0.: 0,
    "yes"   : 2,
    "no"    : 0,
    "maybe" : 1,
}

def untokenize_props(props_list: List[List[int]], prop_map: Dict[int, str]) -> List[List[str]]:
    """
    Convert propositions from tokenized format to string format.

    :param props_list: [description]
    :type props_list: List[List[int]]
    :param prop_map: [description]
    :type prop_map: Dict[int, str]
    :return: List of lists of propositions in string form
    :rtype: List[List[str]]
    """
    return [[prop_map.get(p, p) for p in props] for props in props_list]

class MultiSentSituationsDataset(Dataset):
    """
    Subclass of torch Dataset. Handles loading and yielding MultiInputBase
    examples from a jsonl file of instances. For memory efficiency, instances may 
    optionally encode proposition lists using a unique integer id for each proposition type.
    In this case a mapping between propositions and their unique id must be provided.

    """
    def __init__(
            self,
            root: str,
            split: str = "train",
            prop_map: str = "prop_map.json",
            config = None,
            evaluate = False,
        ) -> None:
        """Dataset constructor.

        :param root: path to folder where data and optionally mapping reside.
        :type root: str
        :param split: split name to be loaded, defaults to "train"
        :type split: str, optional
        :param prop_map: name of file containing proposition mapping, defaults to "prop_map.json".
        If None, no mapping will be used- assumed propositions are already in string form.
        :type prop_map: str, optional
        :param evaluate: run the loader in evaluation model
        :type evaluate: bool,optional
        """
        super().__init__()

        self.split = split
        self.config = config

        self.root_path = Path(root)

        # load proposition map
        self.prop_map = None
        if prop_map:
            prop_map_path = self.root_path / prop_map
            if os.path.isfile(prop_map_path):
                self.prop_map = json.load(prop_map_path.open())
                # convert keys to int
                self.prop_map = {int(k): v for k,v in self.prop_map.items()}

        # load data
        data_path = self.root_path / (f"{split}.jsonl")
        if self.prop_map:
            self.data = MultiSentTokPropsSituationReader.from_file(
                str(data_path),
                config=self.config,
                evaluate=evaluate
            )
        else:
            self.data = MultiSentSituationReader.from_file(
                str(data_path),
                config=self.config,
                evaluate=evaluate
            )
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> MultiTextPropositionInput:
        instance = self.data[index]
        if self.prop_map:
            instance_dict = asdict(instance)
            instance_dict["prop_lists"] = untokenize_props(instance_dict["prop_lists"], self.prop_map)
            inp = MultiTextPropositionInput(**instance_dict)
        else:
            # no need to untokenize props
            inp = instance

        # convert probabilities to labels
        if self.classify_mode:
            inp.outputs = [[PROB_LABEL_MAP.get(p) for p in l] for l in inp.outputs]
        else:
            inp.outputs = [[float(p) for p in l] for l in inp.outputs]
        return inp


    @property
    def classify_mode(self) -> bool:
        """
        Return True if data should be read in classification mode (probabilites mapped to integer labels).
        """
        return self.config and self.config.class_output 


class MultiSentSituationsDummyDataset:
    """
    Dummy version only used for calculating dataset statistics.

    """
    def __init__(
            self,
            root: str,
            split: str = "train",
            prop_map: str = "prop_map.json",
            config = None,
            evaluate = False,
        ) -> None:
        """Dataset constructor.

        :param root: path to folder where data and optionally mapping reside.
        :type root: str
        :param split: split name to be loaded, defaults to "train"
        :type split: str, optional
        :param prop_map: name of file containing proposition mapping, defaults to "prop_map.json".
        If None, no mapping will be used- assumed propositions are already in string form.
        :type prop_map: str, optional
        :param evaluate: run the loader in evaluation model
        :type evaluate: bool,optional
        """

        self.split = split
        self.config = config

        self.root_path = Path(root)

        # load proposition map
        self.prop_map = None
        if prop_map:
            prop_map_path = self.root_path / prop_map
            if os.path.isfile(prop_map_path):
                self.prop_map = json.load(prop_map_path.open())
                # convert keys to int
                self.prop_map = {int(k): v for k,v in self.prop_map.items()}

        # load data
        data_path = self.root_path / (f"{split}.jsonl")
        if self.prop_map:
            self.data = MultiSentTokPropsSituationReader.from_file(
                str(data_path),
                config=self.config,
                evaluate=evaluate
            )
        else:
            self.data = MultiSentSituationReader.from_file(
                str(data_path),
                config=self.config,
                evaluate=evaluate
            )


    def get(self, index):
        instance = self.data[index]
        if self.prop_map:
            instance_dict = asdict(instance)
            instance_dict["prop_lists"] = untokenize_props(instance_dict["prop_lists"], self.prop_map)
            inp = MultiTextPropositionInput(**instance_dict)
        else:
            # no need to untokenize props
            inp = instance
        
        # convert probabilities to labels
        if self.classify_mode:
            inp.outputs = [[PROB_LABEL_MAP.get(p) for p in l] for l in inp.outputs]
        else:
            inp.outputs = [[float(p) for p in l] for l in inp.outputs]
        return inp


    @property
    def classify_mode(self) -> bool:
        """
        Return True if data should be read in classification mode (probabilites mapped to integer labels).
        """
        return self.config and self.config.class_output 
