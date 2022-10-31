import sys
import itertools
import torch
from .input_example import LabeledInput
from .data_reader import JsonlReader

class ClassificationReader(JsonlReader):
    """Reader for simple classification problems
    """

    @staticmethod 
    def _read(instance):
        """Reads situation modeling output

        :param instance: the target instance to be read
        :rtype: situation_modeling.readers.input_example.InputBase
        """
        context    = instance["context"]
        answer     = instance["answer"]
        identifier = instance["id"]

        return LabeledInput(
            guid=identifier,
            texts=[context],
            output=[answer]
        )
