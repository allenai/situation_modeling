from typing import Union, List, Set, Optional
from dataclasses import dataclass, field, asdict

__all__ = [
    "LabeledInput",
    "Text2TextInput",
    "TextPropositionInput",
    "SentencePairInput",
    "MultiTextPropositionInput",
    "MultiTextTokenizedPropsInput"
]

@dataclass
class InputBase:
    """Base class for input examples

    :param text: the input text 
    :param guid: the instance identifier (optional)
    """
    guid: str = field(default='')
    texts: List[str] = field(default_factory=list)
    output : List[str] = field(default_factory=list)
    prop_list : List[float] = field(default_factory=list)
    evaluation : bool = False
    prefix : str = ""
    constraints  : List[str] = field(default_factory=list)

    def __str__(self):
        return "<InputExample> output: {}, text(s): {}, prop_list: {}, evaluation={}, prefix={}".format(
            str(self.output), str(self.texts),str(self.prop_list),str(self.evaluation),self.prefix,
        )

    def __post_init__(self):
        ### sanity check after initialization 
        self._check_data()

    def _check_data(self):
        """Performs any checking needed for the datasets. Passes 
        by default.

        :rtype: None 
        :raises: ValueError 
        """
        pass

    @classmethod
    def from_json(cls,json_instance):
        texts = json_instance["texts"]
        guid = json_instance["id"] if "id" in json_instance else None
        output = json_instance["output"] if "output" in json_instance else None
        prop_list = json_instance["prop_list"] if "prop_list" in json_instance else None
        return cls(guid,texts,output,prop_list)


class LabeledInput(InputBase):
    """Class for labeled input

    :param text: the input text 
    :param guid: the instance identifier (optional)
    :param output: the output label
    """
    def _check_data(self):
        """Checks that the label output contains only a single label
        
        :rtype: None 
        :raises: ValueError 
        """
        if len(self.output) != 1:
            raise ValueError('Label output is too big!')

class SentencePairInput(LabeledInput):

    def _check_data(self):
        """Check that text input is equal to size 2

        :raises: ValueError 
        :rtype: None 
        """
        if len(self.texts) != 2:
            raise ValueError('Text input must be of size 2!')

class Text2TextInput(InputBase):
    """Class for labeled input

    :param text: the input text 
    :param guid: the instance identifier (optional)
    :param output: the output text 
    """
    def _check_data(self):
        """Check that the ouptut is a single text

        :rtype: None 
        :raises: ValueError 
        """
        if len(self.output) != 1:
            raise ValueError('Text output is greater than 1')

class TextPropositionInput(InputBase):
    """Class for labeled input

    :param text: the input text 
    :param guid: the instance identifier (optional)
    :param output: the output labels 
    :param prop_list: the list of the text propositions 

    >>> from situation_modeling.datasets.input_example import *
    >>> t = TextPropositionInput(text=["John moved to the kitchen"],prop_list=["John is in the kitchen","John is in the hallway","Mary is in the kitchen"],output=[1.0,0.0,0.4])
    """
    output : List[float] = field(default_factory=list)

    def _check_data(self):
        """Checks that the output and proposition list representations are equal 
        in size 

        
        :rtype: None 
        :raises: ValueError 
        """
        if len(self.output) != len(self.prop_list):
            raise ValueError('Output and proposition list are of difference sizes!')

@dataclass
class MultiTextPropositionInput:
    """Base class for multi sentence input examples

    :param texts: the input text 
    :param guid: the instance identifier (optional)
    :param outputs: list of list of output belief probabilities 
    :param prop_lists: list of list of the corresponding text propositions 
    
    >>> from situation_modeling.datasets.input_example import *
    >>> t = MultiTextPropositionInput(texts=['Julie journeyed to the kitchen.', 'Afterwards she moved to the school.'],
    prop_list=prop_lists=[['Julie at kitchen', 'Julie at hallway', 'Julie at garden'], ['Julie at school', 'Julie at park']],outputs=[[1.0, 0.0, 0.0], [1.0, 0.0]])
    """
    guid: str = field(default='')
    texts: List[str] = field(default_factory=list)
    outputs : List[List[float]] = field(default_factory=list)
    prop_lists : List[List[str]] = field(default_factory=list)
    question : List[str] = field(default_factory=list)
    answer : List[str] = field(default_factory=list)
    prop_times: Optional[List[List[int]]] = field(default_factory=list)
    evaluation : bool = False
    constraints  : List[str] = field(default_factory=list)
    pos_breakpoints : List[str] = field(default_factory=list)
    neg_breakpoints : List[str] = field(default_factory=list)
    sit_breakpoints : List[str] = field(default_factory=list)
    story_labels : List[str] = field(default_factory=list)
    blocks : List[str] = field(default_factory=list)

    def __str__(self):
        return "<InputExample> outputs: {}, texts: {}, prop_lists: {}, question: {}, answer: {}, evaluation: {}, blocks: {}".format(
            str(self.outputs), "; ".join(self.texts), str(self.prop_lists),self.question,self.answer,self.evaluation,self.blocks
        )

    def __post_init__(self):
        ### sanity check after initialization 
        self._check_data()
        
    def _check_data(self):
        """Checks that the output and proposition list representations are equal 
        in size 

        :rtype: None 
        :raises: ValueError 
        """
        if len(self.outputs) != len(self.prop_lists):
            raise ValueError(
                'Outputs and proposition lists are of different sizes!'
            )

        output_lens = [len(s) for s in self.outputs]
        prop_lens = [len(s) for s in self.prop_lists]

        if output_lens != prop_lens:
            raise ValueError('Outputs and proposition sub-lists are of different sizes!')

    @classmethod
    def from_json(cls,json_instance):
        texts = json_instance["texts"]
        guid = json_instance["id"] if "id" in json_instance else None
        outputs = json_instance["outputs"] if "outputs" in json_instance else []
        prop_lists = json_instance["prop_lists"] if "prop_lists" in json_instance else []
        prop_times = json_instance["prop_times"] if "prop_times" in json_instance else None
        question = json_instance["question"] if "question" in json_instance else []
        answer = json_instance["answer"] if "answer" in json_instance else []
        blocks = json_instance["blocks"] if "blocks" in json_instance else []

        ### constraints
        if "constraints" in json_instance:
            pass

        return cls(
            guid=guid,
            texts=texts,
            outputs=outputs,
            prop_lists=prop_lists,
            prop_times=prop_times,
            question=question,
            answer=answer,
            evaluation=True,
            blocks=blocks
        )
    
    def to_dict(self):
        return asdict(self)

class MultiTextTokenizedPropsInput(MultiTextPropositionInput):
    """Class for multi sentence propsition labeled inputs, where propositions
    are mapped to an integer (for storage efficiency).

    :param texts: the input texts
    :param guid: the instance identifier (optional)
    :param outputs: the output labels 
    :param prop_lists: list of list of the text propositions, where each proposition
    is mapped to a unique integer.

    """
    prop_lists : List[List[int]] = field(default_factory=list)
