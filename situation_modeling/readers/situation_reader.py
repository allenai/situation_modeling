import sys
from .input_example import TextPropositionInput
from .data_reader import JsonlReader

__all__ = [
    "SituationReader"
]

LABELS = {
    "yes"   : 1.0,
    "no"    : 0.0,
    "maybe" : 0.5,
}

class SituationReader(JsonlReader):
    """Reader for situation modeling
    
    >>> from situation_modeling.readers import SituationReader 
    >>> dataset = SituationReader.from_file('etc/examples/situation_data.jsonl')
    >>> len(dataset) 
    1 
    >>> dataset.data 
    [TextPropositionInput(guid='ex01', text=['John walked to the bathroom'], output=[1.0, 0.0, 0.5], prop_list=['John is in the bathroom', 'John is in the kitchen', 'Mary is in the kitchen'])]
    >>> dataset.size 
    1
    >>> dataset[0]
    TextPropositionInput(guid='ex01', text=['John walked to the bathroom'], output=[1.0, 0.0, 0.5], prop_list=['John is in the bathroom', 'John is in the kitchen', 'Mary is in the kitchen'])
    """

    @staticmethod 
    def _read(instance):
        """Reads situation modeling output

        :param instance: the target instance to be read
        :rtype: situation_modeling.readers.input_example.InputBase
        """
        text  = instance["texts"]
        #props = instance["propositions"]
        props = instance["prop_lists"]
        outs  = instance["outputs"]
        identifier = instance["id"]

        ## convert from strings 
        if isinstance(props,str):
            props = props.split(". ")
        elif isinstance(outs,str):
            outs = outs.split("-")
        outs = [float(LABELS.get(v.strip(),v)) for v in outs]

        return TextPropositionInput(
            guid=identifier,
            texts=[text],
            output=outs,
            prop_list=props
        )
