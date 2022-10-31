####

import sys
import itertools
import torch
import uuid
import logging
from .input_example import Text2TextInput
from .data_reader import JsonlReader

util_logger = logging.getLogger('situation_modeling.readers.text_to_text_reader')

class Text2TextReader(JsonlReader):
    """Reader for simple classification problems
    """

    @staticmethod 
    def _read(instance):
        """Reads situation modeling output

        :param instance: the target instance to be read
        :rtype: situation_modeling.readers.input_example.InputBase
        """
        identifier = instance["id"] if "id" in instance else str(uuid.uuid1())

        ### for situation modeling stuff 
        if "prop_lists" in instance:
            if isinstance(instance["question"],str):
                instance["question"] = [instance["question"]]
                instance["answer"] = [instance["answer"]]

            if len(instance["question"]) > 1:
                util_logger.warning(
                    f'Contains multiple questions, picking first one: {instance["question"]}'
                )
                
            ### need to merge with
            # if "full: " not in instance["question"]:
            #     raw_text = '. '.join([t.replace(".","") if t[-1] == "." else t instance["texts"]])
                
            instance["context"] = instance["question"][0]
            instance["answer"] = instance["answer"][0]

            if "full: " not in instance["context"]:
                stext = '. '.join([t.replace(".","") if t[-1] == "." else t for t in instance["texts"]])
                new_context = f"{stext} $query {instance['context']}".strip()
                instance["context"] = new_context
            else:
                instance["context"] = instance["context"].replace("full: ","").strip()

        ### two of the classic formats 
        try: 
            context    = instance["context"]
            answer     = instance["answer"]
        except:
            try: 
                context = instance["question"]["stem"]
                answer = instance["output"]
            except:
                try:
                    context = instance["input"]
                    answer = instance["output"]

                except: 
                    ### patch for situation model-based input  
                    try:
                        story = ' '.join([t+"." if "." not in t else t for t in instance["texts"]])
                        context = f'{story} $question {instance["question"]}'
                        answer = instance["answer"]
                    except KeyError:
                        raise ValueError(
                            'Unrecognizable format: %s' % instance
                        )

        ## prefix
        if "meta" not in instance:
            instance["meta"] = {}
        prefix = instance["meta"]["prefix"] if "prefix" in instance["meta"] else ""
        
        ## add prefix if it exists 
        if prefix:
            context = "%s %s" % (prefix,context)

        return Text2TextInput(
            guid=identifier,
            texts=[context],
            output=[answer],
            prefix=prefix
        )
