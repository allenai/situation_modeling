from typing import List
import os
import logging
import json
import torch
import itertools
import numpy as np
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer, AutoConfig

from ..base import ConfigurableClass
from ..readers import TextPropositionInput, MultiTextPropositionInput
from sentence_transformers import SentenceTransformer
from optparse import OptionParser,OptionGroup
from typing import List, Dict, Optional, Union, Tuple, Callable
from .module_base import Model
from ..readers.situation_reader import SituationReader

from ..readers.mult_sent_sit_dataset import MultiSentSituationsDataset, MultiSentSituationsDummyDataset
from torch import Tensor
from torchtyping import TensorType
from .transformer_model import TransformerModel
from .sit_prop_pooler import create_gather_idx_tensor

util_logger = logging.getLogger('situation_modeling.base_modules.situation_encoder')


def insert_sit_tokens(texts: List[str], sit_token: str) -> str:
    """
    Insert special SIT token after each sentence in list of texts.
    """
    return "".join([t + sit_token for t in texts])

def insert_prop_token(prop: str, prop_token: str) -> str:
    """
    Insert special PROP token before each sentence in list of texts.
    """
    return prop_token + prop

##########################
# DATA READER FUNCTIONS  #
##########################

def simple_situation_loader(
        data_path : str,
        split : str,
        config=None,
        evaluate=False,
        tokenizer=None,
    ) -> List:
    """A data loader for the simple `simple` (i.e., non-time related) situation
    datasets.

    :param data_path: the target data path or directory
    :param split: the target split, e.g., `train`, `dev`, `test`
    :rtype: list
    """
    target_data = os.path.join(data_path,split+".jsonl")
    util_logger.info('Reading data from {}'.format(target_data))

    data_container = SituationReader.from_file(target_data,evaluate=evaluate)

    return data_container.data

def simple_situation_instance_loader(data_instance : str,evaluate=False,tokenizer=None):
    """Loads an instance from string

    :param str_instance: the particular instance
    """
    prop_list = []
    try:
        json_line = json.loads(data_instance.replace("\n","").strip())
        str_list = json_line["texts"]
        if "prop_list" in json_line:
            prop_list = json_line["prop_list"]
    except ValueError as e:
        if isinstance(data_instance,str):
            str_list = [data_instance]
        elif isinstance(data_instance,list):
            str_list = data_instance
    input_instance = TextPropositionInput(
        texts=str_list,
        prop_list=prop_list,
        output=[1.]*len(prop_list),
    )
    return input_instance

def temporal_situation_loader(
        data_path : str,
        split : str, 
        config=None,
        evaluate=False,
        tokenizer=None,
    ) -> List:
    """A data loader for the simple `simple` (i.e., non-time related) situation
    datasets.

    :param data_path: the target data path or directory
    :param split: the target split, e.g., `train`, `dev`, `test`
    """
    util_logger.info('Reading data from {}, split: {}'.format(data_path, split))
    util_logger.info(
        f'batch_size={config.train_batch_size},max_seq={config.max_seq_length},max_prop={config.max_prop_length}'
    )

    # Load dummy dataset to avoid multi-process iteration issues with PyTorch Datasets
    # dummy has same basic functionality but is a simple iterator (not PyTorch Dataset)
    # we use it just for stats collection
    data_loader = MultiSentSituationsDummyDataset(data_path, split, config=config)

    ### use tokenizer here to look at truncation
    #### sanity check here
    if tokenizer is not None and config is not None:
        util_logger.info(f'Sanity checking the data and counting truncation for {split}...')
        input_lengths = []
        prop_lengths = []
        total_inputs = 0
        total_props = 0
        total_questions = 0
        question_lenghts = []

        ### https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        for i in range(len(data_loader.data)):
            instance = data_loader.get(i)

            #for dinput in instance.texts:
            #   assert isinstance(dinput,str),"wrong text format"
            input_w_sit_tokens = insert_sit_tokens(instance.texts, sit_token=config.sit_token)
            total_inputs += len(instance.texts)
            input_lengths.append(len(tokenizer.tokenize(input_w_sit_tokens)))

            for p_list in instance.prop_lists:
                for p in p_list:
                    total_props += 1
                    input_w_prop_token = insert_prop_token(p, config.prop_token)
                    prop_lengths.append(len(tokenizer.tokenize(input_w_prop_token)))


        ## log truncation
        num_over = len([l for l in input_lengths if l > config.max_seq_length])
        util_logger.info(
            f"input stats: max={np.max(input_lengths)},mean={np.mean(input_lengths)}, truncate={num_over} / {total_inputs}"
        )
        prop_over = len([l for l in prop_lengths if l > config.max_prop_length])
        util_logger.info(
            f"props stats: max={np.max(prop_lengths)},mean={np.mean(prop_lengths)}, truncate={prop_over} / {total_props}"
        )
        del os.environ["TOKENIZERS_PARALLELISM"]

    # delete dummy dataset and create actual PyTorch Dataset
    del data_loader
    dataset = MultiSentSituationsDataset(data_path, split, config=config)

    return dataset

def temporal_situation_instance_loader(query_input,**kwargs):
    """Loads an instance from json or raw input 

    :param query_input: the main input
    :param kwargs: the additional args needed
    :raises: ValueError 
    """
    json_instance = {}
    try:
        json_instance = json.loads(str(query_input)) if not isinstance(query_input,dict) else query_input
    except ValueError as e:

        if isinstance(query_input,str):
            texts = [t.strip() for t in query_input.split(". ")]
        elif isinstance(query_input,list):
            texts = query_input
        else:
            raise ValueError(
                'First input must be a string or list'
            )
        json_instance["texts"] = texts
        json_instance.update(kwargs)

        prop_lists = kwargs["prop_lists"]
        outputs = kwargs.get("outputs",[])
        
        if len(prop_lists) != len(texts):
            raise ValueError(
                'Story input and prop lists have different lengths!'
            )

    except KeyError as e:
        raise ValueError(
            'Input not well-formed! {}'.format(e)
        )


    ## create fake outputs
    if "outputs" not in json_instance or not json_instance["outputs"]:
        outputs = [[0.5]*len(l) for l in json_instance["prop_lists"]]
        json_instance["outputs"] = outputs
    if "question" not in json_instance:
        json_instance["question"] = []
    if "answer" not in json_instance:
        json_instance["answer"] = []

    return MultiTextPropositionInput.from_json(json_instance)


###############################
# DEFAULT COLLATOR FUNCTIONS  #
###############################

# def simple_situation_collator(
#         batch : List,
#         tokenizer,
#         max_seq_length : int,
#         max_prop_length : int = None,
#         sit_token: str = None,
#         prop_token: str = None
#     ) -> Tuple:
#     """A collator (i.e., batch featurizer) that works for simpe situation

#     :param batch: a list of data instances
#     :rtype: tuple
#     """

#     main_inputs = list(itertools.chain(*[t.texts for t in batch]))
#     batch_size = len(batch)

#     ## input size features
#     max_props = max([len(p.prop_list) for p in batch])
#     global_props = []
#     global_mask  = []
#     global_attn_mask = []
#     output_labels = []

#     if max_prop_length is None: max_prop_length = max_seq_length

#     for item in batch:

#         prop_list = item.prop_list
#         plen = len(prop_list)
#         prop_list += ["empty"]*(max_props-plen)
#         assert len(prop_list) == max_props
#         pmask = [0. if p == "empty" else 1. for p in prop_list]

#         ### labels
#         outputs = item.output
#         assert len(outputs) == plen,"mismatched prop list and output list"
#         outputs += [0.]*(max_props-plen)
#         output_labels.append(outputs)

#         out_features = tokenizer(
#             prop_list,
#             max_length=max_prop_length,
#             padding='max_length',
#             return_tensors="pt",
#             truncation=True
#         )
#         global_props.append(out_features["input_ids"])
#         global_attn_mask.append(out_features["attention_mask"])
#         global_mask.append(torch.tensor(pmask))

#     main_features = tokenizer(
#         main_inputs,
#         max_length=max_seq_length,
#         padding='max_length',
#         return_tensors="pt",
#         truncation=True
#     )

#     main_features["input_ids"] = main_features["input_ids"].unsqueeze(1)
#     main_features["attention_mask"] = main_features["attention_mask"].unsqueeze(1)

#     prop_rep = torch.stack(global_props).unsqueeze(1)
#     prop_attn_mask = torch.stack(global_attn_mask).unsqueeze(1)
#     prop_mask = torch.stack(global_mask).unsqueeze(1)

#     ### add to main features
#     main_features["prop_input_ids"] = prop_rep
#     main_features["prop_attention_mask"] = prop_attn_mask
#     main_features["prop_mask"] = prop_mask

#     ### output tensor representations
#     #outputs = torch.tensor([o.output for o in batch]).unsqueeze(1)
#     outputs = torch.tensor(output_labels).unsqueeze(1).float()
#     assert outputs.shape == prop_mask.shape,"outputs and masks wrong shape"

#     return (main_features, outputs)


### why not add full `config` here?

def select_random_situations(batch):
    """Randomly select situations to mask out or generate during training 

    :param batch: the target batch 
    """
    random_indices = [np.random.choice(len(b.texts)) for b in batch]
    return [
        ((n,random_indices[n]),(n,random_indices[n]),"same",b.texts[random_indices[n]]) \
        for n,b in enumerate(batch)
    ]

def featurize_stories_propositions(batch,tokenizer,config):
    """Get basic representation of stories and proposition lists 

    :param batch: the incoming batch 
    :param config. 
    """
    sit_token = config.sit_token
    prop_token = config.prop_token
    max_seq_length = config.max_seq_length
    max_prop_length = config.max_prop_length
    
    ### check if an evaluation batch 
    evaluate = [e.evaluation for e in batch][0]

    ##### ENCODE STORY INPUT
    ## 
    main_inputs = [
        insert_sit_tokens(inst.texts, sit_token) for inst in batch
    ]

    main_features = tokenizer.batch_encode_plus(
        main_inputs,
        max_length=max_seq_length,
        padding='max_length',
        return_tensors="pt",
        truncation=True
    )
    #main_features["input_ids"] = main_features["input_ids"].unsqueeze(1)
    #main_features["attention_mask"] = main_features["attention_mask"].unsqueeze(1)

    #### ENCODE PROPOSITIONS
    ## input size features

    ### modified slightly, now have cases with zero props, extra data for [SIT] generation pre-training
    max_props = max(1,max([max([len(p) for p in inst.prop_lists]) for inst in batch]))
    max_timesteps = max(1,max([len(inst.prop_lists) for inst in batch]))
    batch_size = len(batch)
    global_props = []
    global_mask  = []
    global_attn_mask = []
    global_output_labels = []

    # old 
    #global_prop_map = {}
    #prop_counter = 0

    for b,item in enumerate(batch):
        all_inp_props = []
        all_inp_mask  = []
        all_inp_attn_mask = []
        all_output_labels = []
        prop_lists = item.prop_lists

        # pad prop_lists to max over all prop_lists in batch
        p_lists_len = len(prop_lists)
        prop_lists += [["empty"]]*(max_timesteps-p_lists_len)

        for w,prop_list in enumerate(prop_lists):
            plen = len(prop_list)
            prop_list += ["empty"]*(max_props-plen)
            pmask = [0 if p == "empty" else 1 for p in prop_list]

            # add prop special token for proposition represenation summary
            prop_list = [insert_prop_token(p, prop_token) for p in prop_list]
            assert len(prop_list) == max_props

            out_features = tokenizer.batch_encode_plus(
                prop_list,
                max_length=max_prop_length,
                padding='max_length',
                return_tensors="pt",
                truncation=True
            )
            all_inp_props.append(out_features["input_ids"])
            all_inp_attn_mask.append(out_features["attention_mask"])
            all_inp_mask.append(torch.tensor(pmask))

            ####
            # for o,prop in enumerate(prop_list):
            #     global_id = f"{w}_{o}_{b}"
            #     global_prop_map[global_id] = prop_counter
            #     prop_counter += 1

        ### labels
        outputs = item.outputs
        o_lists_len = len(outputs)
        outputs += [[0]]*(max_timesteps-o_lists_len)

        assert len(outputs) == len(prop_lists),"mismatched prop lists and output lists"

        for outputs_list in outputs:
            olen = len(outputs_list)
            outputs_list += [0]*(max_props-olen)
            assert len(outputs_list) == max_props

            all_output_labels.append(torch.tensor(outputs_list))

        global_output_labels.append(torch.stack(all_output_labels))
        global_props.append(torch.stack(all_inp_props))
        global_attn_mask.append(torch.stack(all_inp_attn_mask))
        global_mask.append(torch.stack(all_inp_mask))

    #### proposition encodings

    prop_rep = torch.stack(global_props)
    prop_attn_mask = torch.stack(global_attn_mask)
    prop_mask = torch.stack(global_mask)

    ### add to main features
    main_features["prop_input_ids"] = prop_rep
    main_features["prop_attention_mask"] = prop_attn_mask
    main_features["prop_mask"] = prop_mask

    ### output tensor representations
    output_tensor = torch.stack(global_output_labels)

    main_features["evaluate"] = evaluate

    return (main_inputs,main_features,output_tensor)

def find_sit_gen_targets(batch,tokenizer,config,main_features):
    """Randomly picks situation tokens and their corresponding (left) events to 
    generate from.

    :param batch: the incoming batch 
    :param tokenizer: the model tokenizer 
    :param config: the global model and experiment configuration
    :param main_features: the main input features
    """
    max_output_length = config.max_output_length
    sit_token_id = config.sit_token_id
    
    sit_gen_locations = [np.random.choice(len(b.texts)) for b in batch]
    sit_gen_texts = [b.texts[sit_gen_locations[n]] for n,b in enumerate(batch)]
    sit_gen_locations = [
        ((bi,b),(bi,b),"same") for bi,b in enumerate(sit_gen_locations)
    ]

    sit_locations = torch.tensor([
        [
        b1,[n for n,t in enumerate(main_features["input_ids"][b1]) if t == sit_token_id][l1],
        b2,[n for n,t in enumerate(main_features["input_ids"][b2]) if t == sit_token_id][l2]
        ] for m,((b1,l1),(b2,l2),command) in enumerate(sit_gen_locations)
    ])

    commands = [c[-1] for c in sit_gen_locations]
    sit_outputs = tokenizer.batch_encode_plus(
            sit_gen_texts,
            max_length=max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
    )
    main_features["sit_locations"] = sit_locations
    main_features["sit_commands"] = commands
    main_features["sit_outputs"]  = sit_outputs["input_ids"]

    return (sit_locations,commands,sit_outputs)

class BatchConstraintList(object):
    def __init__(self,batch_constraints):
        self._batch_constraints = batch_constraints 

    def left_right_rules(self):
        for batch in self._batch_constraints:
            for constraint in batch: 
                yield constraint.left_right_values()

    def __iter__(self):
        return self.left_right_rules()

    def __len__(self):
        return len(self._batch_constraints)

    def __bool__(self):
        return len(self._batch_constraints) > 0

    @property 
    def constraint_formula(self):
        for batch in self._batch_constraints:
            yield batch.generate_formula()
    
            
class InstanceConstraintList(list):

    def generate_formula(self):
        ## left side
        var_map = {}
        clauses = []

        ### enumerate constaints
        for constraint in self:
            name,formula = constraint.cnf_expr()
            
            for clause in formula:
                clause_rep = []
                for (polarity,index) in clause:
                    if index not in var_map:
                        var_map[index] = len(var_map) + 1
                    ### add to global clauses
                    clause_rep.append((polarity,var_map[index]))
                clauses.append((name,clause_rep))

        return (clauses,var_map)
    
def parse_constraints(batch):
    """Parse symbolic constraints if specified in batch

    :param batch: the incoming batch 
    :returns: list of global batch constraints in list form 
    :rtype: list 
    """
    global_constraints = []

    batch_constraints = {}
    
    for bid,b in enumerate(batch):
        instance_constraints = b.constraints
        batch_constraints[bid] = []
        
        for constraint in instance_constraints:
            constraint.batch_id = bid
            batch_constraints[bid].append(constraint)

            # left = [
            #     [int(i) for i in idx.replace("~","").split("_")]+[0] if "~" in idx else\
            #     [int(i) for i in idx.replace("~","").split("_")]+[2] \
            #     for idx in constraint.antecedent
            # ]
            # right = [
            #     [int(i) for i in idx.replace("~","").split("_")]+[0] if "~" in idx else\
            #     [int(i) for i in idx.replace("~","").split("_")]+[2] \
            #     for idx in constraint.consequent
            # ]
            # global_constraints.append((constraint.name,constraint.operator_type,left,right))

    return BatchConstraintList([InstanceConstraintList(c) for c in batch_constraints.values()])

def parse_questions_answers(batch,tokenizer,config,main_features):
    """Parse the provided questions and answers in the batch 

    :param batch: the incoming batch 
    :param config: the global config
    """
    sit_token = config.sit_token
    prop_token = config.prop_token
    max_seq_length = config.max_seq_length
    max_prop_length = config.max_prop_length
    gen_param = config.gen_param
    no_gen = config.no_generation
    max_output_length = config.max_output_length

    main_raw_inputs = [
        insert_sit_tokens(inst.texts, sit_token) for inst in batch
    ]
    questions = list(itertools.chain(*[
        ["%s $question %s" % (main_raw_inputs[k],q) for q in w.question] for k,w in enumerate(batch)
    ]))
    answers = list(
        itertools.chain(*[q.answer for q in batch])
    )

    if questions and gen_param > 0.0 and no_gen is False:
        if answers: 
            assert len(questions) == len(answers), "mismatched question and answer length!"

        question_inputs = tokenizer(
            questions,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        main_features["source_ids"] = question_inputs["input_ids"].squeeze(-1)
        main_features["source_mask"] = question_inputs["attention_mask"].squeeze(-1)
        if answers: 
            answer_outputs = tokenizer(
                answers,
                max_length=max_output_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )
            main_features["target_ids"] = answer_outputs["input_ids"].squeeze(-1)
            main_features["target_mask"] = answer_outputs["attention_mask"].squeeze(-1)
        
    return (questions,answers)
        

def temporal_situation_collator(
        batch : List,
        tokenizer,
        config,
    ):
    """[summary]
    :param batch: Batch of data instances
    :type batch: List
    :param tokenizer: Model tokenizer
    :type tokenizer: [type]
    """
    ### featurize basic batch input 
    main_inputs,main_features,output_tensor = featurize_stories_propositions(batch,tokenizer,config)

    ### parse batch symbolic constraints
    global_constraints = parse_constraints(batch)
    main_features["constraints"] = global_constraints

    ### find situation generation token locations (optional) 
    if config.sit_gen:
        sit_locations,commands,sit_outputs = find_sit_gen_targets(batch,tokenizer,config,main_features)

    ### parse questions and answers (if provided)
    questions,answers = parse_questions_answers(batch,tokenizer,config,main_features)

    ## unsqueeze the final input and attention mask tensors
    main_features["input_ids"] = main_features["input_ids"].unsqueeze(1)
    main_features["attention_mask"] = main_features["attention_mask"].unsqueeze(1)

    features = {}
    ## `main_features` has funny features related to tokenizer
    features.update(main_features)

    ## should you do an evaluation and run generation with this batch?
    features["gen"] = True if (config.gen_param > 0.0 and features["evaluate"] and questions) else False

    ## extras for printing, only turned on if the batch is an evaluation batch 
    if features["evaluate"] is True: 
        features.update({
            "print_out" : {
                "sit_input"  : main_inputs,
                "text_in"    : questions,
                "text_out"   : answers,
                "prop_lists" : [i.prop_lists for i in batch],
                "guid"       : [i.guid if i.guid else str(k) for k,i in enumerate(batch)],
            },
        })

    return features, output_tensor
    


### factories

_READERS = {
    "simple_situation" : (
        simple_situation_loader,
        simple_situation_instance_loader
    ),
    "temporal_situation" : (
        temporal_situation_loader,
        temporal_situation_instance_loader,
    )
}

_COLLATORS = {
    #"simple_situation_collator"   : simple_situation_collator,
    "temporal_situation_collator" : temporal_situation_collator,
}

def CollateFunction(config):
    """Factory method for finding a default pre-build collocator function

    :param config: the global configuration
    :raises: ValueError
    """
    dcollator = _COLLATORS.get(config.data_collator,None)
    if dcollator is None:
        raise ValueError(
            'Unknown collator: %s, current collators available=%s' \
            % (config.data_collator,', '.join(_COLLATORS))
        )
    return dcollator

def ReaderFunction(config):
    """Factory method for finding a default pre-build datarederds

    :param config: the global experiment configuration
    :raises: ValueError
    """
    dreader = _READERS.get(config.data_builder,None)
    if dreader is None:
        raise ValueError(
            'Unknown data reader: %s, current readers available=%s' \
            % (config.data_builder,', '.join(_READERS))
        )
    return dreader

def _aggregate_sit_tokens(hidden_states,atype):
    """Performs an aggregation on the target situation tokens ()

    :param hidden_states: torch tensor of dim (2,dim_size) containing two [SIT] token hiddens states 
    :param atype: the type of aggregation to use, `mean`, `same` (=`mean`), `diff`, 
    :raises: ValueError 
    """
    if atype == "same":
        return hidden_states[0]
    elif atype == "mean":
        return torch.mean(hidden_states,0)
    elif atype == "mult" or atype == "and":
        return hidden_states[0,:] * hidden_states[1,:]
    elif atype == "diff" or atype == "sub":
        return torch.abs(hidden_states[0,:] - hidden_states[1,:])
    else: 
        raise ValueError(
            f'Unknown agg type: {atype}'
        )

def _prefix_map(d):
    """Returns a map of one-hot prefix vectors 

    :rtype: dict 
    """
    return {
        "mult" : torch.tensor([0 if i != 0 else 1 for i in range(d)],requires_grad=False),
        "mean" : torch.tensor([0 if i != 1 else 1 for i in range(d)],requires_grad=False),
        "sub"  : torch.tensor([0 if i != 2 else 1 for i in range(d)],requires_grad=False),
        "same" : torch.tensor([0 if i != 3 else 1 for i in range(d)],requires_grad=False),
    }
    

class SituationEncoder(TransformerModel):
    """Base `TransformerModel` class. The main job of this class is to
    generate embeddings for text, as well as read certain types of text.
    """

    @property
    def prefix_vectors(self):
        """For [SIT] generation, computes one-hot prefix representations to inform the generator 

        :returns: dict 
        """
        try:
            return self._prefixes
        except AttributeError:
            dim = self.get_word_embedding_dimension()
            prefixes = _prefix_map(dim)
            self._prefixes = prefixes
            return prefixes 

    def forward(self, features : Dict[str, Tensor], labels: Tensor = None):
        """Returns embeddings for provided input in `features`. Technically, will add
        tokens embeddings, CLS token embeddings, etc, to the `features` input to be used with
        other modules.

        :note: this is repeated here from `SentenceTransformers.models.Transformer` just to see how it is
        implemented. Different variants here can be created as needed (e.g., a version that returns decoder
        output in the case of a transformer with decoder).
        """
        ## what the input should include
        input_ids : TensorType["B","1","L"] = features["input_ids"]
        attention_mask : TensorType["B","1","L"] = features["attention_mask"]
        B,S,L = input_ids.shape
        D = self.get_word_embedding_dimension()

        ### run encoder on inputs
        trans_features = {
            "input_ids"      : input_ids.squeeze(1),
            "attention_mask" : attention_mask.squeeze(1),
        }
        output_states = self.run_encoder(trans_features)

        metrics = {}

        ## maybe selectively add cls versus full token reps by knowing the
        ## type of subsequence pooling being used
        output_tokens = output_states[0]

        if "sit_swaps" in features:
            nswaps = 0.
            for (b1,l1,b2,l2) in features["sit_swaps"]:
                sit_rep_1 = output_tokens[b1][l1].clone()
                sit_rep_2 = output_tokens[b2][l2].clone()
                ## swap
                output_tokens[b2][l2] = sit_rep_1
                output_tokens[b1][l1] = sit_rep_2
                nswaps += 1.
            #metrics["num_swaps"] = torch.tensor([nswaps])

        ### [SIT] generation loss, re-use the computed encoder hidden states 
        if "sit_locations" in features and "sit_outputs" in features:
            sit_locations = features["sit_locations"]
            commands = features["sit_commands"]

            ### 
            sit_hidden_states = torch.stack(
                [
                    _aggregate_sit_tokens(
                        torch.stack([
                            output_tokens[b1].index_select(0,l1),
                            output_tokens[b2].index_select(0,l2)
                        ]),
                        commands[b]
                    ) \
                    for b,(b1,l1,b2,l2) in enumerate(sit_locations)
                ]).to(output_tokens.device)

            if self.global_config.add_gen_prefix:
                command_map = self.prefix_vectors
                prefix_vectors = torch.stack([command_map[c] for c in commands]).unsqueeze(1).to(output_tokens.device)
                sit_hidden_states = torch.cat([sit_hidden_states,prefix_vectors],dim=1).to(output_tokens.device)

            ### the outputs to predict 
            sit_lm_labels = features["sit_outputs"]
            sit_lm_labels[sit_lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

            ### requires a decoder, should specify 
            sit_gen_loss = self.model(
                encoder_outputs=(sit_hidden_states,),
                labels=sit_lm_labels,
            )

            features["gen_loss"] = sit_gen_loss.loss
            metrics["num_gens"] = torch.tensor([float(len(commands))])

            ### log the amount of generation being done
                        

        features.update({
            "token_embeddings"     : output_tokens.unsqueeze(1),
            "attention_mask"       : features["attention_mask"], ### why?
            "metrics"              : metrics,
        })

        ### run encoder on propositions
        prop_ids : TensorType["B","T","P","L2"] = features["prop_input_ids"]
        prop_mask : TensorType["B","T","P","L2"] = features["prop_attention_mask"]
        _,T,P,L2 = prop_ids.shape ##<-- different length, set at `self.max_prop_length`
        NP = int(B*T*P)
        prop_ids_flat  : TensorType["NP","L"] = prop_ids.view(-1).reshape(NP,L2)
        prop_mask_flat : TensorType["NP", "L"] = prop_mask.view(-1).reshape(NP,L2)

        prop_features = {
            "input_ids"      : prop_ids_flat,
            "attention_mask" : prop_mask_flat,
        }
        prop_states = self.run_encoder(prop_features)
        prop_output_tokens = prop_states[0]
        #prop_cls_tokens = prop_output_tokens[:,0,:].reshape(B,T,P,D)

        prop_output_tokens : TensorType["B","T","P","L2","D"] =\
          prop_output_tokens.reshape(B,T,P,L2,D)

        ### decoder pooling
        if self.global_config.decoder_pooling:

            ## implements the poolingin `sit_prop_pooler`; should be merged 
            if self.global_config.add_sit_token:

                token_embeddings: TensorType["B","1","L","D"] = features['token_embeddings']

                sit_gather_ids = (input_ids == self.global_config.sit_token_id).nonzero()
                sit_gather_idxs, sit_sp_mask = create_gather_idx_tensor(sit_gather_ids, 
                                                    token_embeddings.shape,
                                                    max_gather=T,
                                                    device=token_embeddings.device)

                sit_sp_embs = torch.gather(token_embeddings,dim=2,index=sit_gather_idxs)
                sit_sp_embs: TensorType["B","T","D"] = sit_sp_embs.squeeze(1)
                sit_sp_mask: TensorType["B","T"] = sit_sp_mask.squeeze(1)

                ## 
                sit_mask_resized: TensorType["B","T","D"] = sit_sp_mask.unsqueeze(-1).repeat(1,1,D)
                masked_sit_sp_embs: TensorType["B","T","D"] = sit_sp_embs * sit_mask_resized

                features.update({
                    "sentence_embedding": masked_sit_sp_embs,
                    "sit_special_mask": sit_sp_mask
                })
                
                ### add in features from corresponding situation tokens?
                prop_states[0][:,-1,:] = masked_sit_sp_embs.unsqueeze(2).repeat(1,1,P,1).reshape(B*P*T,D)
                prop_mask_flat[:,-1] = 1


            ## put single label output to just grab 
            single_pad_out = self.model._shift_right(prop_ids_flat)[0:,0:1]

            ### adds pooled representation of full story
            # if self.global_config.add_full_input_rep:
            #     input_decoder_out = self.model.decoder(
            #         input_ids=single_pad_out,
            #         encoder_hidden_states=output_states[0],
            #         encoder_attention_mask=trans_features["attention_mask"],
            #     )
            #     prop_states[0][:,-2,:] =  input_decoder_out.last_hidden_state[0:,0:1,0:]
            #     prop_mask_flat[:,-2]   = 1
                
            decoder_out = self.model.decoder(
                input_ids=single_pad_out,
                encoder_hidden_states=prop_states[0],
                encoder_attention_mask=prop_features["attention_mask"],
            )

            ### first the first hidden state of the decoder for each item, take these to be `pooled` proposition representations 
            decoder_first_hidden_states = decoder_out.last_hidden_state[0:,0:1,0:]

            ## pooled props
            pooled_props : TensorType["B","T","P","D"] = decoder_first_hidden_states.reshape(B,T,P,D)
            features["proposition_matrix"] = pooled_props

        features.update({
             'prop_token_embeddings' : prop_output_tokens,
        })

        return features

    ### generalized this to work with situation encoder decoder
    def collate_fn(self,batch : List):
        """Function for creating batches and turning them into features
        and tensors.

        :param batch: the model batch
        :rtype batch: list
        :raises: NotImplementedError
        """
        if not self._collator:
            raise NotImplementedError(
                'Collator is not implemented or built during object creation!'
            )
        ### simplfiied 
        return self._collator(
            batch,
            self.tokenizer,
            self.global_config,
        )

    @classmethod
    def from_config(cls,config):
        """Loads a transformer model from configuration

        :param config: the global configuration
        """
        model_args = {}
        
        ### might want to use additional config values to populate this
        model_args     = {}
        ### same here 
        tokenizer_args = {}
        do_lowercase = False
        cache = None if not config.cache_dir else config.cache_dir
        
        ## datareader and collators
        reader_func,instance_func = ReaderFunction(config)
        collator_func = CollateFunction(config)

        # add special tokens if not already existing
        add_toks = [tok for tok in [config.sit_token, config.prop_token] if not tok in config.special_tokens]
        config.special_tokens += ";".join(add_toks)

        ### start from an existing sentence bert model 
        if config.sen_bert_encoder:
            m = SentenceTransformer(config.sen_bert_encoder)
            auto_model = m._first_module().auto_model
            tokenizer = m._first_module().tokenizer
            model_config = None #<--- I don't think model config ever gets used for anything 

        else: 
            model_config = AutoConfig.from_pretrained(
                config.model_name_or_path,
                **model_args,
                cache_dir=config.cache_dir
            )
            auto_model = AutoModel.from_pretrained(
                config.model_name_or_path,
                config=model_config,
                cache_dir=config.cache_dir
                )
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path if config.tokenizer_name_or_path is not None else config.model_name_or_path,
                cache_dir=config.cache_dir,
                **tokenizer_args
            )


        if config.max_seq_length is None:
            if hasattr(auto_model, "config") and \
              hasattr(auto_model.config, "max_position_embeddings") and \
              hasattr(tokenizer, "model_max_length"):
                config.max_seq_length = min(
                    auto_model.config.max_position_embeddings,
                    tokenizer.model_max_length
                )

        return cls(
            auto_model,
            tokenizer,
            model_config,
            reader_function=reader_func,
            collator_function=collator_func,
            instance_function=instance_func,
            special_tokens=config.special_tokens,
            global_config=config,
            max_seq_length=config.max_seq_length,
            max_prop_length=config.max_prop_length
        )

    def load_instance(self,query_input,**kwargs):
        """high-level interface for querying models

        :param query_input: the query input 
        """
        ### create `batch`
        instance = temporal_situation_instance_loader(
            query_input,
            **kwargs
        )
        instance.evaluation = True
        return instance

        
        
        
def params(config):
    from .transformer_model import params as t_params
    t_params(config)

    group = OptionGroup(config,"situation_modeling.models.situation_encoder",
                            "Standard settings for the situation encoder modules")

    ### are tied to loss function, should be moved there

    group.add_option("--sit_gen",
                         dest="sit_gen",
                         action='store_true',
                         default=False,
                         help="Add [SIT] generation loss [default=False]")
    group.add_option("--sit_gen_type",
                         dest="sit_gen_type",
                         default='event',
                         help="The type of situation generation to train [default=False]")
    group.add_option("--no_generation",
                         dest="no_generation",
                         action='store_true',
                         default=False,
                         help="Avoids doing any generation outside of [SIT] prediction [default=False]")
    group.add_option("--no_qa",
                         dest="no_qa",
                         action='store_true',
                         default=False,
                         help="Avoids doing QA of any kind [default=False]")
    group.add_option("--invariance_training",
                         dest="invariance_training",
                         action='store_true',
                         default=False,
                         help="Allow for invariance training/swapping [default=False]")
    group.add_option("--add_gen_prefix",
                         dest="add_gen_prefix",
                         action='store_true',
                         default=False,
                         help="Add generation prefix when doing [SIT] generation [default=False]")

    config.add_option_group(group)
