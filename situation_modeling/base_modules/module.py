from optparse import OptionParser,OptionGroup
from .transformer_model import TransformerModel
from .situation_encoder import SituationEncoder #,SinglePassSituationEncoder
from .pooling import BasicPooler,SituationPooler
from .scoring import DotProductScorer, BilinearScorer, ConcatScorer
from .sit_aggregation import  NaiveSelfAttSITAggregator, DefaultSITAggregator,KeyValueSITSelfAttention
from .sit_prop_pooler import SituationPropositionPooler
from .pretrained_sequence_classifier import PretrainedSequenceClassifier as SequenceClassifier
from .pretrained_encoder_decoder import PretrainedEncoderDecoder as Text2TextModel
from .situation_encoder_decoder import SituationEncoderDecoder
from ..utils import Register


### module parameters
from .pooling import params as p_params
from .scoring import params as scr_params
from .sit_aggregation import params as sa_params
from .situation_encoder import params as s_params
from .pretrained_encoder import params as e_params
from .pretrained_sequence_classifier import params as sc_params
from .pretrained_encoder_decoder import params as ed_params
from .situation_encoder_decoder import params as sed_params

_module_params = [
    p_params,
    s_params,
    e_params,
    sc_params,
    ed_params,
    scr_params,
    sed_params,
    sa_params,
]

def params(config):
    """Sets a global configuration instance based on different default model implementations
    here. 

    :param config: the configuration to add option groups to 
    :returns: updated config with new params from each sub-module. 
    """
    for param_func in _module_params:
        param_func(config) 
    
    group = OptionGroup(
        config,
        "situation_modeling.models.base_modules",
        "Standard settings for basic pytorch modules"
    )
    config.add_option_group(group)

_MODELS = {
    "transformer_model"         : SituationEncoder, #TransformerModel, ##<-- should be changed later 
    "situation"                 : SituationEncoder,
    #"situation_single"          : SinglePassSituationEncoder,
    "sequence_classifier"       : SequenceClassifier,
    "situation_encoder_decoder" : SituationEncoderDecoder,
}

_TEXT2TEXT = {
    "basic" : Text2TextModel,
}

_POOLERS = {
    #"sentence_pooler" : SentencePooler,
    "situation_pooler" : SituationPooler,
    "sit_prop_pooler": SituationPropositionPooler
}

__SIT_AGGREGATORS = {
    "default"             : DefaultSITAggregator,
    "naive_self_att"      : NaiveSelfAttSITAggregator,
    "key_value_self_attn" : KeyValueSITSelfAttention,
}

__SCORERS = {
    "dot_prod_scorer" : DotProductScorer,
    "bilinear_scorer": BilinearScorer,
    "concat_scorer": ConcatScorer
}

### register new models

class RegisterModule(Register):
    def factory_update_method(self,class_to_register):
        cname = self.name
        _MODELS.update(
            {cname : class_to_register}
        )
        if hasattr(class_to_register,"PARAMS") and class_to_register.PARAMS:
            _module_params.append(class_to_register.PARAMS)
        

class RegisterText2Text(Register):
    def factory_update_method(self,class_to_register):
        cname = self.name
        _TEXT2TEXT.update(
            {cname : class_to_register}
        )        


def Module(config):
    """Factor method for building a model

    :param config: the global configuration 
    """
    mclass = _MODELS.get(config.module_type,None)
    if mclass is None:
        raise ValueError(
            'Unknown model class: %s, available_models=%s' %\
            (config.module_type,', '.join(_MODELS.keys()))
        )
    return mclass

def Text2TextModule(config):
    tclass = _TEXT2TEXT.get(config.text2text_type)
    if tclass is None:
        raise ValueError(
            'Unknown text2text type: %s, available_models=%s' %\
             (config.text2text_type,', '.join(_TEXT2TEXT.keys()))
        )
    return tclass

def BuildModule(config):
    """Factor method for building a model

    :param config: the global configuration 
    :raises: ValueError 
    """
    mclass = Module(config)
    return mclass.from_config(config)

def BuildText2Text(config):
    """Factor method for building a model

    :param config: the global configuration 
    :raises: ValueError 
    """
    tclass = Text2TextModule(config)
    return tclass.from_config(config)

def Pooler(config):
    """Factor method for building a pooler

    :param config: the global configuration 
    :raises: ValueError 
    """
    pclass = _POOLERS.get(config.pooler_type,None)
    if pclass is None:
        raise ValueError(
            'Unknown pooler type: %s, available poolers=%s' %\
            (config.pooler_type,', '.join(_POOLERS.keys()))
        )
    return pclass

def BuildPooler(config):
    """Factor method for building a pooler

    :param config: the global configuration 
    :raises: ValueError 
    """
    pclass = Pooler(config)
    return pclass.from_config(config)    

def Scorer(config):
    """
    Factory method for building a scorer.

    :param config: the global configuration 
    :raises: ValueError 
    """
    sclass = __SCORERS.get(config.scorer_type, None)
    if sclass is None:
        raise ValueError(
            'Unknown scorer type: %s, available scorers=%s' %\
            (config.scorer_type,', '.join(__SCORERS.keys()))
        )
    return sclass

def BuildScorer(config):
    """Factory method for building a scorer

    :param config: the global configuration 
    :raises: ValueError 
    """
    sclass = Scorer(config)
    return sclass.from_config(config) 


def SITAggregator(config):
    """
    Factory method for building a situation aggregator.

    :param config: the global configuration 
    :raises: ValueError 
    """
    sclass = __SIT_AGGREGATORS.get(config.sit_agg_type, None)
    if sclass is None:
        raise ValueError(
            'Unknown situation aggregator type: %s, available situation aggregators=%s' %\
            (config.sit_agg_type,', '.join(__SIT_AGGREGATORS.keys()))
        )
    return sclass

def BuildSITAggregator(config):
    """Factory method for building a situation aggregator.

    :param config: the global configuration 
    :raises: ValueError 
    """
    sclass = SITAggregator(config)
    return sclass.from_config(config) 
