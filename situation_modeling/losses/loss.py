from optparse import OptionParser,OptionGroup
from .bce_situation_loss import (
    BCELoss,
    JointBCEGenerationLoss
)    
from .multi_class_ce_situation_loss import (
    MultiClassCELoss,
    JointMultiCEGenerationLoss,
    LogicalRelaxationLoss,
) 

_LOSSES = {
    "bce"       : BCELoss,
    "bce_multi" : JointBCEGenerationLoss,
    "mcce"      : MultiClassCELoss,
    "mcce_multi": JointMultiCEGenerationLoss,
    "logic_loss": LogicalRelaxationLoss,
}
    
def params(config):
    """Main configuration settings for this module 

    :param config: a global configuration instance
    :rtype: None 
    """
    group = OptionGroup(config,"situation_modeling.models.losses",
                            "Standard settings for loss functions and modules")

    group.add_option("--loss_type",
                         dest="loss_type",
                         default="bce",
                         type=str,
                         help="the type of loss function to use [default='bce']")
    group.add_option("--logic_aggr",
                         dest="logic_aggr",
                         default="mean",
                         type=str,
                         help="The logic aggregation type to use [default='mean']")
    group.add_option("--joint_aggr",
                         dest="joint_aggr",
                         default="mean",
                         type=str,
                         help="The joint aggregation to use [default='mean']")
    group.add_option("--loss_aggr",
                         dest="loss_aggr",
                         default="mean",
                         type=str,
                         help="The type of aggregation to use for loss [default='mean']")
    group.add_option("--acc_eval_type",
                         dest="acc_eval_type",
                         default="by_prop",
                         type=str,
                         help="the type of evaluation method to use [default='by_prop']")
    group.add_option("--constraint_weights",
                         dest="constraint_weights",
                         default="",
                         type=str,
                         help="The weights to use [default='']")
    group.add_option("--logic_loss_weight",
                         dest="logic_loss_weight",
                         default=1.,
                         type=float,
                         help="The weight for the main logic loss [default=1.]")
    group.add_option("--label_dropout",
                         dest="label_dropout",
                         default=0.,
                         type=float,
                         help="The parameter for label dropout [default=0.]")
    group.add_option("--no_constraint_loss",
                         dest="no_constraint_loss",
                         action='store_true',
                         default=False,
                         help="Do not do constraint loss [default=False]")
    group.add_option("--model_agreement",
                         dest="model_agreement",
                         action='store_true',
                         default=False,
                         help="Model agreement (requires additional rules) [default=False]")
    group.add_option("--uncertainty_loss",
                         dest="uncertainty_loss",
                         action='store_true',
                         default=False,
                         help="Learn weight functions via uncertainty [default=False]")
    group.add_option("--log_all",
                         dest="log_all",
                         action='store_true',
                         default=False,
                         help="Do extensive logging of loss information [default=False]")
    group.add_option("--logic_type",
                         dest="logic_type",
                         type=str,
                         default='fuzzy',
                         help="The type of logic to use for constrained loss [default='fuzzy']")
    group.add_option("--prod_norm",
                         dest="prod_norm",
                         default='r_prod',
                         type=str,
                         help="The type of product norm to use (if using logical relaxations) [default=False]")
    group.add_option("--exclude_rules",
                         dest="exclude_rules",
                         default='',
                         type=str,
                         help="The rules to exclude (if using logical relaxations) [default=False]")

    ### loss weight parameters (for multi-task loss)
    group.add_option("--gen_param",
                         dest="gen_param",
                         default=0.5,
                         type=float,
                         help="Interpolation parameter for generation in multi-task learning [default=0.5]")
    group.add_option("--sit_param",
                         dest="sit_param",
                         default=0.5,
                         type=float,
                         help="Interpolation parameter for controlling sit prediction loss [default=0.5]")
    group.add_option("--class_param",
                         dest="class_param",
                         default=1.0,
                         type=float,
                         help="Interpolation parameter for classification in multi-task learning [default=0.5]")
    group.add_option("--main_loss_weight",
                         dest="main_loss_weight",
                         default=1.,
                         type=float,
                         help="The weight for the main (conjunctive) loss [default=1.]")
    group.add_option("--qa_loss_weight",
                         dest="qa_loss_weight",
                         default=1.,
                         type=float,
                         help="The weight for qa sub system [default=1.]")
    group.add_option("--qa_param",
                         dest="qa_param",
                         default=0.5,
                         type=float,
                         help="Parameter controlling QA loss (for multi-task models) [default=0.5]")

    ### loss warmup parameters 
    group.add_option("--consistency_warmup",
                         dest="consistency_warmup",
                         default=0,
                         type=int,
                         help="The stage to introduce consistency losses (if using logical relaxation) [default=0]")
    group.add_option("--generation_warmup",
                         dest="generation_warmup",
                         default=0,
                         type=int,
                         help="The stage to introduce generation losses (if using multi-task model) [default=0]")
    group.add_option("--qa_warmup",
                         dest="qa_warmup",
                         default=0,
                         type=int,
                         help="The stage to introduce qa losses (if using multi-task model) [default=0]")
    group.add_option("--main_warmup",
                         dest="main_warmup",
                         default=0,
                         type=int,
                         help="The stage to introduce main loss (if using multi-task model) [default=0]")

    ### loss min/max values values
    group.add_option("--qa_min",
                         dest="qa_min",
                         default=0.0,
                         type=float,
                         help="The minimum value for QA loss when shut off (for multi-task models) [default=0.0]")
    group.add_option("--qa_max",
                         dest="qa_max",
                         default=1000000,
                         type=int,
                         help="The minimum value for QA loss when shut off (for multi-task models) [default=0.0]")    
    group.add_option("--gen_min",
                         dest="gen_min",
                         default=0.0,
                         type=float,
                         help="The value for QA loss when shut off (for multi-task models) [default=0.0]")
    group.add_option("--gen_max",
                         dest="gen_max",
                         default=1000000,
                         type=int,
                         help="The maxmimum number of epochs to run gen loss (for multi-task models) [default=100000]")
    group.add_option("--main_min",
                         dest="main_min",
                         default=0.0,
                         type=float,
                         help="The value for main loss when shut off (for multi-task models) [default=0.0]")
    group.add_option("--main_max",
                         dest="main_max",
                         default=1000000,
                         type=int,
                         help="The maximum number of epochs to run with main loss [default=1000000]")    
    group.add_option("--consistency_min",
                         dest="consistency_min",
                         default=0.0,
                         type=float,
                         help="The value for consistency loss when shut off (for multi-task models) [default=0.0]")    
    group.add_option("--consistency_max",
                         dest="consistency_max",
                         default=1000000,
                         type=int,
                         help="The maximum number of epochs to run with consistency loss [default=1000000]")    
    
    
    config.add_option_group(group)


def BuildLoss(config):
    """factory method for building a loss function 

    :param config: the global configuration instance 
    """
    lclass = _LOSSES.get(config.loss_type)
    if lclass is None:
        raise ValueError(
            'Unknown loss type: %s' % config.loss_type 
        )
    return lclass.from_config(config)

