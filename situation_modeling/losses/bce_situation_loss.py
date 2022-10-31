import os
import torch
from torchtyping import TensorType
from torch import Tensor,nn
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type

from ..evaluation import round_acc_eval, close_acc_eval, band_acc_eval
from .loss_base import ScoreLoss

def _run_eval(
        eval_type,
        pred_probs,
        labels,
        prop_mask
    ):
    """Computes an accuracy metric based on evaluation functions defined 
    in `situation_encoder.evaluation`

    :params eval_type: the type of evaluator to use 
    :param pred_probs: the prediced situation probabilities 
    :param labels: the target labels
    :param prop_mask: the proposition mask
    """
    if eval_type == "round_acc_eval":
        acc = round_acc_eval(pred_probs, labels, prop_mask)
    elif eval_type == "close_acc_eval":
        acc = close_acc_eval(pred_probs, labels, prop_mask)
    elif eval_type == "band_acc_eval":
        acc = band_acc_eval(pred_probs, labels, prop_mask)
    else:
        # default to round (maybe warning?)
        acc = round_acc_eval(pred_probs, labels, prop_mask)

    return acc


class BCELoss(ScoreLoss):
    """Binary cross-entropy loss
    """
    def __init__(self, config):
        super(BCELoss, self).__init__(config)
        if hasattr(config, "class_output"):
            # compatibility with older versions that don't have this flag in config
            assert(not config.class_output), "BCELoss can't be used with --class_output flag!"
        assert(config.acc_eval_type in ["round_acc_eval","close_acc_eval",
                                        "band_acc_eval"]), f"BCELoss must be used with --config.acc_eval_type in ['round_acc_eval','close_acc_eval', 'band_acc_eval'], is {config.acc_eval_type}!"
        self.loss_fn = nn.BCELoss()

    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        """The main forward method
        """
        scores: TensorType["B", "T", "P", 1] = features["scores"]

        # sigmoid + mask inactive propositions for loss calculation
        pred_probs: TensorType["B", "T", "P"] = torch.sigmoid(scores.squeeze(-1)) * features["prop_mask"]
        
        loss = None
        acc  = None

        if labels is not None:
            
            loss = self.loss_fn(pred_probs,labels)
            
            ## get an accuracy measurement (moved to global function above)
            ## important: only want to compute if we have `labels`
            acc = _run_eval(
                self.config.acc_eval_type,
                pred_probs,
                labels,
                features["prop_mask"],
            )

        ## items to carry over for printing (if provided) 
        print_out = {} if "print_out" not in features else features["print_out"]
        data_stats = {} if "data_stats" not in features else features["data_stats"]

        output_dict = {
            "loss"      : loss,
            "metrics"   : {"acc" : acc},
            "outputs"   : pred_probs, 
            "print_out" : print_out,
            "labels"    : labels,
            "data_stats": data_stats,
        }

        return output_dict

    @classmethod
    def from_config(cls,config):
        return cls(config)


class JointBCEGenerationLoss(BCELoss):
    """Loss function for training a multi-task situation classifier and 
    generation model.
    """

    def __init__(self,config):
        super().__init__(config)
        if config.class_param + config.gen_param != 1.0:
            raise ValueError(
                'Interpolation parameters do not sum to 1.0!'
            )


    def forward(self, features: Dict[str, Tensor], labels: Tensor = None):
        """The main forward method
        """
        ### encodings for text inputs and propositions
        out_dict = super().forward(features,labels)
    
        ### add interpolated loss 
        if (labels is not None) and (self.config.gen_param > 0.0) and ("gen_loss" in features):
            orig_loss = out_dict["loss"]

            out_dict["full_losses"] = {}
            out_dict["full_losses"]["sloss"] = orig_loss
            out_dict["full_losses"]["gloss"] = features["gen_loss"]

            interp_loss = (self.config.class_param*orig_loss) + (self.config.gen_param*features["gen_loss"])
            out_dict["loss"] = interp_loss

        return out_dict 
