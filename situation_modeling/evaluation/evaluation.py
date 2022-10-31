import os
import torch
from torchtyping import TensorType
from torch import Tensor,nn
from typing import List, Dict, Optional, Union, Tuple, Iterable,Type


def round_acc_eval(pred_probs: Tensor, labels: Tensor, mask: Tensor):
    """
    Compute accuracy by rounding predicted probability to gold probability.
    """

    preds: TensorType["N"] = pred_probs.view(-1)
    golds: TensorType["N"] = labels.view(-1)
    mask: TensorType["N"] = mask.view(-1)
    discrete_out = torch.round(preds)
    num_correct = torch.sum(golds == discrete_out)
    
    exclude = mask.shape[0] - mask.count_nonzero()
    
    acc = (num_correct - exclude) / mask.count_nonzero()
    
    return acc


def close_acc_eval(pred_probs: Tensor, labels: Tensor, mask: Tensor, tol: float = 0.25):
    """
    Compute accuracy by specified distance (`tol`) between predicted probability
    and gold probability.
    """
    preds: TensorType["N"] = pred_probs.view(-1)
    golds: TensorType["N"] = labels.view(-1)
    mask: TensorType["N"] = mask.view(-1)
    
    
    num_correct = torch.sum(torch.isclose(preds, golds, atol=tol))
    
    exclude = mask.shape[0] - mask.count_nonzero()
    
    acc = (num_correct - exclude) / mask.count_nonzero()
    
    return acc

def band_acc_eval(pred_probs: Tensor, labels: Tensor, mask: Tensor, bands: int = 3):
    """
    Compute accuracy by first mapping each gold & predicted probability to one of `bands` equally spaced bins.
    E.g., for `bands=3` 1.0 -> 2, 0.5 -> 1, 0.0 -> 0
    """
    preds: TensorType["N"] = pred_probs.view(-1)
    golds: TensorType["N"] = labels.view(-1)
    mask: TensorType["N"] = mask.view(-1)
    
    # to make sure total number of bands will be `bands`
    # o.w. value of 1.0 will get mapped by floor to separate band (e.g., for 1 // 0.33 = 3)
    divisor = 1 / (bands - 0.01) 
    
    pred_bands = torch.div(preds, divisor, rounding_mode='floor').to(torch.int)
    gold_bands = torch.div(golds, divisor, rounding_mode='floor').to(torch.int)
    
    num_correct = torch.sum(pred_bands == gold_bands)
    
    exclude = mask.shape[0] - mask.count_nonzero()
    
    acc = (num_correct - exclude) / mask.count_nonzero()
    
    return acc