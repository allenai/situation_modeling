import logging
import torch
import random
import numpy as np
from pathlib import Path

util_logger = logging.getLogger('situation_modeling.utils.gen_utils')


def set_seed(seed):
    """Sets the random seed 
 
    :param seed: the initial seed for randomization 
    :type seed: int
    :rtype: None 
    """
    util_logger.info('Setting up the random seed, seed=%d' % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# Flatten list of lists https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
def flatten(t):
    return [item for sublist in t for item in sublist]

# shuffle lines of file using seed if provided
def shuffle_lines(file_path: str, seed: int = None):
    seed = seed if seed else np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)

    file_path = Path(file_path)
    
    # read lines
    with file_path.open() as source:
        data = [(rng.random(), line) for line in source]
    
    # sort data randomly
    data.sort()

    shuffled = [line for _, line in data]
    
    return shuffled

def probs_to_bins(probs: torch.Tensor, num_bins: int) -> torch.Tensor:
    """
    Map each probability to one of `num_bins` equally spaced bins.
    E.g., for `num_bins=3` 1.0 -> 2, 0.5 -> 1, 0.0 -> 0

    :param probs: Tensor of probabilities (in [0,1])
    :type probs: torch.Tensor
    :param num_bins: Total number of bins to map probabilities to.
    :type num_bins: int
    :return: Binned probabilities.
    :rtype: torch.Tensor
    """
    assert(len(probs.shape) == 1), f"`probs` should be a dimension 1 tensor, has {len(probs.shape)} dims!"

    # to make sure total number of bands will be `num_bins`
    # o.w. value of 1.0 will get mapped by floor to separate bin (e.g., for 1 // 0.33 = 3)
    divisor = 1 / (num_bins - 0.01) 

    binned_probs = torch.div(probs, divisor, rounding_mode='floor').to(torch.long)

    return binned_probs