import random
import torch
from torch.backends import cudnn


def min_max_normalize(x):
    x = x - torch.min(x)
    x = x / torch.max(x)
    return x

def init_seed(seed):
    '''
        Initalizes the seed for all random number generators used. This is
        important to be able to reproduce results and experiment with different
        random setups of the same code and experiments.
    '''
    if seed is not None:
        random.seed(seed)
        # numpy.random.seed(seed)       # we don't use NumPy in this code, but you would want to set its random number generator seed, too
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True