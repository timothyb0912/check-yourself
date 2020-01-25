# PyTorch is used for numeric computation and automatic differentiation
import torch
# Used to access various pytorch utilities
import torch.nn as nn
# Used to get sparse matrices
import torch.sparse


class MIXLB(nn.Module):
    def __init__(self):
        """"
        PyTorch implementation of `Mixed Logit B` in [1].

        References
        ----------
        [1] Brownstone, David, and Kenneth Train. "Forecasting new product
        penetration with flexible substitution patterns." Journal of
        econometrics 89.1-2 (1998): 109-129.
        """
        # Should store all needed information required to specifcy the
        # computational steps needed to calculate the probability function
        # corresponding to `Mixed Logit B`
        return None

    def forward(self):
        # Should specify the computational steps for calculating the
        # probability function corresponding to `Mixed Logit B`.
        raise NotImplementedError()
