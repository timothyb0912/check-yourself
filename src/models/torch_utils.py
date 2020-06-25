# -*- coding: utf-8 -*-
"""
Common utilities for PyTorch models.
"""
import numpy as np
import torch
import torch.sparse as sparse

import pylogit.choice_tools as ct


def create_sparse_mapping_torch(id_array: np.ndarray) -> sparse.FloatTensor:
    """
    Creates a sparse mapping matrix from rows to unique values of `id_array` as
    a torch.sparse.FloatTensor object.

    Parameters
    ----------
    id_array : 1D ndarray.
        Should denote the identifying values that we want to map every row to.

    Returns
    -------
    mapping : torch.sparse.FloatTensor.
        Sparse matrix containing only zeros and ones. Each row pertains to an
        element in `id_array`, and each column pertains to a unique value in
        `id_array`, in order of appearance.
    """
    mapping_scipy = ct.create_sparse_mapping(id_array).tocoo()
    torch_mapping_indices =\
        torch.LongTensor(torch.from_numpy(
            np.concatenate((mapping_scipy.row[None, :],
                            mapping_scipy.col[None, :]),
                           axis=0).astype(np.int_)))
    torch_mapping_values =\
        torch.from_numpy(mapping_scipy.data.astype(np.float32)).double()
    num_rows = mapping_scipy.data.size
    num_cols = ct.get_original_order_unique_ids(id_array).size
    mapping_torch =\
        sparse.FloatTensor(
            torch_mapping_indices,
            torch_mapping_values,
            torch.Size([num_rows, num_cols]))
    return mapping_torch


# Create the loss function
def log_loss(probs: torch.Tensor,
             targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the log-loss (i.e., the negative log-likelihood) for given
    long-format tensors of probabilities and choice indicators.

    Parameters
    ----------
    probs : 1D torch.Tensor.
        The probabilities of each row's alternative being chosen for the
        given choice situation.
    targets : 1D torch.Tensor.
        A Tensor of zeros and ones indicating which row was chosen for each
        choice situation. Should have the same size as `probs`.

    Returns
    -------
    neg_log_likelihood : scalar torch.Tensor.
        The negative log-likelihood computed from `probs` and `targets`.
    """
    log_likelihood = torch.sum(targets * torch.log(probs))
    return -1 * log_likelihood
