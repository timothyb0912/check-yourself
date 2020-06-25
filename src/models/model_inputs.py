"""
Contains classes for packaging inputs to the forward functions of the various
pytorch models for the project.
"""
# Built-in modules
from collections import OrderedDict
from typing import List

# Third party modules
import attr
import torch
import numpy as np
import pandas as pd
import pylogit as pl
import pylogit.mixed_logit_calcs as mlc

# Local modules
import src.models.mixlb as mixlb
import src.models.torch_utils as utils


class InputMixl(object):
    def __init__(self, *args):
        return None


@attr.s
class InputMixlB:
    # Needed attributes are
    # design matrix
    design: torch.Tensor = attr.ib()
    # rows_to_obs
    obs_mapping: torch.sparse.FloatTensor = attr.ib()
    # rows_to_mixers
    mixing_mapping: torch.sparse.FloatTensor = attr.ib()
    # list of normal random variates
    normal_rvs: List[torch.Tensor] = attr.ib()

    @classmethod
    def from_df(cls,
                df: pd.DataFrame,
                mixing_seed: int=601,
                num_draws: int=250) -> 'InputMixlB':
        """
        Creates a class instance from a dataframe with the requisite data.

        Parameters
        ----------
        df : pandas DataFrame.
            Should be a long-format dataframe containing the following columns:
            `[alt_id, obs_id, choice]`.
        mixing_seed : optional, int.
            Denotes the random seed to use when generating the normal random
            variates for Monte Carlo integration in the maximum simulated
            likelihood procedure.
        num_draws : optional, int.
            Denotes the number of random draws to use for Monte Carlo
            integration in the maximum simulated likelihood procedure.

        Returns
        -------
        Instantiated 'InputMixlB' object.
        """
        # Note the columns that will be needed
        alt_id_column = 'alt_id'
        obs_id_column = 'obs_id'
        choice_column = 'choice'
        # Create specification and name dictionaries
        mnl_spec, mnl_names = OrderedDict(), OrderedDict()

        for col, display_name in mixlb.DESIGN_TO_DISPLAY_DICT.items():
            mnl_spec[col] = 'all_same'
            mnl_names[col] = display_name

        # Instantiate a MNL with the same design matrix as the MIXL.
        mnl_model =\
            pl.create_choice_model(data=df,
                                   alt_id_col=alt_id_column,
                                   obs_id_col=obs_id_column,
                                   choice_col=choice_column,
                                   specification=mnl_spec,
                                   model_type='MNL',
                                   names=mnl_names)

        # Get the design matrix from the original and forecast data
        design_matrix_np = mnl_model.design
        design_matrix =\
            torch.tensor(design_matrix_np.astype(np.float32))

        # Get the rows_to_obs and rows_to_mixers matrices.
        observation_ids = df[obs_id_column].values
        rows_to_obs =\
            utils.create_sparse_mapping_torch(observation_ids)
        rows_to_mixers =\
            utils.create_sparse_mapping_torch(observation_ids)

        ####
        # Get the normal random variates.
        ####
        # Determine the number of observations with randomly distributed
        # sensitivities
        num_mixers = np.unique(observation_ids).size

        # Get the random draws needed for the draws of each coeffcient
        # Each element in the list will be a 2D ndarray of shape
        # num_mixers by num_draws
        normal_rvs_list_np =\
            mlc.get_normal_draws(num_mixers,
                                 num_draws,
                                 len(mixlb.MIXING_VARIABLES),
                                 seed=mixing_seed)
        normal_rvs_list =\
            [torch.from_numpy(x).double() for x in normal_rvs_list_np]

        # Create and return the class object
        return cls(design=design_matrix,
                   obs_mapping=rows_to_obs,
                   mixing_mapping=rows_to_mixers,
                   normal_rvs=normal_rvs_list)
