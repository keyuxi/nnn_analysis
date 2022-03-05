"""
The ArrayData class reads and stores all replicates, annotation and combined data
for an array library.

Yuxi Ke, Feb 2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict
from . import fileio, processing

class ArrayData(object):
    """
    Attributes:
        data - pd.DataFrame, each row is a variant, contains fitted parameters, n_clusters, combined
               and corrected parameters
        curve - dict of pd.DataFrame, keys are replicate,
                levels are variant-replicate-median/se-temperature. 
                Green normed data.
        annotation - pd.DataFrame, designed library
        replicates - pd.Dataframe with the information of replicate locations
                     No actual data is loaded
        

    """
    def __init__(self, lib_name, replicate_df, annotation_file, master_annotation_file) -> None:
        """
        Args:
            replicate_df - df, cols ['replicate', 'chip', 'norm_file', 'get_cond', 'notes']
        """
        self.lib_name = lib_name
        self.replicates = replicate_df
        self.n_rep = len(replicate_df)
        self.annotation = fileio.read_annotation(annotation_file, master_annotation_file)
        self.data, self.curve = self.read_data()

    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        reps = [fileio.read_fitted_variant(fn) for fn in self.replicates.filename] # List[DataFrame]
        conds = [exec(get_cond) for rep,get_cond in zip(reps, self.replicates.get_cond)] # List[List[str]]

        data = processing.combine_and_correct_interexperiment_error(reps)