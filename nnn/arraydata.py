"""
The ArrayData class reads and stores all replicates, annotation and combined data
for an array library.

Yuxi Ke, Feb 2022
"""

from distutils.log import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict
from . import fileio, processing


class ErrorAdjust(object):
    """
    Error adjust from intra- to inter- experiment
    \Sigma (\sigma) = A \sigma ^{k}
    """
    def __init__(self, param='dG_37') -> None:
        self.A = 1.0
        self.k = 1.0
        self.param = param

    def adjust_sigma(self, sigma):
        return self.A * sigma**self.k

class ArrayData(object):
    """
    Contains all replicates of the same condition,
    combined and error adjusted.
    Attributes:
        name - str
        lib_name - str, 'nnn_lib2b'
        n_rep - int
        buffer - Dict[str: value], {'sodium': float, in M}
        data - pd.DataFrame, each row is a variant, contains fitted parameters from individual replicates,
               n_clusters, combined and corrected parameters
        curve - dict of pd.DataFrame, keys are replicate,
                levels are variant-replicate-median/se-temperature. 
                Green normed data.
        annotation - pd.DataFrame, designed library
        replicate_df - pd.Dataframe with the information of replicate locations
                     No actual data is loaded
    

    """
    def __init__(self, replicate_df, annotation_file, name='',
                 lib_name='NNNlib2b', filter_misfold: bool=False,
                 learn_error_adjust_from: Tuple[str]=None, error_adjust: ErrorAdjust=None ) -> None:
        """
        Args:
            replicate_df - df, cols ['name', 'replicate', 'chip', 'filename', 'get_cond', 'notes']
            TODO filter_misfold - whether to filter out misfolded secondary structures
            learn_error_adjust_from - Tuple[str], 2 replicates to learn adjustion from
                required if error_adjust is None
            error_adjust - if given, use error adjust function already determined externally
        """
        self.name = name
        self.lib_name = lib_name
        self.replicate_df = replicate_df
        self.n_rep = len(replicate_df)
        
        self.annotation = fileio.read_annotation(annotation_file)
        if filter_misfold:
            pass
        self.data, self.curve = self.read_data()

        if error_adjust is not None:
            self.error_adjust = error_adjust
        else:
            if learn_error_adjust_from is None:
                raise Exception('Need to give `learn_error_adjust_from` if no ErrorAdjust is given!')
            else:
                self.error_adjust = ErrorAdjust()




    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        reps = [fileio.read_fitted_variant(fn) for fn in self.replicate_df.filename] # List[DataFrame]
        conds = [exec(get_cond) for rep,get_cond in zip(reps, self.replicate_df.get_cond)] # List[List[str]]

        data = processing.combine_replicates(reps, self.replicate_df['name'])
        curves = {rep_name: rep[cond] for rep, rep_name, cond in zip(reps, self.replicate_df['name'], conds)}

        return data, curves



    def get_replicate_data(self, replicate: str, attach_annotation: bool = False) -> pd.DataFrame:
        """
        Lower level, returns the original df of fitted variant data
        Compatible with older functions
        """
        filename = self.replicate_df.loc[self.replicate_df['replicate'] == replicate, 'filename']
        if attach_annotation:
            annotation = self.annotation
        else:
            annotation = None

        return fileio.read_fitted_variant(filename, filter=True, annotation=annotation)

    def get_error_adjust_function(self):
        return self.error_adjust