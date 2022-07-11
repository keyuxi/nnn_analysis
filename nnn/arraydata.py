"""
The ArrayData class reads and stores all replicates, annotation and combined data
for an array library.

Yuxi Ke, Feb 2022
"""

from distutils.log import error
from pickletools import float8
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
        data_all - df, all shitty columns
        data - df, just the clean combined parameters
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
        if len(replicate_df.shape) == 2:
            self.n_rep = len(replicate_df)
        else:
            self.n_rep = 1
            

        if self.n_rep > 1:
            assert np.all(np.isclose(replicate_df['sodium'], replicate_df['sodium'][0])), "Sodium concentration not equal in the replicates"
            self.buffer = {'sodium': replicate_df['sodium'][0]}
        else:
            self.buffer = {'sodium': replicate_df['sodium']}
        
        self.annotation = fileio.read_annotation(annotation_file, sodium=self.buffer['sodium'])
        if filter_misfold:
            pass

        if self.n_rep > 1:
            self.data_all, self.curve, self.curve_se = self.read_data()
        else:
            self.data_all, self.curve, self.curve_se = self.read_data_single()


        if error_adjust is not None:
            self.error_adjust = error_adjust
        else:
            if learn_error_adjust_from is None:
                # OK to have no adjust
                self.error_adjust = ErrorAdjust()
                # raise Exception('Need to give `learn_error_adjust_from` if no ErrorAdjust is given!')
            else:
                self.error_adjust = self.learn_error_adjust_function(learn_error_adjust_from)

        self.data = self.data_all[[c for c in self.data_all.columns if not ('-' in c)]]




    def read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        reps = [fileio.read_fitted_variant(fn, add_chisq_test=False, annotation=None) for fn in self.replicate_df.filename] # List[DataFrame]
        
        conds = []
        for rep, drop_last, reverse in zip(reps, self.replicate_df.drop_last, self.replicate_df.reverse):
            cond = [x for x in rep.columns if x.endswith('_norm')]
            if drop_last:
                cond = cond[:-1]
            if reverse:
                cond = cond[::-1]
            conds.append(cond)

        data = processing.combine_replicates(reps, self.replicate_df['name'].tolist(), verbose=False)
        curves = {rep_name: rep[cond] for rep, rep_name, cond in zip(reps, self.replicate_df['name'], conds)}
        curves_se = {rep_name: rep[[c+'_std' for c in cond]].rename(columns=lambda c: c.replace("_std", "_se")) / np.sqrt(rep['n_clusters']).values.reshape(-1,1)
            for rep, rep_name, cond in zip(reps, self.replicate_df['name'], conds)}
        return data, curves, curves_se

    def read_data_single(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        rep = fileio.read_fitted_variant(self.replicate_df.filename, add_chisq_test=False, annotation=None)
        
        cond = [x for x in rep.columns if x.endswith('_norm')]
        if self.replicate_df.drop_last:
            cond = cond[:-1]
        if self.replicate_df.reverse:
            cond = cond[::-1]

        data = processing.combine_replicates([rep], self.replicate_df['name'], verbose=False)
        curves = {self.replicate_df['name']: rep[cond]}
        curves_se = {self.replicate_df['name']: rep[[c+'_std' for c in cond]].rename(columns=lambda c: c.replace("_std", "_se")) / np.sqrt(rep['n_clusters']).values.reshape(-1,1)}
        return data, curves, curves_se


    def get_replicate_data(self, replicate_name: str, attach_annotation: bool = False, verbose=True) -> pd.DataFrame:
        """
        Lower level, returns the original df of fitted variant data
        Compatible with older functions
        """
        # assert replicate_name in self.replicate_df['name'], "Invalid replicate name"
        filename = self.replicate_df.loc[self.replicate_df.name == replicate_name, 'filename'].values[0]
        if verbose:
            print('Load from file', filename)
        if attach_annotation:
            annotation = self.annotation
        else:
            annotation = None
            
        return fileio.read_fitted_variant(filename, filter=True, annotation=annotation, sodium=self.buffer['sodium'])
            
    def get_replicate_curves(self, replicate_name: str, verbose: bool = True):
        """
        Return the normalized values and std of a replicate experiment.
        """
        repdata = self.get_replicate_data(replicate_name=replicate_name, verbose=verbose)
        values = repdata[[c for c in repdata.columns if c.endswith('_norm')]]
        se = repdata[[c for c in repdata.columns if c.endswith('_norm_std')]] / np.sqrt(repdata['n_clusters'].values.reshape(-1,1))
        xdata = np.array([c.split('_')[1] for c in repdata.columns if c.endswith('_norm_std')], dtype=float)
        xdata += 273.15
        
        return xdata, values, se


    def learn_error_adjust_function(self, learn_error_adjust_from, debug=False, figdir='./fig/error_adjust'):
        r1_name, r2_name = learn_error_adjust_from
        r1, r2 = self.get_replicate_data(r1_name), self.get_replicate_data(r2_name)

        error_adjust = ErrorAdjust()
        error_adjust.A, error_adjust.k = processing.correct_interexperiment_error(r1, r2, figdir=figdir, return_debug=debug)

        return error_adjust