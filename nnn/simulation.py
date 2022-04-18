from tkinter import N
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
import colorcet as cc
from scipy.stats import chi2, pearsonr, norm
from sklearn.metrics import r2_score
from lmfit.models import PowerLawModel
from ipynb.draw import draw_struct
import nupack
from matplotlib.backends.backend_pdf import PdfPages
from pandarallel import pandarallel


from . import util

np.random.seed(42)

def distance_2_norm_fluor(x, a=93):
    norm_fluor = 1 / (1 + (a * x**(-3.0)))
    # norm_fluor[np.isinf(norm_fluor)] = 0.0
    return norm_fluor

def get_transform_curve(max_len=40, a=93):
    nt_range = np.arange(max_len)
    transform_curve = distance_2_norm_fluor(nt_range, a=a)
    return transform_curve
    

def simulate_nupack_curve_p_closing(seq, sodium=1.0, T=np.arange(20,62.5,2.5)):
    """
    Simulate P[closing base pair] as a function of temperatures
    returns P[open base pair] to mimic melt curve in an experiment
    """

    p_close = np.zeros_like(T)
    for i,celsius in enumerate(T):
        p_close[i] = util.get_seq_end_pair_prob(seq, celsius=celsius, sodium=sodium, n_pair=2)
    
    return 1 - p_close


def sample_nupack_nt_distance(seq, num_sample=100, sodium=1.0, celsius=37, verbose=False):
    """
    Sample distances at one temperature point.
    """
    my_model = nupack.Model(material='DNA', celsius=celsius, sodium=sodium, magnesium=0.0)
    sampled_structures = nupack.sample(strands=[seq], num_sample=num_sample, model=my_model)
    if verbose:
        print(sampled_structures)
    nt_distances = np.array([util.get_fluor_distance_from_structure(str(s)) for s in sampled_structures], dtype=int)
    
    return nt_distances


def sample_nupack_curve_distance(seq, num_sample=100, sodium=1.0, T=np.arange(20, 62.5, 2.5), verbose=False):
    """
    Simulates a melt curve of distances in nt
    Returns:
        nt_distances - (n_sample, n_temperature) np.array
    """
    nt_distances = np.zeros((num_sample, len(T)), dtype=int)
    
    for i,celsius in enumerate(T):
        nt_distances[:,i] = sample_nupack_nt_distance(seq, num_sample, sodium, celsius, verbose)
    
    return nt_distances


def simulate_nupack_curve(seq, num_sample=1000, sodium=1.0, T=np.arange(20, 62.5, 2.5),
                          transform_param={'a': 93}, transform_curve=None):
    """
    Args:
        transform_param - param for the transform curve
        transform_curve - array like, `transform_curve[n]` is the normalized fluorescence 
    Returns:
        simulated fluorescence curve (macroscopic mean of the ensemble)
    """
    if transform_curve is None:
        # if did not supply existing transform_curve array, requiring transform_param dict
        nt_range = np.arange(30)
        transform_curve = distance_2_norm_fluor(nt_range, a=transform_param['a'])
        
    nt_distances = sample_nupack_curve_distance(seq, num_sample=num_sample, T=T, sodium=sodium)
    nt_distances = transform_curve[nt_distances]

    return np.mean(nt_distances, axis=0)
        
        
def plot_nupack_curve(seq, num_sample=1000, sodium=1.0, T=np.arange(20, 62.5, 2.5),
                      transform_curve=None):
    """
    Args:
        transform_curve - array like, `transform_curve[n]` is the normalized fluorescence 
        expected at a distance of n nt. If not given, only plot the raw distance.
    """
    nt_distances = sample_nupack_curve_distance(seq, num_sample=num_sample, T=T, sodium=sodium)
    if transform_curve is not None:
        nt_distances = transform_curve[nt_distances]

    df = pd.DataFrame(nt_distances, columns=T).melt()
    df.columns = ['temperature', 'distance_nt']
    # plt.errorbar(T, np.median(nt_distances, axis=1), yerr=np.std(nt_distances, axis=1), fmt='k.')
    fig, ax = plt.subplots(figsize=(12,4))
    sns.violinplot(data=df, x='temperature', y='distance_nt', palette='magma', 
                   inner=None, linewidth=0, ax=ax)
    ax.plot(np.median(nt_distances, axis=0), 'orange', linewidth=2, zorder=9, label='median')
    ax.plot(np.mean(nt_distances, axis=0), 'purple', linewidth=4, zorder=10, label='mean')
    ax.legend(loc='lower right')

    if transform_curve is not None:
        ax.set_ylabel('Normalized Fluorescence')


def simulate_CPseries(annotation, num_sample=1000, sodium=0.075, T=np.arange(20, 62.5, 2.5)):
    """
    Simulates melt curves for the entire library.
    Args:
        transform_param - param for the transform curve
    Returns:
        series_df - (n_seq, n_temp) simulated fluorescence curve (macroscopic mean of the ensemble)
    """
    pandarallel.initialize()
    n_seq = annotation.shape[0]
    n_temp = len(T)
    transform_curve = get_transform_curve(max_len=40, a=93.0)
    series_df = np.zeros((n_seq, n_temp))
    
    curves = annotation.RefSeq.parallel_apply(lambda seq: simulate_nupack_curve(seq, num_sample=num_sample, 
                                     sodium=sodium, T=T, transform_curve=transform_curve))
    return np.stack(curves.values, axis=0)