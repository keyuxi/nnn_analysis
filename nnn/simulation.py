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

from . import util

np.random.seed(42)

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
        nt_distances - (n_temperature, n_sample) np.array
    """
    nt_distances = np.zeros((len(T), num_sample), dtype=int)
    
    for i,celsius in enumerate(T):
        nt_distances[i,:] = sample_nupack_nt_distance(seq, num_sample, sodium, celsius, verbose)
    
    return nt_distances


def plot_nupack_curve_distance(seq, num_sample=1000, sodium=1.0, T=np.arange(20, 62.5, 2.5)):
    nt_distances = sample_nupack_curve_distance(seq, num_sample=num_sample, T=T, sodium=sodium)
    df = pd.DataFrame(nt_distances.T, columns=T).melt()
    df.columns = ['temperature', 'distance_nt']
    # plt.errorbar(T, np.median(nt_distances, axis=1), yerr=np.std(nt_distances, axis=1), fmt='k.')
    fig, ax = plt.subplots(figsize=(12,4))
    sns.violinplot(data=df, x='temperature', y='distance_nt', palette='magma', 
                inner=None, linewidth=0, ax=ax)
    ax.plot(np.mean(nt_distances, axis=1), 'orange', linewidth=4, zorder=10, label='mean')
    ax.plot(np.median(nt_distances, axis=1), 'purple', linewidth=4, zorder=10, label='median')
    ax.legend(loc='lower right')