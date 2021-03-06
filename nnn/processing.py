from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from lmfit.models import PowerLawModel
from tqdm import tqdm
tqdm.pandas()

from .util import *

sns.set_style('ticks')
sns.set_context('paper')


def get_combined_ddX(p1, p2, e1, e2):
    e = np.sqrt(e1**2 + e2**2)
    ddx = p1 - p2
    z =( ddx - np.nanmean(ddx)) / e

    return z, e, np.nanmean(ddx)


def get_combined_param(params, errors):
    """
    Simply combine, weighted by 1/var
    Args:
        params, errors - (n_variant, n_rep) array like
    Returns:
        parameter and error - (n_variant, 2) array, of the combined dataset
    """
    params, errors = np.array(params), np.array(errors)

    assert params.shape == errors.shape, f"Shapes don't match between params {params.shape} and errors {errors.shape}"

    e = (np.nansum(1 / errors**2, axis=1))**(-1)
    p = np.nansum(params / errors**2, axis=1) * e
    return p, np.sqrt(e)


def combine_replicates(reps, rep_names, verbose=True) -> pd.DataFrame:
    """
    Args:
        reps - iterable, each is a df from one replicate
        rep_names - iterable of str
    """
    def get_cols_2_join(rep, params):
        return [c for c in rep.columns if any(c.startswith(p) for p in params)] + ['n_clusters']

    def get_rep_param_from_df(df, param):
        """
        Returns:
            (n_variant, n_rep) df
        """
        return df[[c for c in df.columns if (('-' in c) and c.split('-')[0] == param)]]


    params = ['dH', 'Tm', 'dG_37', 'dS', 'fmax', 'fmin']
    cols = [get_cols_2_join(rep, params) for rep in reps]
    for i, (rep, col, rep_name) in enumerate(zip(reps, cols, rep_names)):
        if i == 0:
            df = rep[col].rename(columns={c: c+'-'+rep_name for c in col})
        else:
            df = df.join(rep[col].rename(columns={c: c+'-'+rep_name for c in col}), how='outer')

    for param in params:
        if verbose:
            print(f'\nCombining {param}')
        df[param], df[param+'_se'] = get_combined_param(get_rep_param_from_df(df, param), get_rep_param_from_df(df, param+'_se'))
        df[param+'_lb'], _ = get_combined_param(get_rep_param_from_df(df, param+'_lb'), get_rep_param_from_df(df, param+'_se'))
        df[param+'_ub'], _ = get_combined_param(get_rep_param_from_df(df, param+'_ub'), get_rep_param_from_df(df, param+'_se'))

    return df


def correct_interexperiment_error(r1, r2, plot=True, figdir=None, return_debug=False):
    """
    Returns:
        A, k - correction parameters
    """
    def plot_zscores(df, figdir):
        fig, ax = plt.subplots(1,3,figsize=(18,4))
        sns.histplot(df['ddH_zscore'], bins=50, color=palette[0], ax=ax[0])
        sns.histplot(df['dTm_zscore'], bins=50, color=palette[1], ax=ax[1])
        sns.histplot(df['ddG_37_zscore'], bins=50, color=palette[2], ax=ax[2])
        ax[0].set_xlim([-10, 10])
        ax[0].set_title('dH offset: %.2f kcal/mol'%offset['dH'])
        ax[1].set_xlim([-10, 10])
        ax[1].set_title('Tm offset: %.2f K'%offset['Tm'])
        ax[2].set_xlim([-10, 10])
        ax[2].set_title('dG 37??C offset: %.2f kcal/mol\nzscore std: %.2f'%(offset['dG_37'], np.std(df['ddG_37_zscore'])))
        if figdir is not None:
            save_fig(os.path.join(figdir, 'zscores.pdf'), fig=fig)
        else:
            plt.show()

    def plot_powerlaw(powerlaw_result):
        powerlaw_result.plot(xlabel='intra-experimental error',
            ylabel='std of ddG z-score')
        if figdir is not None:
            save_fig(os.path.join(figdir, 'fit_powerlaw.pdf'),)
        else:
            plt.show()

    def plot_corrected_dG_se(df):
        fig, ax = plt.subplots()
        sns.histplot(df.dG_37_se, color='gray', stat='density')
        sns.histplot(df.dG_37_se_corrected, color='brown', stat='density')
        plt.legend(['before correction', 'after correction'])
        plt.xlim([0,0.5])
        if figdir is not None:
            save_fig(os.path.join(figdir, 'corrected_dG_se.pdf'),)
        else:
            plt.show()

    def plot_corrected_zscore(df):
        fig, ax = plt.subplots()

        l = 20
        offset = df['dG_37-x'] - df['dG_37-y']
        bins = np.arange(-l, l, 0.5)
        plt.plot(bins, norm.pdf(bins, 0, 1), 'k--')
        zscore = df['ddG_37_zscore']
        sns.histplot(zscore[np.abs(zscore)<l], bins=bins, stat='density', color='gray')
        zscore = df['ddG_37_zscore_corrected']
        sns.histplot(zscore[np.abs(zscore)<l], bins=bins, stat='density', color='brown')
        plt.xlim([-l, l])
        plt.legend(['expected', 'before correction', 'after correction'])
        plt.xlabel('ddG z-score')
        if figdir is not None:
            save_fig(os.path.join(figdir, 'corrected_ddG_zscore.pdf'),)
        else:
            plt.show()

    if not figdir is None:
        if not os.path.isdir(figdir):
            os.makedirs(figdir)

    df = combine_replicates((r1, r2), ('x', 'y'))

    params = ['dG_37', 'Tm', 'dH']
    offset = {p:0.0 for p in params}
    for param in params:
        df[f'd{param}_zscore'], df[f'd{param}_se'], offset[f'{param}'] = get_combined_ddX(df[f'{param}-x'], df[f'{param}-y'], df[f'{param}_se-x'], df[f'{param}_se-y'])
    plot_zscores(df, figdir)

    df['dG_bin'] = pd.qcut(df.ddG_37_se, 100)
    sigma_df = df[['ddG_37_zscore', 'dG_bin']].groupby('dG_bin').apply(np.std).rename(columns={'ddG_37_zscore':'ddG_37_zscore_std'})
    sigma_df['intra_err'] = [x.mid for x in sigma_df.index.values]

    model = PowerLawModel()
    powerlaw_result = model.fit(sigma_df.ddG_37_zscore_std[1:-1], x=sigma_df.intra_err[1:-1])
    plot_powerlaw(powerlaw_result)

    df['dG_37_se_corrected'] = df.dG_37_se * powerlaw_result.eval(x=df.ddG_37_se)
    df['ddG_37_se_corrected'] = df.ddG_37_se * powerlaw_result.eval(x=df.ddG_37_se)
    offset = df['dG_37-x'] - df['dG_37-y']
    df['ddG_37_zscore_corrected'] = (offset - np.mean(offset)) / df.ddG_37_se_corrected
    plot_corrected_dG_se(df)
    plot_corrected_zscore(df)

    if return_debug:
        return powerlaw_result, df
    else:
        return powerlaw_result.best_values['amplitude'], powerlaw_result.best_values['exponent']

