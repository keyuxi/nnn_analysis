import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from lmfit.models import PowerLawModel

from util import *

sns.set_style('ticks')
sns.set_context('paper')


def get_combined_ddX(p1, p2, e1, e2):
    e = np.sqrt(e1**2 + e2**2)
    ddx = p1 - p2
    z =( ddx - np.nanmean(ddx)) / e

    return z, e, np.nanmean(ddx)


def get_combined_param(p1, p2, e1, e2):
    e = (1 / e1**2 + 1/e2**2)**(-1)
    p = (p1 / e1**2 + p2 / e2**2) * e
    return p, e


def combine_experiments(r1, r2):
    params = ['dH', 'Tm', 'dG_37', 'dS']
    cols_r1 = [c for c in r1.columns if 'dH' in c or 'Tm' in c or 'dG_37' in c or 'dS' in c] + ['n_clusters']
    cols_r2 = [c for c in r2.columns if 'dH' in c or 'Tm' in c or 'dG_37' in c or 'dS' in c] + ['n_clusters']
    df = r1[cols_r1].merge(r2[cols_r2], left_index=True, right_index=True)

    df['dH'], df['dH_se'] = get_combined_param(df.dH_x, df.dH_y, df.dH_se_x, df.dH_se_y)
    df['dS'], df['dS_se'] = get_combined_param(df.dS_x, df.dS_y, df.dS_se_x, df.dS_se_y)
    df['Tm'], df['Tm_se'] = get_combined_param(df.Tm_x, df.Tm_y, df.Tm_se_x, df.Tm_se_y)
    df['dG_37'], df['dG_37_se'] = get_combined_param(df.dG_37_x, df.dG_37_y, df.dG_37_se_x, df.dG_37_se_y)

    return df


def combine_and_correct_interexperiment_error(r1, r2, figdir=None):
    """
    Returns:
        df - combined 2 experiments
    """
    def plot_zscores(df, figdir):
        fig, ax = plt.subplots(1,3,figsize=(18,4))
        sns.histplot(df['ddH_zscore'], bins=50, color=palette[0], ax=ax[0])
        sns.histplot(df['dTm_zscore'], bins=50, color=palette[1], ax=ax[1])
        sns.histplot(df['ddG_37_zscore'], bins=50, color=palette[2], ax=ax[2])
        ax[0].set_xlim([-50, 50])
        ax[0].set_title('dH offset: %.2f kcal/mol'%dH_offset)
        ax[1].set_xlim([-50, 50])
        ax[1].set_title('Tm offset: %.2f K'%Tm_offset)
        ax[2].set_xlim([-50, 50])
        ax[2].set_title('dG 37°C offset: %.2f kcal/mol'%dG_37_offset)
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
        sns.histplot(df.dG_37_se, color='gray')
        sns.histplot(df.dG_37_se_corrected, color='brown')
        plt.legend(['before correction', 'after correction'])
        plt.xlim([0,0.01])
        if figdir is not None:
            save_fig(os.path.join(figdir, 'corrected_dG_se.pdf'),)
        else:
            plt.show()

    def plot_corrected_zscore(df):
        fig, ax = plt.subplots()

        l = 20
        offset = df.dG_37_x - df.dG_37_y
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

    df = combine_experiments(r1, r2)

    df['ddH_zscore'], df['ddH_se'], dH_offset = get_combined_ddX(df.dH_x, df.dH_y, df.dH_se_x, df.dH_se_y)
    df['dTm_zscore'], df['dTm_se'], Tm_offset = get_combined_ddX(df.Tm_x, df.Tm_y, df.Tm_se_x, df.Tm_se_y)
    df['ddG_37_zscore'], df['ddG_37_se'], dG_37_offset = get_combined_ddX(df.dG_37_x, df.dG_37_y, df.dG_37_se_x, df.dG_37_se_y)
    plot_zscores(df, figdir)

    df['ddG_bin'] = pd.qcut(df.ddG_37_se, 100)
    sigma_df = df[['ddG_37_zscore', 'ddG_bin']].groupby('ddG_bin').apply(np.std).rename(columns={'ddG_37_zscore':'ddG_37_zscore_std'})
    sigma_df['intra_err'] = [x.mid for x in sigma_df.index.values]

    model = PowerLawModel()
    powerlaw_result = model.fit(sigma_df.ddG_37_zscore_std[1:-1], x=sigma_df.intra_err[1:-1])
    plot_powerlaw(powerlaw_result)

    df['dG_37_se_corrected'] = df.dG_37_se * powerlaw_result.eval(x=df.ddG_37_se)
    df['ddG_37_se_corrected'] = df.ddG_37_se * powerlaw_result.eval(x=df.ddG_37_se)
    offset = df.dG_37_x - df.dG_37_y
    df['ddG_37_zscore_corrected'] = (offset - np.mean(offset)) / df.ddG_37_se_corrected
    plot_corrected_dG_se(df)
    plot_corrected_zscore(df)

    return df

