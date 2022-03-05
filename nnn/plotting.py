import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
import colorcet as cc
from scipy.stats import chi2, pearsonr, norm
from sklearn.metrics import r2_score
from ipynb.draw import draw_struct

from .util import *

sns.set_style('ticks')
sns.set_context('paper')

# palette = ['#2f4f4f','#228b22','#00ff00','#000080','#1e90ff','#00ffff','#ff8c00','#deb887','#8b4513','#ff0000','#ff69b4','#800080',]
palette = cc.glasbey_dark
# palette=[
#     '#201615',
#     '#4e4c4f',
#     '#4c4e41',
#     '#936d60',
#     '#5f5556',
#     '#537692',
#     '#a3acb1']#cc.glasbey_dark


def plot_scatter_comparison(df, col, lim, color='#deb887'):
    fig, ax = plt.subplots(figsize=(6,6))
    sns.scatterplot(data=df, x=col, y=col+'_final', color=color)
    plt.plot(lim, lim, 'k--')
    plt.xlim(lim)
    plt.ylim(lim)
    
def plot_kde_comparison(df, col, lim, color='#8b4513'):
    fig, ax = plt.subplots(figsize=(6,6))
    sns.kdeplot(data=df, x=col, y=col+'_final', color=color)
    plt.plot(lim, lim, 'k--')
    plt.xlim(lim)
    plt.ylim(lim)
    
def plot_dis_comparison(df, col, lim, color='#deb887'):
    # fig, ax = plt.subplots(figsize=(6,6))
    sns.displot(data=df, x=col, y=col+'_final', color=color)
    plt.plot(lim, lim, 'k--')
    plt.xlim(lim)
    plt.ylim(lim)

def plot_se_dist(df):
    fig, ax = plt.subplots(1,3,figsize=(12,3), sharey=True)
    if 'dG_37_se' in df.columns:
        sns.histplot(df.dG_37_se, kde=False, bins=30, color=palette[0], ax=ax[0])
    sns.histplot(df.Tm_se, kde=False, bins=30, color=palette[1], ax=ax[1])
    sns.histplot(df.dH_se, kde=False, bins=30, color=palette[2], ax=ax[2])


def plot_rep_comparison(r1, r2, param, lim, kind='kde', add_final=False, color='#deb887'):
    if add_final:
        col = param + '_final'
    else:
        col = param
        
    df = r1[[col]].merge(r2[[col]], left_index=True, right_index=True)
    rsqr = r2_score(df[col+'_x'], df[col+'_y'])
    pearson, _ = pearsonr(df[param+'_x'], df[param+'_y'])

    fig, ax = plt.subplots(figsize=(6,6))
    l = np.abs(lim[1] - lim[0])
    plt.plot(lim, lim, '--', c='gray')
    if kind == 'kde':
        sns.kdeplot(data=df, x=col+'_x', y=col+'_y', color=color)
    else:
        sns.scatterplot(data=df, x=col+'_x', y=col+'_y', color=color)

    plt.xlim(lim)
    plt.ylim(lim)    
    plt.xlabel('r1')
    plt.ylabel('r2')
    plt.text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
    plt.text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
    plt.title(param)
    
def plot_rep_comparison_by_series(r1, r2, annotation, param, lim,
    suffixes=('_x', '_y'), xlabel='r1', ylabel='r2'):
    df = r1[[param]].merge(r2[[param]], left_index=True, right_index=True)
    df = df.merge(annotation, left_index=True, right_index=True)

    series = df.groupby('Series').apply(len).sort_values(ascending=False)
    l = np.abs(lim[1] - lim[0])

    fig, ax = plt.subplots(3,4,figsize=(20,15), sharex=True, sharey=True)
    ax = ax.flatten()

    for i, s in enumerate(series.index):
        series_df = df.query('Series == "%s"'%s)
        print('Series %s,  %d variants' % (s, len(series_df)))
        ax[i].plot(lim, lim, '--', c='gray')
        if len(series_df) > 100:
            sns.kdeplot(data=series_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])
            rsqr = r2_score(series_df[param+suffixes[0]], series_df[param+suffixes[1]])
            pearson, _ = pearsonr(series_df[param+suffixes[0]], series_df[param+suffixes[1]])
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
        else:
            sns.scatterplot(data=series_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])

        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)    
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title('%s, N=%d'%(s, series[s]))

    plt.suptitle(param)

    return fig, ax


def plot_rep_comparison_by_ConstructType(r1, r2, annotation, param, series, lim,
    suffixes=('_x', '_y'), xlabel='r1', ylabel='r2'):

    df = r1[[param]].merge(r2[[param]], left_index=True, right_index=True)
    df = df.merge(annotation, left_index=True, right_index=True).query('Series == "%s"'%series)

    types = df.groupby('ConstructType').apply(len).sort_values(ascending=False)
    l = np.abs(lim[1] - lim[0])

    fig, ax = plt.subplots(1, 3, figsize=(18,6), sharex=True, sharey=True)
    ax = ax.flatten()

    for i, s in enumerate(types.index):
        types_df = df.query('ConstructType == "%s"'%s)
        print('ConstructType %s,  %d variants' % (s, len(types_df)))
        ax[i].plot(lim, lim, '--', c='gray')
        if len(types_df) > 100:
            sns.kdeplot(data=types_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])
            rsqr = r2_score(types_df[param+suffixes[0]], types_df[param+suffixes[1]])
            pearson, _ = pearsonr(types_df[param+suffixes[0]], types_df[param+suffixes[1]])
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.1*l, r'$R^2 = %.3f$'%rsqr, va='bottom')
            ax[i].text(lim[0] + 0.1*l, lim[1] - 0.15*l, r"$Pearson's\ r = %.3f$"%pearson, va='bottom')
        else:
            sns.scatterplot(data=types_df, x=param+suffixes[0], y=param+suffixes[1],
                color=palette[i % len(palette)], ax=ax[i])

        ax[i].set_xlim(lim)
        ax[i].set_ylim(lim)    
        ax[i].set_xlabel(xlabel)
        ax[i].set_ylabel(ylabel)
        ax[i].set_title('%s, N=%d'%(s, types[s]))

    plt.suptitle(param)

    return fig, ax

def plot_actual_and_expected_fit(row, ax, c='k', conds=None):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*(Tm**-1 - x)))
    if conds is None:
        conds = [x for x in row.keys() if x.endswith('_norm')]
        
        errs = [x for x in row.keys() if x.endswith('_norm_std')]
    else:
        errs = [x+'_std' for x in conds]

    vals = np.array(row[conds].values,dtype=float) 
    errors = np.array(row[errs].values / np.sqrt(row['n_clusters']),dtype=float)

    T_celsius=[float(x.split('_')[1]) for x in conds]
    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH'],row['Tm'], row['fmax'], row['fmin'], T_inv)
    
    ax.set_xlim([13,62])
    ax.set_ylim([-0.1,1.4])

    ax.errorbar(T_celsius, vals, yerr=errors,fmt='.',c=c)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)
    ax.set_title('%s, RMSE: %.3f  [%d%d]'% (row.name, row['RMSE'], row['enforce_fmax'], row['enforce_fmin']))

def plot_renorm_actual_and_expected_fit(row, ax, c='k', conds=None):
    """
    Re-normalized to between 0 and 1
    NOT TESTED
    """
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*(Tm**-1 - x)))
    renorm = lambda x, fmax, fmin: x / (fmax - fmin)  - fmin
    if conds is None:
        conds = [x for x in row.keys() if x.endswith('_norm')]
        
        errs = [x for x in row.keys() if x.endswith('_norm_std')]
    else:
        errs = [x+'_std' for x in conds]

    fmax, fmin = row.fmax, row.fmin
    vals = renorm( np.array(row[conds].values,dtype=float), fmax, fmin )
    errors = np.array(row[errs].values / np.sqrt(row['n_clusters']),dtype=float) / (fmax - fmin)

    T_celsius=[float(x.split('_')[1]) for x in conds]
    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH'],row['Tm'], 1, 0, T_inv)
    
    ax.set_xlim([13,62])
    ax.set_ylim([-0.1,1.4])

    ax.errorbar(T_celsius, vals, yerr=errors,fmt='.',c=c)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)
    ax.set_title('%s, RMSE: %.3f  [%d%d]'% (row.name, row['RMSE'], row['enforce_fmax'], row['enforce_fmin']))


def plot_NUPACK_curve(row, ax, T_celsius=np.arange(20,62.5,2.5), c='k'):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*(Tm**-1 - x)))

    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH_NUPACK'],row['Tm_NUPACK']+273.15, 1, 0, T_inv)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)

    ax.set_xlim([13,62])
    ax.set_ylim([-0.1,1.4])

    ax.set_title('%s, NUPACK dH = %.2f, Tm = %.2f'% (row.name, row['dH_NUPACK'],row['Tm_NUPACK']))


def plot_corrected_NUPACK_curve(row, ax, T_celsius=None, c='k', conds=None):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*(Tm**-1 - x)))

    if T_celsius is None:
        T_celsius = np.arange(20,62.5,2.5)
        ax.set_xlim([13,62])
    else:
        ax.set_xlim([np.min(T_celsius)-5, np.max(T_celsius)+5])
        
    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    GC_content = get_GC_content(row.RefSeq)
    Tm = get_NaCl_adjusted_Tm(row['Tm_NUPACK'], row['dH_NUPACK'], GC_content)
    pred_fit = function(row['dH_NUPACK'],Tm+273.15, 1, 0, T_inv)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)

    
    ax.set_ylim([-0.1,1.4])

    ax.set_title('%s, NUPACK dH = %.2f, Tm = %.2f'% (row.name, row['dH_NUPACK'],Tm))


def plot_candidate_variant_summary(candidate, df_with_targetstruct, df_with_curve, df_with_nupack):
    fig,ax = plt.subplots(1,3,figsize=(12,4))
    draw_struct(df_with_targetstruct.loc[candidate, 'RefSeq'], df_with_targetstruct.loc[candidate, 'TargetStruct'],ax=ax[0])
    plot_actual_and_expected_fit(df_with_curve.loc[candidate,:], ax=ax[1])
    plot_corrected_NUPACK_curve(df_with_nupack.loc[candidate,:], ax=ax[2])

    print('====Library Info===\n', df_with_nupack.loc[candidate,:])
    cols = ['dH', 'Tm', 'dS', 'dG_37', 'dG_37_se_corrected', 'RMSE']
    print('\n====Fit Info===\n', df_with_targetstruct.loc[candidate,cols])
    print('\n%d clusters'%df_with_curve.loc[candidate,'n_clusters'])


def plot_motif_param_errorbar(motif_df, param):
    fig, ax = plt.subplots(figsize=(10,3))
    plt.errorbar(range(len(motif_df)), motif_df[param], motif_df[param+'_se'], fmt='.')
    plt.xticks(range(len(motif_df)), motif_df.index, rotation=20)
    plt.title('WC $%s$ NN parameters' % param)


def plot_fitting_evaluation(fitted_variant_df_list, legend, save_pdf_file=None):
    for col in ['RMSE', 'rsqr', 'chisq', 'red_chisq', 'dH_se', 'Tm_se', 'dG_37_se', 'dS_se']:
        plt.figure()
        for i,df in enumerate(fitted_variant_df_list):
            sns.kdeplot(df[col], color=palette[i])

        plt.legend(legend)
    
    if save_pdf_file is not None:
        save_multi_image(save_pdf_file)
