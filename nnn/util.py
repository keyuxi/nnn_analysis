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
# from arnie.free_energy import free_energy


palette=[
    '#201615',
    '#4e4c4f',
    '#4c4e41',
    '#936d60',
    '#5f5556',
    '#537692',
    '#a3acb1']#cc.glasbey_dark


def save_fig(filename, fig=None):
    if fig is None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        
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
    
def filter_variant_table(df, variant_filter):
    filtered_df = df.query(variant_filter)
    print('%.2f%% variants passed the filter %s' % (100 * len(filtered_df) / len(df), variant_filter))
    return filtered_df

def plot_rep_comparison(r1, r2, param, lim, add_final=False, color='#deb887'):
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
    sns.kdeplot(data=df, x=col+'_x', y=col+'_y', color=color)    
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


def plot_NUPACK_curve(row, ax, T_celsius=np.arange(20,62.5,2.5), c='k', conds=None):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*(Tm**-1 - x)))

    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    pred_fit = function(row['dH_NUPACK'],row['Tm_NUPACK']+273.15, 1, 0, T_inv)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)

    ax.set_xlim([13,62])
    ax.set_ylim([-0.1,1.4])

    ax.set_title('%s, NUPACK dH = %.2f, Tm = %.2f'% (row.name, row['dH_NUPACK'],row['Tm_NUPACK']))


def plot_corrected_NUPACK_curve(row, ax, T_celsius=np.arange(20,62.5,2.5), c='k', conds=None):
    function = lambda dH, Tm, fmax, fmin, x: fmin + (fmax - fmin) / (1 + np.exp(dH/0.00198*(Tm**-1 - x)))

    T_kelvin=[x+273.15 for x in T_celsius]
    T_inv = np.array([1/x for x in T_kelvin])
    GC_content = get_GC_content(row.RefSeq)
    Tm = get_NaCl_adjusted_Tm(row['Tm_NUPACK'], row['dH_NUPACK'], GC_content)
    pred_fit = function(row['dH_NUPACK'],Tm+273.15, 1, 0, T_inv)
    ax.plot(T_celsius, pred_fit, c=c, lw=3)

    ax.set_xlim([13,62])
    ax.set_ylim([-0.1,1.4])

    ax.set_title('%s, NUPACK dH = %.2f, Tm = %.2f'% (row.name, row['dH_NUPACK'],Tm))


def get_GC_content(seq):
    return 100 * np.sum([s in ['G','C'] for s in seq]) / len(seq)


def get_NaCl_adjusted_Tm(Tm, dH, GC, Na_mM=0.025):
    Tmadj = Tm + (-3.22*GC/100 + 6.39)*(np.log(Na_mM/1))
    return Tmadj

def get_dG(dH, Tm, t):
    return dH * (1 - (273.15 + t) / Tm)
    
def get_dof(row):
    n_T = len(np.arange(20, 62.5, 2.5))
    dof = row['n_clusters'] * n_T - 4 + row['enforce_fmax'] + row['enforce_fmin']
    return dof

def get_chi2(row):
    return row['dof'] * row['chisq']

def pass_chi2(row):
    cutoff = chi2.ppf(q=.99, df=row['dof'])
    return row['chi2'] < cutoff

def add_dG_chi2_test(df):
    df['dof'] = df.apply(get_dof, axis=1)
    df['chi2'] = df.apply(get_chi2, axis=1)
    df['pass_chi2'] = df.apply(pass_chi2, axis=1)
    
    n_pass = sum(df['pass_chi2'])
    print('%d / %d, %.2f%% varaints passed the chi2 test' % (n_pass, len(df), 100 * n_pass / len(df)))
    
    df['dG_37'] = get_dG(df['dH'], df['Tm'], t=37)
    df['dG_37_ub'] = get_dG(df['dH_ub'], df['Tm_lb'], t=37)
    df['dG_37_lb'] = get_dG(df['dH_lb'], df['Tm_ub'], t=37)

    return df

"""
def calc_dH_dS_Tm(seq, package='nupack',dna=True):
    '''Return dH (kcal/mol), dS(kcal/mol), Tm (˚C)'''

    T1=0
    T2=50

    dG_37C = free_energy(seq, T=37, package=package, dna=dna)

    dG_1 = free_energy(seq, T=T1, package=package, dna=dna)
    
    dG_2 = free_energy(seq, T=T2, package=package, dna=dna)
    
    dS = -1*(dG_2 - dG_1)/(T2 - T1)
    
    assert((dG_1 + dS*(T1+273.15)) - (dG_2 + dS*(T2+273.15)) < 1e-6)

    dH = dG_1 + dS*(T1+273.15)
    
    if dS != 0:
        Tm = (dH/dS) - 273.15 # to convert to ˚C
    else:
        Tm = np.nan
    #print((dH - dS*273),free_energy(seq, T=0, package=package, dna=dna))
    
    return dH, dS, Tm, dG_37C
"""
def add_p_unfolded_NUPACK(df, T_celcius):
    """
    Calculate p_unfolded at T_celcius and add as a column to df
    """
    def get_p_unfolded(row):
        return 1 / (1 + np.exp(row.dH_NUPACK/0.00198*((row.Tm_NUPACK + 273.15)**-1 - (T_celcius+273.15)**-1)))

    df['p_unfolded_%dC'%T_celcius] = df.apply(get_p_unfolded, axis=1)

    return df


def convert_santalucia_motif_representation(motif):
    strand = motif.split('_')
    return(f'{strand[0]}_{strand[1][::-1]}')

def read_santalucia_df(santalucia_file):
    santa_lucia = pd.read_csv(santalucia_file, sep='\t')
    santa_lucia['motif'] = santa_lucia['motif_paper'].apply(convert_santalucia_motif_representation)
    santa_lucia.drop(columns='motif_paper', inplace=True)
    
    return santa_lucia

def read_fitted_variant(filename, filter=True, annotation=None):
    """
    Args:
        annotation - df, if given, merge onto fitted variant file
    """
    df = pd.read_csv(filename, sep='\t').set_index('SEQID')
    if 'chisquared_all_clusters' in df.columns:
        df.rename(columns={'chisquared_all_clusters': 'chisq'}, inplace=True)
    df.rename(columns={s: s.replace('_final', '') for s in df.columns if s.endswith('_final')}, inplace=True)

    # Add dG and chi2
    if not 'dG_37' in df.columns:
        df = add_dG_chi2_test(df)

    # Add standard error columns for params
    sqrt_n = np.sqrt(df['n_clusters'])
    for c in df.columns:
        if (c.endswith('_std') and (not c.endswith('_norm_std'))):
            df[c.replace('_std', '_se')] = df[c] / sqrt_n

    # Filter variants
    if filter:
        variant_filter = "n_clusters > 5 & dG_37_se < 0.2 & Tm_se < 2.5 & dH_se < 2.5 & RMSE < 0.5"
        df = filter_variant_table(df, variant_filter)

    if annotation is not None:
        df = df.join(annotation, how='left')

    return df


def read_annotation(annotation_file, mastertable_file):
    """
    Returns:
        annotation - df indexed on 'SEQID' with construct class information
    """
    annotation = pd.read_csv(annotation_file, sep='\t').set_index('SEQID')
    annotation.rename(columns={c: c+'_NUPACK' for c in ['dH', 'dS', 'Tm', 'dG_37C']}, inplace=True)
    annotation.rename(columns={'dG_37C_NUPACK': 'dG_37_NUPACK'}, inplace=True)

    mastertable = pd.read_csv(mastertable_file, sep='\t')
    annotation = annotation.reset_index().merge(mastertable[['Series', 'ConstructClass']], on='Series', how='left').set_index('SEQID')

    return annotation


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

def print_candidate_variant_summary(candidate, df_with_targetstruct, df_with_curve, df_with_nupack):
    fig,ax = plt.subplots(1,3,figsize=(12,4))
    draw_struct(df_with_targetstruct.loc[candidate, 'RefSeq'], df_with_targetstruct.loc[candidate, 'TargetStruct'],ax=ax[0])
    plot_actual_and_expected_fit(df_with_curve.loc[candidate,:], ax=ax[1])
    plot_corrected_NUPACK_curve(df_with_nupack.loc[candidate,:], ax=ax[2])

    print('====Library Info===\n', df_with_nupack.loc[candidate,:])
    cols = ['dH', 'Tm', 'dS', 'dG_37', 'dG_37_se_corrected', 'RMSE']
    print('\n====Fit Info===\n', df_with_targetstruct.loc[candidate,cols])
    print('\n%d clusters'%df_with_curve.loc[candidate,'n_clusters'])