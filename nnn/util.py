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
        
    
def filter_variant_table(df, variant_filter):
    filtered_df = df.query(variant_filter)
    print('%.2f%% variants passed the filter %s' % (100 * len(filtered_df) / len(df), variant_filter))
    return filtered_df


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

    return annotation


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
