"""
Functions that all other modules use.
    - save_fig
    - get_* for calculating things
    - format conversion functions
    - other handy lil functions
"""


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

    figdir, _ = os.path.split(filename)
    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    if fig is None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def convert_santalucia_motif_representation(motif):
    strand = motif.split('_')
    return(f'{strand[0]}_{strand[1][::-1]}')



def filter_variant_table(df, variant_filter):
    filtered_df = df.query(variant_filter)
    print('%.2f%% variants passed the filter %s' % (100 * len(filtered_df) / len(df), variant_filter))
    return filtered_df


def get_GC_content(seq):
    return 100 * np.sum([s in ['G','C'] for s in str(seq)]) / len(str(seq))


def get_Na_adjusted_Tm(Tm, dH, GC, Na=0.075, from_Na=1.0):
    Tmadj = Tm + (-3.22*GC/100 + 6.39)*(np.log(Na/from_Na))
    return Tmadj

def get_dG(dH, Tm, celsius):
    return dH * (1 - (273.15 + celsius) / (273.15 + Tm))
    
def get_dof(row):
    n_T = len(np.arange(20, 62.5, 2.5))
    dof = row['n_clusters'] * n_T - 4 + row['enforce_fmax'] + row['enforce_fmin']
    return dof

def get_red_chisq(row):
    return row['chisq'] / row['dof']

def pass_chi2(row):
    cutoff = chi2.ppf(q=.99, df=row['dof'])
    return row['chisq'] < cutoff

def add_dG_37(df):
    df['dG_37'] = get_dG(df['dH'], df['Tm'], t=37)
    df['dG_37_ub'] = get_dG(df['dH_ub'], df['Tm_lb'], t=37)
    df['dG_37_lb'] = get_dG(df['dH_lb'], df['Tm_ub'], t=37)

def add_chisq_test(df):
    df['red_chisq'] = df.apply(get_red_chisq, axis=1)
    df['pass_chi2'] = df.apply(pass_chi2, axis=1)
    
    n_pass = sum(df['pass_chi2'])
    print('%d / %d, %.2f%% varaints passed the chi2 test' % (n_pass, len(df), 100 * n_pass / len(df)))
    

    return df

def add_p_unfolded_NUPACK(df, T_celcius, sodium=1.0):
    """
    Calculate p_unfolded at T_celcius and add as a column to df
    """
    def get_p_unfolded(row):
        Tm_NUPACK = get_Na_adjusted_Tm(row.Tm_NUPACK, row.dH_NUPACK, get_GC_content(row.RefSeq), Na=sodium)
        return 1 / (1 + np.exp(row.dH_NUPACK/0.00198*((Tm_NUPACK + 273.15)**-1 - (T_celcius+273.15)**-1)))

    df['p_unfolded_%dC'%T_celcius] = df.apply(get_p_unfolded, axis=1)

    return df

def get_ddX(df, param='dG_37', by='ConstructType'):
    class_median = df.groupby(by).apply('median')[param]
    return df.apply(lambda row: row[param] - class_median[row[by]], axis=1)
    

def get_symmetric_struct(len_seq, len_loop):
    return '('*int((len_seq - len_loop)/2) + '.'*len_loop + ')'*int((len_seq - len_loop)/2)

def get_target_struct(row):

    def get_mismatch_struct(seq, mismatch_list):
        """
        Assumes all mismatches constructs are symmetric
        """
        target_struct_list = list(get_symmetric_struct(len(seq), 4))
        for i in range(int((len(seq)-4)/2)):
            if seq[i] + seq[-1-i] in mismatch_list:
                target_struct_list[i] = '.'
                target_struct_list[-1-i] = '.'
        target_struct = ''.join(target_struct_list)
        return target_struct

    series = row['Series']
    construct_type = row['ConstructType']
    seq = row['RefSeq']
    if series in ['WatsonCrick', 'TETRAloop'] or construct_type in ['SuperStem']:
        target_struct = get_symmetric_struct(len(seq), 4)
    elif series == 'TRIloop':
        target_struct = get_symmetric_struct(len(seq), 3)
    elif series == 'VARloop':
        target_struct = get_symmetric_struct(len(seq), len(seq) - 4)
    elif series in ['REPeatControls', 'PolyNTControls']:
        target_struct = '.'* len(seq)
    elif construct_type.startswith('StemDangle'):
        topScaffold = row['topScaffold']
        stem_pos = seq.find(topScaffold[:len(topScaffold)//2])
        target_struct = '.'*stem_pos + get_symmetric_struct(len(topScaffold) + 4, 4) + '.'*int(len(seq) - stem_pos - len(topScaffold) - 4)
    elif (series == 'MisMatches') or (construct_type == 'BaeControls'):
        mismatch_list = []
        for x in 'ATCG':
            for y in 'ATCG':
                if not x+y in ['AT', 'TA', 'CG', 'GC']:
                    mismatch_list.append(x+y)
        
        target_struct = get_mismatch_struct(seq, mismatch_list)
    else:
        # generated elsewhere
        target_struct = '.'*len(seq)

    return target_struct

def get_curve_pred(row, conds=None):
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

    return pred_fit, vals, errors


def get_mfe_struct(seq, return_free_energy=False, verbose=False, celsius=0.0, sodium=1.0):
    my_model = nupack.Model(material='DNA', celsius=celsius, sodium=sodium, magnesium=0.0)
    mfe_structures = nupack.mfe(strands=[seq], model=my_model)
    mfe_struct = str(mfe_structures[0].structure)

    if verbose:
        print('Number of mfe structures:', len(mfe_structures))
        print('Free energy of MFE proxy structure: %.2f kcal/mol' % mfe_structures[0].energy)
        print('MFE proxy structure in dot-parens-plus notation: %s' % mfe_structures[0].structure)

    if return_free_energy:
        return mfe_struct, mfe_structures[0].energy
    else:
        return mfe_struct


def get_seq_ensemble_dG(seq, celsius, sodium=1.0, verbose=False):
    my_model = nupack.Model(material='DNA', celsius=celsius, sodium=sodium, magnesium=0.0)
    _, dG = nupack.pfunc(seq, model=my_model)
    return dG


def get_seq_structure_dG(seq, structure, celsius, sodium=1.0):
    my_model = nupack.Model(material='DNA', celsius=celsius, sodium=sodium, magnesium=0.0)
    return nupack.structure_energy([seq], structure=structure, model=my_model)

def get_seq_end_pair_prob(seq:str, celsius:float, sodium=1.0, n_pair:int=2) -> float:
    """
    Pr[either last or second to last basepair in the hairpin paired]
    """
    my_model = nupack.Model(material='DNA', celsius=celsius, sodium=sodium, magnesium=0.0)
    pair_mat = nupack.pairs([seq], model=my_model).to_array()
    if n_pair == 1:
        return pair_mat[0,-1]
    elif n_pair == 2:
        try:
            p1, p2 = pair_mat[0,-1], pair_mat[1,-2]
        except:
            p1, p2 = np.nan, np.nan

        return p1 + p2 - p1 * p2
    else:
        raise ValueError("n_pair value not allowed")


def get_nupack_dH_dS_Tm_dG_37(seq, struct):
    '''Return dH (kcal/mol), dS(kcal/mol), Tm (˚C), dG_37(kcal/mol)'''
    T1=0
    T2=50

    dG_37 = get_seq_structure_dG(seq, struct, 37)

    dG_1 = get_seq_structure_dG(seq, struct, T1)
    dG_2 = get_seq_structure_dG(seq, struct, T2)

    dS = -1*(dG_2 - dG_1)/(T2 - T1)
    assert((dG_1 + dS*(T1+273.15)) - (dG_2 + dS*(T2+273.15)) < 1e-6)
    
    dH = dG_1 + dS*(T1+273.15)
    
    if dS != 0:
        Tm = (dH/dS) - 273.15 # to convert to ˚C
    else:
        Tm = np.nan
    
    return dH, dS, Tm, dG_37



def is_diff_nupack(df, param):
    return df.apply(lambda row: (row[param+'_lb'] > row[param+'_NUPACK_salt_corrected']) or (row[param+'_ub'] < row[param+'_NUPACK_salt_corrected']), axis=1)


# Fluor simulation related

def get_fluor_distance_from_structure(structure: str):
    return len(structure) - len(structure.strip('.'))
"""
# Old nupack dH dS Tm code, has problems
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
