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
#import colorcet as cc
from scipy.stats import chi2, pearsonr, norm
# from sklearn.metrics import r2_score
# from lmfit.models import PowerLawModel
from ipynb.draw import draw_struct
import nupack
from matplotlib.backends.backend_pdf import PdfPages
# from arnie.free_energy import free_energy
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

palette=[
    '#201615',
    '#4e4c4f',
    '#4c4e41',
    '#936d60',
    '#5f5556',
    '#537692',
    '#a3acb1']#cc.glasbey_dark

complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', '-':'-'}


def save_fig(filename, fig=None):

    figdir, _ = os.path.split(filename)
    if not os.path.isdir(figdir):
        os.makedirs(figdir)

    if fig is None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        
def save_multi_image(filename, figs=None):
    pp = PdfPages(filename)
    if figs is None:
        fig_nums = plt.get_fignums()
        figs = [plt.figure(n) for n in fig_nums]
        
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def absolute_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
            

def rcompliment(seq):
    return "".join(complement.get(base, base) for base in reversed(seq))

def nrcompliment(seq):
    return "".join(complement.get(base, base) for base in seq)
            
def convert_santalucia_motif_representation(motif):
    strand = motif.split('_')
    return(f'{strand[0]}_{strand[1][::-1]}')



def filter_variant_table(df, variant_filter):
    filtered_df = df.query(variant_filter)
    print('%.2f%% variants passed the filter %s' % (100 * len(filtered_df) / len(df), variant_filter))
    return filtered_df


def get_GC_content(seq):
    return 100 * np.sum([s in ['G','C'] for s in str(seq)]) / len(str(seq))


def get_Na_adjusted_Tm(Tm, dH, GC, Na=0.088, from_Na=1.0):
    # Tmadj = Tm + (-3.22*GC/100 + 6.39)*(np.log(Na/from_Na))
    Tmadj_inv = (1 / (Tm + 273.15) + (4.29 * GC/100 - 3.95) * 1e-5 * np.log(Na / from_Na)
        + 9.4 * 1e-6 * (np.log(Na)**2 - np.log(from_Na)**2))
    Tmadj = 1 / Tmadj_inv - 273.15
    
    return Tmadj

def get_dG(dH, Tm, celsius):
    return dH * (1 - (273.15 + celsius) / (273.15 + Tm))
    
def get_Tm(dH, dG, celsius=37):
    return (273.15 + celsius) / (1 - dG / dH) - 273.15

def get_dS(dH, Tm):
    return dH / (Tm + 273.15)
    
def get_dG_err(dH, dH_err, Tm, Tm_err, celsius):
    """
    Error propagation
    dG = dH - TdS
    dG_err = sqrt(dH_err^2 + (TdS)_err^2)
           = sqrt(dH_err^2 + (T*dS_err)^2)
    """
    dS_err = get_dS_err(dH, dH_err, Tm, Tm_err)
    return np.sqrt(dH_err**2 + ((celsius + 273.15) * dS_err)**2)
    
def get_dS_err(dH, dH_err, Tm, Tm_err):
    """
    Error propagation
    dS = dH / Tm
    dS_err = - dS * sqrt((dH_err / dH)^2 + (Tm_err / Tm)^2)
    """
    dS = get_dS(dH, Tm)
    return  - dS * np.sqrt((dH_err / dH)**2 + (Tm_err / (Tm + 273.15))**2)
    
def get_Na_adjusted_dG(Tm, dH, GC, celsius, Na=0.088, from_Na=1.0):
    Tm_adjusted = get_Na_adjusted_Tm(Tm, dH, GC, Na, from_Na)
    return get_dG(dH, Tm_adjusted, celsius)
    
def get_Na_adjusted_dG_37(Tm, dH, GC, Na=0.088, from_Na=1.0):
    Tm_adjusted = get_Na_adjusted_Tm(Tm, dH, GC, Na, from_Na)
    return get_dG(dH, Tm_adjusted, 37)

def get_Na_adjusted_param(Na=1.0, from_Na=0.088, **data_dict):
    """
    data_dict - dict, with keys dH, Tm, and seq
    """
    dH, Tm = data_dict['dH'], data_dict['Tm']
    GC_content = get_GC_content(data_dict['seq'])
    Tm_adjusted = get_Na_adjusted_Tm(Tm, dH, GC_content, Na, from_Na)
    dG_37_adjusted = get_dG(dH, Tm_adjusted, 37)
    return dict(dH=dH, dS=dH/Tm_adjusted, Tm=Tm_adjusted, dG_37=dG_37_adjusted)
    
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
    
def dotbracket2edgelist(dotbracket_str:str):

    assert dotbracket_str.count('(') == dotbracket_str.count(')'), \
        'Number of "(" and ")" should match in %s' % dotbracket_str

    # Backbone edges
    N = len(dotbracket_str)
    edge_list = [[i, i+1] for i in range(N-1)]

    # Hydrogen bonds
    flag3p = N - 1
    for i,x in enumerate(dotbracket_str):
        if x == '(':
            for j in range(flag3p, i, -1):
                if dotbracket_str[j] == ')':
                    edge_list.append([i, j])
                    flag3p = j - 1
                    break

    return edge_list
    

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


def get_mfe_struct(seq, return_free_energy=False, verbose=False, celsius=0.0, sodium=1.0, param_set='dna04'):
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0)
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


def get_seq_ensemble_dG(seq, celsius, sodium=1.0, verbose=False, param_set='dna04'):
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0)
    _, dG = nupack.pfunc(seq, model=my_model)
    return dG


def get_seq_structure_dG(seq, structure, celsius, sodium=1.0, param_set='dna04', **kwargs):
    """
    **kwargs passed to nupack.Model
    """
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0, **kwargs)
    if isinstance(seq, str):
        return nupack.structure_energy([seq], structure=structure, model=my_model)
    else:
        return nupack.structure_energy(seq, structure=structure, model=my_model)


def get_seq_end_pair_prob(seq:str, celsius:float, sodium=1.0, n_pair:int=2, param_set='dna04') -> float:
    """
    Pr[either last or second to last basepair in the hairpin paired]
    """
    my_model = nupack.Model(material=param_set, celsius=celsius, sodium=sodium, magnesium=0.0)
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


def get_nupack_dH_dS_Tm_dG_37(seq, struct, sodium=1.0, return_dict=False, **kwargs):
    '''
    Return dH (kcal/mol), dS(kcal/mol), Tm (˚C), dG_37(kcal/mol)
    Use the better sodium correction equation
    '''
    T1=0
    T2=50

    dG_37 = get_seq_structure_dG(seq, struct, 37, **kwargs)

    dG_1 = get_seq_structure_dG(seq, struct, T1, **kwargs)
    dG_2 = get_seq_structure_dG(seq, struct, T2, **kwargs)

    dS = -1*(dG_2 - dG_1)/(T2 - T1)
    assert((dG_1 + dS*(T1+273.15)) - (dG_2 + dS*(T2+273.15)) < 1e-6)
    
    dH = dG_1 + dS*(T1+273.15)
    
    if dS != 0:
        Tm = (dH/dS) - 273.15 # to convert to ˚C
        Tm = get_Na_adjusted_Tm(Tm=Tm, dH=dH, GC=get_GC_content(seq), 
                                    Na=sodium, from_Na=1.0)
        dG_37 = get_dG(Tm=Tm, dH=dH, celsius=37)
        dS = dH / (Tm + 273.15)
    else:
        Tm = np.nan
    if return_dict:
        return dict(dH=dH, dS=dS, Tm=Tm, dG_37=dG_37)
    else:
        return [dH, dS, Tm, dG_37]



def is_diff_nupack(df, param):
    return df.apply(lambda row: (row[param+'_lb'] > row[param+'_NUPACK_salt_corrected']) or (row[param+'_ub'] < row[param+'_NUPACK_salt_corrected']), axis=1)

def add_intercept(arr):
    """
    Helper function for fitting with LinearRegressionSVD
    """
    arr_colvec = arr.reshape(-1, 1)
    return np.concatenate((arr_colvec, np.ones_like(arr_colvec)), axis=1)

class LinearRegressionSVD(LinearRegression):
    """
    The last feature is intercept
    self.intercept_ is set to 0 to keep consistency
    y does not have nan ('raise' behavior)
    """
    def __init__(self, param='dG_37', **kwargs):
        super().__init__(fit_intercept=False, **kwargs)
        self.intercept_ = 0.0
        self.param = param
                
    def fit(self, X:np.array, y:np.array, y_err:np.array, sample_weight=None, 
            feature_names=None, singular_value_rel_thresh:float=1e-15, skip_rank:bool=False):

        if sample_weight is not None:
            assert len(sample_weight) == len(y)
            y_err = 1 / sample_weight
            
        A = X / (y_err.reshape(-1,1))
        
        if not skip_rank:
            rank_A = np.linalg.matrix_rank(A)
            n_feature = A.shape[1]
            if rank_A < n_feature:
                print('Warning: Rank of matrix A %d is smaller than the number of features %d!' % (rank_A, n_feature))
            
        b = (y / y_err).reshape(-1,1)
        u,s,vh = np.linalg.svd(A, full_matrices=False)
        s_inv = 1/s
        s_inv[s < s[0]*singular_value_rel_thresh] = 0
        self.coef_se_ = np.sqrt(np.sum((vh.T * s_inv.reshape(1,-1))**2, axis=1))
        self.coef_ = (vh.T @ np.diag(s_inv) @ u.T @ b).flatten()
        
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        
        self.metrics = self.calc_metrics(X, y, y_err)
    
    
    def calc_metrics(self, X, y, y_err):        
        """
        Returns:
            metrics - Dict[str: float], ['rsqr', 'rmse', 'dof', 'chisq', 'redchi']
        """
        y_pred = self.predict(X=X)
        ss_total = np.nansum((y - y.mean())**2)
        ss_error = np.nansum((y - y_pred)**2)
        
        rsqr = 1 - ss_error / ss_total
        rmse = np.sqrt(ss_error / len(y))
        dof = len(y) - len(self.coef_)
        chisq = np.sum((y - y_pred.reshape(-1,1))**2 / y_err**2)
        redchi = chisq / dof

        metrics = dict(rsqr=rsqr, rmse=rmse, dof=dof, chisq=chisq, redchi=redchi)
        
        return metrics
    
    
    def fit_with_some_coef_fixed(self, X:np.array, y:np.array, y_err:np.array,
            feature_names, fixed_feature_names, coef_df, coef_se_df=None,# index_col='motif',
            singular_value_rel_thresh=1e-15, debug=False):
        """
        Fix a given list of coef of features and fit the rest. Calls self.fit()
        Args:
            feature_names - list like, ALL the feats (column names of X)
            fixed_feature_names - list like, names of the indices to look up from coef_df
            coef_df - pd.DataFrame, indices are feature names to look up, only one column with the coef
            coef_se_df - pd.DataFrame, indices are feature names to look up, only one column with the coef, 
                optional. If not given, se of the known parameters are presumably set to 0
        """
        
        A = X / (y_err.reshape(-1,1))
        b = (y / y_err).reshape(-1,1)
        
        known_param = [f for f in feature_names if f in fixed_feature_names]
        known_param_mask = np.array([(f in fixed_feature_names) for f in feature_names], dtype=bool)
        A_known, A_unknown = A[:, known_param_mask], A[:, ~known_param_mask]
        if debug:
            print('known_param_mask: ', np.sum(known_param_mask), known_param)
            print('A_unknown, A_known: ', A_unknown.shape, A_known.shape)
        n_feature = A.shape[1]
        n_feature_to_fit = A_unknown.shape[1]
        # x_unknown is to be solved; x_known are the known parameters
        x_known = coef_df.loc[known_param, :].values.flatten()
        if debug:
            print('x_known: ', x_known.shape)
        b_tilde = b - A_known @ x_known.reshape(-1, 1)
        
        rank_A1 = np.linalg.matrix_rank(A_unknown)
        if rank_A1 < n_feature_to_fit:
            print('Warning: Rank of matrix A_unknown, %d, is smaller than the number of features %d!' % (rank_A1, n_feature_to_fit))
        
        # initialize
        self.coef_ = np.zeros(n_feature)
        self.coef_se_ = np.zeros(n_feature)
        
        u,s,vh = np.linalg.svd(A_unknown, full_matrices=False)
        s_inv = 1/s
        s_inv[s < s[0]*singular_value_rel_thresh] = 0
        x_unknown = (vh.T @ np.diag(s_inv) @ u.T @ b_tilde).flatten()
        x_se_unknown = np.sqrt(np.sum((vh.T * s_inv.reshape(1,-1))**2, axis=1))
        
        self.coef_[known_param_mask] = x_known
        self.coef_[~known_param_mask] = x_unknown
        self.coef_se_[~known_param_mask] = x_se_unknown
        if coef_se_df is not None:
            self.coef_se_[known_param_mask] = coef_se_df.loc[known_param, :].values.flatten()
        
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        
        self.metrics = self.calc_metrics(X, y, y_err)
    
        
    def predict_err(self, X):
        return (X @ self.coef_se_ .reshape(-1,1)).flatten()
        
        
    def fit_intercept_only(self, X, y):
        self.coef_[-1] = np.mean(y - X[:,:-1] @ self.coef_[:-1].reshape(-1,1))
        
        
    def set_coef(self, feature_names, coef_df, index_col='index'):
        """
        Force set coef of the model to that supplied in `coef_df`,
        sorted in the order of `feature_names`.
        For instance, external parameter sets like SantaLucia.
        `coef_se_` is set to 0.
        Args:
            feature_names - array-like of str, **WITH** the 'intercept' at the end of feature matrices
            coef_df - df, with a column named `self.param` e.g. dG_37
                and a columns called `index_col` to set the names of the coef to.
            index_col - str. col name to set names of the coef to.
                if 'index', use index.
        """
        if index_col != 'index':
            coef_df = coef_df.set_index(index_col)
            
        self.coef_ = np.append(coef_df.loc[feature_names[:-1],:][self.param].values, 0)
        self.coef_se_ = np.zeros_like(self.coef_)
        self.feature_names_in_ = feature_names
        
    @property
    def intercept(self):
        return self.coef_[-1]
    @property
    def intercept_se(self):
        return self.coef_se_[-1]
    @property
    def coef_df(self):
        return pd.DataFrame(index=self.feature_names_in_,
                            data=self.coef_.reshape(-1,1), columns=[self.param])
    @property
    def coef_se_df(self):
        return pd.DataFrame(index=self.feature_names_in_,
                            data=self.coef_se_.reshape(-1,1), columns=[self.param + '_se'])
        
# Fluor simulation related

def get_fluor_distance_from_structure(structure: str):
    return len(structure) - len(structure.strip('.'))
    
    
def get_duplex_row(hairpin_row):
    """
    Assume loop size is 4
    """
    row = hairpin_row.copy()
    stem_len = int((len(row.RefSeq) - 4.0) / 2.0)
    row.TargetStruct = row.TargetStruct.replace('....','+')
    row.RefSeq = row.RefSeq[:stem_len] + '+' + row.RefSeq[-stem_len:]
    return row
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

def rmse(y1, y2):
    return np.sqrt(np.mean(np.square(y1 - y2)))
def mae(y1, y2):
    return np.mean(np.abs(y1 - y2))