from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')
sns.set_context('paper')
from RiboGraphViz import RGV
from RiboGraphViz import LoopExtruder, StackExtruder

from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from .util import *
from . import plotting
from . import feature_list


def search_GU_pairs(seq):
    GU_counter=0
    for i, char in enumerate(seq[:7]):

        if char=='G':
            if seq[-1-i]=='T':
                GU_counter+=1
        elif char=='T':
            if seq[-1-i]=='G':
                GU_counter+=1
    return GU_counter


def get_model_metric(y, y_err, preds, n_feature):
    n = len(y)
    m = {}
    rss = np.sum(np.square(y - preds))
    m['bic'] = n*np.log(rss/n) + n_feature*np.log(n) 
    m['aic'] = n*np.log(rss/n) + 2*n_feature
    m['rmse'] = np.sqrt(np.mean(np.square(y - preds)))
    m['chi2'] = np.sqrt(np.mean(np.square(y - preds) / np.square(y_err)))
    m['dof'] = n - n_feature

    return m

def get_unknown_yerr_motif_se(X, res):
    """
    Calculate heteroskedasticity robust standard errors for fitted parameters
    """
    XT_X_1 = np.linalg.inv(X.T @ X)
    var_coef = XT_X_1 @ X.T @ np.diag(res**2) @ X @ XT_X_1
    return np.sqrt(np.diag(var_coef))


def solve_linear_regression(X, y, y_err, singular_value_rel_thresh=1e-15,
                        verbose=False):
    """
    Solves linear reggression with measurement error using SVD.
    With more plotting functions for sanity checks.
    Args:
        X is the feature matrix, may have a feature for intercept
        y - the measured parameters to be 
        y_err - a vector of the error of the measurements of each datapoint
    Return:

    """
    A = X / (y_err.reshape(-1,1))
    b = (y / y_err).reshape(-1,1)
    if verbose:
        print('Rank of the design matrix A is %d / %d'%(np.linalg.matrix_rank(A), A.shape[1]))
        print('Condition number of the design matrix A is %.2f'%np.linalg.cond(A))

    u,s,vh = np.linalg.svd(A, full_matrices=False)
    if verbose:
        # plot the singular values
        _, ax = plt.subplots()
        ax.plot(s / s[0], 'gx')
        ax.set_ylim(bottom=0)
        plt.show()
        
    s_inv = 1/s
    s_inv[s < s[0]*singular_value_rel_thresh] = 0
    coef_se = np.sqrt(np.sum((vh.T * s_inv.reshape(1,-1))**2, axis=1))
    # equivalent to np.linalg.pinv(A) @ b
    # doing this since we've already SVD'ed
    coef = (vh.T @ np.diag(s_inv) @ u.T @ b).flatten()
    
    if verbose:
        _, ax = plt.subplots(figsize=(10,6))
        ax.errorbar(np.arange(len(coef)), coef, coef_se, fmt='k.', capsize=3)
        if isinstance(X, pd.DataFrame):
            ax.set_xticks(np.arange(len(coef)))
            ax.set_xticklabels(X.columns, rotation=30)
        plt.show()
    
    return coef, coef_se
    
 


def get_feature_count_matrix(df, feature_method='get_stack_feature_list', **kwargs):
        """
        Args:
            df - (n_variant, N) df
            **kwargs - passed to `feature_method`
        Returns:
            feats - a (n_variant, n_feature) feature df
        """
        df['feature_list'] = df.apply((lambda row: getattr(feature_list, feature_method)(row, **kwargs)), axis=1)
        cv = CountVectorizer(token_pattern=r"\b[ATCGNxy().+_]+\s", lowercase=False)
        feats = pd.DataFrame.sparse.from_spmatrix(cv.fit_transform([' '.join(x)+' ' for x in df['feature_list']]),
                        index=df.index, columns=[x.strip() for x in cv.get_feature_names_out()])

        # Remove features that every construct contains and is not intercept
        for k in feats.keys():
            if len(feats[k].unique())==1 and k!='INTERCEPT':
                feats = feats.drop(columns=[k])
                
        if 'INTERCEPT' in feats.columns:
            intercept = feats.pop('INTERCEPT')
            feats['intercept'] = intercept
            

        return feats


def fit_linear_motifs(df, feature_method='get_stack_feature_list',
                      param='dG_37', err='_se_corrected', lim=None,
                      stack_sizes=[1,2,3], fit_intercept=False):

    fig, ax = plt.subplots(1, 3, figsize=(9,4), sharey=True)

    N_SPLITS = 5
    y = df[param]
    y_err = df[param + err]

    titles = {1:'N_N features (stacks only)', 2:'NN_NN (Nearest neighbor)', 3:'N(3)_N(3)', 4:'N(4)_N(4)', 5:'N(5)_N(5)'}

    coef_dfs=[]
    for i, stack_size in enumerate(stack_sizes):
        
        feats = get_feature_count_matrix(df, feature_method=feature_method, stack_size=stack_size)
        n_feature = feats.shape[1]
        #Perform linear regression fit
        mdl = Ridge(fit_intercept=fit_intercept)

        X = feats.values

        results = cross_validate(mdl, X, y, cv=N_SPLITS, return_estimator=True)
        coef_df = pd.DataFrame(columns=['motif', param])
        for x in results['estimator']:
            for j in range(len(feats.columns)):
                coef_df = coef_df.append({'motif': feats.columns[j], param: x.coef_[j]}, ignore_index=True)
        
        preds = cross_val_predict(mdl, X, y, cv=N_SPLITS)
        df['tmp_pred'] = preds
        residuals = y.values - preds
        motif_se = get_motif_se(X, y_err)
        # coef_df[param+'_se'] = motif_se
        # print(motif_se)

        coef_dfs.append(coef_df)
        
        m = get_model_metric(y, y_err, preds, n_feature)

        plt.subplot(1,3,stack_size)
        #errorbar(y, preds, xerr=y_err, fmt='.', alpha=0.1,zorder=0, color='k')
        hue_order = ['WC_5ntstem', 'WC_6ntstem', 'WC_7ntstem']
        sns.scatterplot(x=param, y='tmp_pred', data=df, hue='ConstructType', hue_order=hue_order, linewidth=0, s=10, alpha=0.6, palette='plasma')
        plt.xlabel('Fit '+param)
        plt.ylabel('CV-test-split predicted '+param)
        plt.title("%s, %d features\n RMSE: %.2f, $\chi^2$: %.2f \n BIC: %.2f" % (titles[stack_size], n_feature, m['rmse'], m['chi2'], m['bic']))
        
        if lim is not None:
            plt.plot(lim,lim,'--',color='grey',zorder=0)
            plt.xlim(lim)
            plt.ylim(lim)
        if i!=0: plt.legend([],frameon=False)

        plt.tight_layout()

    return coef_dfs, motif_se, feats, preds, results


def fit_NN_cv(df, feature_method='get_stack_feature_list_simple_loop', stack_size=2,
           param='dG_37', err='_se_corrected', lim=None,
           fit_intercept=False):


    N_SPLITS = 5
    y = df[param]
    y_err = df[param + err]
    y_weight = 1 / y_err**2

    feats = get_feature_count_matrix(df, feature_method=feature_method, stack_size=stack_size)
    n_feature = feats.shape[1]

    mdl = Ridge(fit_intercept=fit_intercept)
    X = feats.values
    results = cross_validate(mdl, X, y, cv=N_SPLITS, return_estimator=True, fit_params={'sample_weight': y_weight})
    preds = cross_val_predict(mdl, X, y, cv=N_SPLITS)
    m = get_model_metric(y, y_err, preds, n_feature)
    motif_se = get_motif_se(X, y_err)

    coef_df = pd.DataFrame(columns=['motif', param])
    for estimator in results['estimator']:
        for j in range(len(feats.columns)):
            coef_df = coef_df.append({'motif': feats.columns[j], param: estimator.coef_[j]}, ignore_index=True)

    motif_df = coef_df.groupby('motif').median()
    motif_df[param+'_cv_std'] = coef_df.groupby('motif').std()
    # assert motif_df.index.tolist() == feats.columns.tolist()
    motif_df = motif_df.join(pd.DataFrame(data=motif_se, index=feats.columns, columns=[param+'_se']))
    df['tmp_pred'] = preds

    hue_order = ['WC_5ntstem', 'WC_6ntstem', 'WC_7ntstem']
    fig, ax = plt.subplots(figsize=(4,4))
    sns.scatterplot(x=param, y='tmp_pred', data=df, hue='ConstructType', hue_order=hue_order,
                    linewidth=0, s=10, alpha=0.5, palette='plasma', ax=ax)
    plt.xlabel('Fit '+ param)
    plt.ylabel('CV-test-split predicted '+param)
    plt.title("%d features\n RMSE: %.2f, $\chi^2$: %.2f, \n BIC: %.2f" % (n_feature, m['rmse'], m['chi2'], m['bic']))

    if lim is not None:
        plt.plot(lim,lim,'--',color='grey',zorder=0)
        plt.xlim(lim)
        plt.ylim(lim)

    return motif_df, feats


def compare_fit_with_santalucia(df, santa_lucia, params=['dH', 'dS', 'dG_37']):

    for param in params:
        coef_dfs, _, _ = fit_NN_cv(df, param=param, err='_se')
        fit_param = pd.DataFrame(coef_dfs[1].groupby('motif').apply(np.nanmean), columns=[param]).join(
             pd.DataFrame(coef_dfs[1].groupby('motif').apply(lambda x: np.nanstd(x)/np.sqrt(5)), columns=[param+'_cv_se']))
        santa_lucia = santa_lucia.merge(fit_param, on='motif', how='inner', suffixes=('_SantaLucia', '_MANIfold'))

    return santa_lucia
    
def get_X_y(arr, split_dict, param, feats=None, split='train'):
    """
    Returns:
        feats - DataFrame, needs .values when fed into LinearRegressionSVD
    """
    assert split_dict is not None
    seqids = split_dict[split+'_ind']
    if feats is not None:
        X = feats.loc[seqids, :].values
    else:    
        feats = mf.get_feature_count_matrix(arr, feature_method='get_feature_list', 
                                            fit_intercept=False, symmetry=False)
        X = feats.values
    df = arr.loc[seqids, :]
    y = df[param].values
    y_err = df[param+'_se'].values
    
    return dict(X=X, y=y, y_err=y_err, feature_names=feats.columns.tolist(), param=param, split=split)
    
    
def fit_param(arr, data_split_dict, param, feats, ax, mode='val'):
    color_dict = dict(dH='c', Tm='cornflowerblue', dG_37='teal', dS='steelblue')
    train_data = get_X_y(arr, data_split_dict, param=param, feats=feats, split='train')
    val_data = get_X_y(arr, data_split_dict, param=param, feats=feats, split=mode)
    
    lr = LinearRegressionSVD()
    lr.fit(train_data['X'], train_data['y'], train_data['y_err'], feature_names=train_data['feature_names'],
           skip_rank=False)
    
    plotting.plot_truth_predict(lr, val_data, ax=ax, title='NNN OLS model',
                            color=color_dict[param], alpha=.05)
    
    return lr
    
    
def pred_variant(q_row, lr, verbose=False):
    """
    Predict a single variant given a linear regression model
    """
    q_feat_list = feature_list.get_feature_list(q_row, fit_intercept=False, symmetry=False, sep_base_stack=True)

    pred = lr.coef_df.loc[q_feat_list].sum().values[0]
    gt = q_row[lr.param]
    
    if verbose:
        print("%.4f\t%.4f" % (pred, gt))
    
    return dict(pred=pred, gt=gt)